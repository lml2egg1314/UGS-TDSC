function flag = juni_cost_embed(stego_dir, cost_dir, stego_dct_dir, payload, QF)
    Boss_dir = sprintf('/data/lml/jpeg_test/BossBase-1.01-cover-resample-256-jpeg-%d',QF);
    Bows_dir = sprintf('/data/lml/jpeg_test/BOWS2-cover-resample-256-jpeg-%d',QF);
    
    if not(exist(stego_dir,'dir'))
        mkdir(stego_dir)
    end

    if not(exist(cost_dir,'dir'))
        mkdir(cost_dir)
    end
    
    if not(exist(stego_dct_dir, 'dir'))
        mkdir(stego_dct_dir)
    end
    
%     if not(exist(cover_dct_dir, 'dir'))
%         mkdir(cover_dct_dir)
%     end
    wetConst = 10^13;
    sgm = 2^(-6);

    %% Get 2D wavelet filters - Daubechies 8
    % 1D high pass decomposition filter
    hpdf = [-0.0544158422, 0.3128715909, -0.6756307363, 0.5853546837, 0.0158291053, -0.2840155430, -0.0004724846, 0.1287474266, 0.0173693010, -0.0440882539, ...
            -0.0139810279, 0.0087460940, 0.0048703530, -0.0003917404, -0.0006754494, -0.0001174768];
    % 1D low pass decomposition filter
    lpdf = (-1).^(0:numel(hpdf)-1).*fliplr(hpdf);

    F{1} = lpdf'*hpdf;
    F{2} = hpdf'*lpdf;
    F{3} = hpdf'*hpdf;
    
    C_QUANT = jpeg_table(QF);
    %% Pre-compute impact in spatial domain when a jpeg coefficient is changed by 1
    spatialImpact = cell(8, 8);
    for bcoord_i=1:8
        for bcoord_j=1:8
            testCoeffs = zeros(8, 8);
            testCoeffs(bcoord_i, bcoord_j) = 1;
            spatialImpact{bcoord_i, bcoord_j} = idct2(testCoeffs)*C_QUANT(bcoord_i, bcoord_j);
        end
    end

        %% Pre compute impact on wavelet coefficients when a jpeg coefficient is changed by 1
    waveletImpact = cell(numel(F), 8, 8);
    for Findex = 1:numel(F)
        for bcoord_i=1:8
            for bcoord_j=1:8
                waveletImpact{Findex, bcoord_i, bcoord_j} = imfilter(spatialImpact{bcoord_i, bcoord_j}, F{Findex}, 'full');
            end
        end
    end
    parfor index = 1:20000

        if index <= 10000
            cover_path = [Boss_dir, '/', num2str(index), '.jpg'];
        else
            cover_path = [Bows_dir, '/', num2str(index-10000), '.jpg'];
        end
     
        stego_path = [stego_dir, '/', num2str(index), '.jpg'];
        stego_dct_path = [stego_dct_dir, '/', num2str(index), '.mat'];
        cost_path = [cost_dir, '/', num2str(index), '.mat'];
        
        %%
        C_SPATIAL = double(imread(cover_path));
        C_STRUCT = jpeg_read(cover_path);
        C_COEFFS = C_STRUCT.coef_arrays{1};
        

        

        %% Create reference cover wavelet coefficients (LH, HL, HH)
        % Embedding should minimize their relative change. Computation uses mirror-padding
        padSize = max([size(F{1})'; size(F{2})']);
        C_SPATIAL_PADDED = padarray(C_SPATIAL, [padSize padSize], 'symmetric'); % pad image

        RC = cell(size(F));
        for i=1:numel(F)
            RC{i} = imfilter(C_SPATIAL_PADDED, F{i});
        end

        [k, l] = size(C_COEFFS);

        nzAC = nnz(C_COEFFS)-nnz(C_COEFFS(1:8:end,1:8:end));
        rho = zeros(k, l);
        tempXi = cell(3, 1);

        %% Computation of costs
        for row = 1:k
            for col = 1:l
                modRow = mod(row-1, 8)+1;
                modCol = mod(col-1, 8)+1;

                subRows = row-modRow-6+padSize:row-modRow+16+padSize;
                subCols = col-modCol-6+padSize:col-modCol+16+padSize;

                for fIndex = 1:3
                    % compute residual
                    RC_sub = RC{fIndex}(subRows, subCols);
                    % get differences between cover and stego
                    wavCoverStegoDiff = waveletImpact{fIndex, modRow, modCol};
                    % compute suitability
                    tempXi{fIndex} = abs(wavCoverStegoDiff) ./ (abs(RC_sub)+sgm);
                end
                rhoTemp = tempXi{1} + tempXi{2} + tempXi{3};
                rho(row, col) = sum(rhoTemp(:));
            end
        end
    
        rhoM1 = rho;
        rhoP1 = rho;

        rhoP1(rhoP1 > wetConst) = wetConst;
        rhoP1(isnan(rhoP1)) = wetConst;
        rhoP1(C_COEFFS > 1023) = wetConst;

        rhoM1(rhoM1 > wetConst) = wetConst;
        rhoM1(isnan(rhoM1)) = wetConst;
        rhoM1(C_COEFFS < -1023) = wetConst;

        %% Embedding simulation
        [S_COEFFS, ~] = EmbeddingSimulator(C_COEFFS, rhoP1, rhoM1, round(payload * nzAC),0);

        S_STRUCT = C_STRUCT;
        S_STRUCT.coef_arrays{1} = S_COEFFS;
        
    	%% save stego and cost
    	jpeg_write(S_STRUCT, stego_path);
        save_dct2(S_COEFFS, stego_dct_path);
    	save_cost(rhoP1, rhoM1, cost_path);

    end



    flag = 'Finish';

end



function save_cost(rhoP1, rhoM1, costPath)
  	save(costPath, 'rhoP1', 'rhoM1');
end

function save_dct(C_COEFFS, dct_path)
    save(dct_path, 'C_COEFFS');
end

function save_dct2(S_COEFFS, dct_path)
    save(dct_path, 'S_COEFFS');
end

function save_prob(prob, prob_path)
    save(prob_path, 'prob');
end

function save_prob2(pP1, pM1, prob2_path)
    save(prob2_path, 'pP1', 'pM1');
end



%% --------------------------------------------------------------------------------------------------------------------------
% Embedding simulator simulates the embedding made by the best possible ternary coding method (it embeds on the entropy bound). 
% This can be achieved in practice using "Multi-layered  syndrome-trellis codes" (ML STC) that are asymptotically aproaching the bound.
function [y, prob, pChangeP1, pChangeM1] = EmbeddingSimulator(x, rhoP1, rhoM1, m, fixEmbeddingChanges)

    n = numel(x);   
    lambda = calc_lambda(rhoP1, rhoM1, m, n);
    pChangeP1 = (exp(-lambda .* rhoP1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
    pChangeM1 = (exp(-lambda .* rhoM1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
    prob = pChangeP1 + pChangeM1;
    if fixEmbeddingChanges == 1
        RandStream.setGlobalStream(RandStream('mt19937ar','seed',139187));
    else
        RandStream.setGlobalStream(RandStream('mt19937ar','Seed',sum(100*clock)));
    end
    randChange = rand(size(x));
    y = x;
    y(randChange < pChangeP1) = y(randChange < pChangeP1) + 1;
    y(randChange >= pChangeP1 & randChange < pChangeP1+pChangeM1) = y(randChange >= pChangeP1 & randChange < pChangeP1+pChangeM1) - 1;
    
    function lambda = calc_lambda(rhoP1, rhoM1, message_length, n)

        l3 = 1e+3;
        m3 = double(message_length + 1);
        iterations = 0;
        while m3 > message_length
            l3 = l3 * 2;
            pP1 = (exp(-l3 .* rhoP1))./(1 + exp(-l3 .* rhoP1) + exp(-l3 .* rhoM1));
            pM1 = (exp(-l3 .* rhoM1))./(1 + exp(-l3 .* rhoP1) + exp(-l3 .* rhoM1));
            m3 = ternary_entropyf(pP1, pM1);
            iterations = iterations + 1;
            if (iterations > 10)
                lambda = l3;
                return;
            end
        end        
        
        l1 = 0; 
        m1 = double(n);        
        lambda = 0;
        
        alpha = double(message_length)/n;
        % limit search to 30 iterations
        % and require that relative payload embedded is roughly within 1/1000 of the required relative payload        
        while  (double(m1-m3)/n > alpha/1000.0 ) && (iterations<30)
            lambda = l1+(l3-l1)/2; 
            pP1 = (exp(-lambda .* rhoP1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
            pM1 = (exp(-lambda .* rhoM1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
            m2 = ternary_entropyf(pP1, pM1);
    		if m2 < message_length
    			l3 = lambda;
    			m3 = m2;
            else
    			l1 = lambda;
    			m1 = m2;
            end
    		iterations = iterations + 1;
        end
    end
    
    function Ht = ternary_entropyf(pP1, pM1)
        p0 = 1-pP1-pM1;
        P = [p0(:); pP1(:); pM1(:)];
        H = -((P).*log2(P));
        H((P<eps) | (P > 1-eps)) = 0;
        Ht = sum(H);
    end
end
