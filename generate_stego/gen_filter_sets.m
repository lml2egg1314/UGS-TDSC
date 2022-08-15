function flag = gen_filter_sets(params)

    addpath(fullfile('JPEG_Toolbox'));
    BB_Dir = params.cover_dir;
   
    filter_dir = params.filter_dir

    
    if not(exist(filter_dir,'dir'))
        mkdir(filter_dir);
    end
    
  
    
    parfor index = 1:20000
        

        %% load data
   
        coverPath = [BB_Dir, '/', num2str(index), '.jpg'];
        
        filter_path = [filter_dir, '/', num2str(index), '.mat'];
        
        %% load cover hill stego   
        cover_struct = jpeg_read(coverPath);
        cover_coeffs = cover_struct.coef_arrays{1};
        cover_quant = cover_struct.quant_tables{1};

            
        %% Calculate 2 filter of image cover residual & juni stego residual & distance between them
        f_n = 3;
        m = 4;
        
        fun = @(x) idct2(x.data .* cover_quant);
        cover_spatial = blockproc(cover_coeffs, [8 8], fun);
      

        col_1 = im2col(cover_spatial, [1, f_n], 'sliding');
        col_2 = im2col(cover_spatial', [1, f_n], 'sliding');
        col = cat(2, col_1, col_2);
        neighbor = cat(1, col(1 : floor(1 * f_n / 2), :), col(floor(1 * f_n / 2) + 2 : 1 * f_n, :));
        target = col(floor(1 * f_n / 2) + 1, :);

        sol = lsqlin(neighbor', target);

        base_f = -ones(1, f_n);
        base_f(1 : floor(1 * f_n / 2)) = sol(1 : floor(1 * f_n / 2));
        base_f(floor(1 * f_n / 2) + 2 : 1 * f_n) = sol(floor(1 * f_n / 2) + 1 : 1 * f_n - 1);

        f_array = {
          padarray(base_f, [floor(f_n / 2), 0]),
          padarray(base_f', [0, floor(f_n / 2)]),
        };
    
        save_filter(filter_path, f_array)
    
              
    end

    flag = 'Finish';

end


function save_filter(filter_path, filter)
    save(filter_path, 'filter')
end




function save_cost(best_cost_p1, best_cost_m1, costPath)
    
    rhoP1 = best_cost_p1;
    rhoM1 = best_cost_m1;
    save(costPath, 'rhoP1', 'rhoM1');
    
end



function save_dis(disPath, dis_list, p1_list, p2_list)
    
    save(disPath, 'dis_list', 'p1_list', 'p2_list');

    
end



function [pre_rhoP1, pre_rhoM1] = load_cost(preCostPath)

    Pre_Rho = load(preCostPath);
    pre_rhoP1 = Pre_Rho.rhoP1;
    pre_rhoM1 = Pre_Rho.rhoM1;

end





function [grad, pred] = load_grad(preGradPath)

    Grad = load(preGradPath);
    grad = Grad.cover_grad;
    pred = Grad.pred;


end





%% --------------------------------------------------------------------------------------------------------------------------
% Embedding simulator simulates the embedding made by the best possible ternary coding method (it embeds on the entropy bound). 
% This can be achieved in practice using "Multi-layered  syndrome-trellis codes" (ML STC) that are asymptotically aproaching the bound.
function [y] = EmbeddingSimulator(x, rhoP1, rhoM1, m)

    x = double(x);
    n = numel(x);

    lambda = calc_lambda(rhoP1, rhoM1, m, n);
    pChangeP1 = (exp(-lambda .* rhoP1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
    pChangeM1 = (exp(-lambda .* rhoM1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));

    randChange = rand(size(x));
    y = x;
    y(randChange < pChangeP1) = y(randChange < pChangeP1) + 1;
    y(randChange >= pChangeP1 & randChange < pChangeP1+pChangeM1) = y(randChange >= pChangeP1 & randChange < pChangeP1+pChangeM1) - 1;

    prob_map = pChangeP1 + pChangeM1;

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
        pP1 = pP1(:);
        pM1 = pM1(:);
        Ht = -(pP1.*log2(pP1))-(pM1.*log2(pM1))-((1-pP1-pM1).*log2(1-pP1-pM1));
        Ht(isnan(Ht)) = 0;
        Ht = sum(Ht);
    end

end
