
function [S_STRUCT, prob_map] = UERD(cover_path, payload)

C_STRUCT = jpeg_read(cover_path);
C_COEFFS = C_STRUCT.coef_arrays{1};
dct_coef2 = C_COEFFS;

[n1, n2] = size(dct_coef2);

q_tab = C_STRUCT.quant_tables{1};

dct_coef2(1:8:end,1:8:end) = 0;

q_tab(1,1) = 0.5*(q_tab(2,1)+q_tab(1,2));
q_matrix = repmat(q_tab,[floor(n1 / 8), floor(n2 / 8)]);

%%% energy of each block
%             fun = @(block_struct) sum(sum(abs(q_tab.*block_struct.data)))*ones(8);
%             J = blockproc(dct_coef2,[8 8],fun);
dct_coef2 = im2col(q_matrix.*dct_coef2,[8 8],'distinct');
J2 = sum(abs(dct_coef2));
J = ones(64,1)*J2;
J = col2im(J,[8 8], [n1, n2], 'distinct');


% decide = q_matrix./J; % version 1

pad_size = 8;
im2 = padarray(J,[pad_size pad_size],'symmetric'); % energies of eight-neighbor blocks
size2 = 2*pad_size;
im_l8 = im2(1+pad_size:end-pad_size,1:end-size2);
im_r8 = im2(1+pad_size:end-pad_size,1+size2:end);
im_u8 = im2(1:end-size2,1+pad_size:end-pad_size);
im_d8 = im2(1+size2:end,1+pad_size:end-pad_size);
im_l88 = im2(1:end-size2,1:end-size2);
im_r88 = im2(1+size2:end,1+size2:end);
im_u88 = im2(1:end-size2,1+size2:end);
im_d88 = im2(1+size2:end,1:end-size2);

decide = q_matrix./(J+0.25*(im_l8+im_r8+im_u8+im_d8)+0.25*(im_l88+im_r88+im_u88+im_d88)); % version 2

% decide = decide/min(min(decide));

rhoM1 = decide;
rhoP1 = decide;

wetConst = 10^13;

rhoP1(rhoP1 > wetConst) = wetConst;
rhoP1(isnan(rhoP1)) = wetConst;
rhoP1(C_COEFFS > 1023) = wetConst;

rhoM1(rhoM1 > wetConst) = wetConst;
rhoM1(isnan(rhoM1)) = wetConst;
rhoM1(C_COEFFS < -1023) = wetConst;

nzAC = nnz(C_COEFFS)-nnz(C_COEFFS(1:8:end,1:8:end));

%% Embedding simulation
[S_COEFFS, prob_map] = EmbeddingSimulator(C_COEFFS, rhoP1, rhoM1, round(payload * nzAC));

S_STRUCT = C_STRUCT;
S_STRUCT.coef_arrays{1} = S_COEFFS;

function [y, prob_map] = EmbeddingSimulator(x, rhoP1, rhoM1, m)

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

end
