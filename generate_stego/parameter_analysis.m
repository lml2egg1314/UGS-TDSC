function flag = parameter_analysis(params)


    params_dir = params.params_dir;
    GENERATE_NUM  = params.GENERATE_NUM;

    listNum = params.listNum;
    
    parameter_path = sprintf('%s-%s.mat', params.sp_dir, num2str(listNum));

    %% load test index list
    indexListPath = ['./index_list/', num2str(listNum), '/test_list.mat'];
    IndexList = load(indexListPath);
    index_list = IndexList.index;
    len = length(index_list);
    
    parameters = zeros(2,len);
    parfor index_it = 1:len
        index = index_list(index_it);

        params_path = [params_dir, '/', num2str(index), '.mat'];
     
        
%         [grad, ~] = load_grad(preGradPath);
        [~, myp1, myp2, ~] = load_params(params_path, GENERATE_NUM);
      
        parameters(:,index_it) = [myp1, myp2];
                    
    end
    para = parameters;
    save(parameter_path, 'para');

    flag = 'Finish';

end


function [randChange, myp1, myp2, myalpha] = load_params(params_path, GENERATE_NUM)
params_mat  = load(params_path);
randChange = params_mat.randChange;
pred = params_mat.pred;
residual_dis = params_mat.residual_dis;

p1 = params_mat.p1;
p2 = params_mat.p2;
alpha = params_mat.alpha;

partial_pred = pred(1:GENERATE_NUM);
partial_dis = residual_dis(1:GENERATE_NUM);
partial_p1 = p1(1:GENERATE_NUM);
partial_p2 = p2(1:GENERATE_NUM);

select_index = partial_pred < 1;
if sum(select_index) == 0
    myp1 = 0;
    myp2 = 0;
    myalpha = 0;
    
else
    select_dis = partial_dis(select_index);
    select_p1 = partial_p1(select_index);
    select_p2 = partial_p2(select_index);


    [~, final_index] = min(select_dis);

    myp1 = select_p1(final_index);
    myp2 = select_p2(final_index);
    if numel(alpha) == 1
        myalpha = double(alpha);
    else
        select_alpha = alpha(select_index);
        myalpha = select_alpha(final_index);
    end
end
end


% 
% function [p1, p2] = load_params(disPath)
% params_mat  = load(disPath);
% dis_list = params_mat.dis_list;
% p1_list = params_mat.p1_list;
% p2_list = params_mat.p2_list;
% [~, index] = min(dis_list);
% p1 = p1_list(index);
% p2 = p2_list(index);
% end

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
function [y] = EmbeddingSimulator(x, rhoP1, rhoM1, m, randChange)

    x = double(x);
    n = numel(x);

    lambda = calc_lambda(rhoP1, rhoM1, m, n);
    pChangeP1 = (exp(-lambda .* rhoP1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
    pChangeM1 = (exp(-lambda .* rhoM1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));

%     randChange = rand(size(x));
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
