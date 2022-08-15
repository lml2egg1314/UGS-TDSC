function flag = gen_final_stego_multiplication(params)

    addpath(fullfile('JPEG_Toolbox'));
    BB_Dir = params.cover_dir;
    stego_dir = params.output_stego_dir;
    cost_dir = params.output_cost_dir;
    pre_cost_dir = params.cost_dir;
    pre_stego_dir = params.stego_dir;
    pre_grad_dir = params.grad_dir;
    GENERATE_NUM = params.GENERATE_NUM;

    param_name = params.param_name;
%     parameter_path = sprintf('%s-%s-%s.mat', params.sp_dir, num2str(params.listNum), param_name);

    params_dir = params.params_dir;
%     Disdir = params.DisDir;
    payload = params.payload;
    list_num = params.listNum;
    if not(exist(stego_dir,'dir'))
        mkdir(stego_dir)
    end

    if not(exist(cost_dir,'dir'))
        mkdir(cost_dir)
    end
   
%     if not(exist(Disdir,'dir'))
%         mkdir(Disdir)
%     end
    


    %% hyperparams
    IMAGE_SIZE = 256;
%     RANDOM_NUM = 200;
%     ALPHA = 2;
%     L_P = 1.0;
%     S_P = 0.01;
  
    
    %% load test index list
    indexListPath = ['./index_list/', num2str(list_num), '/test_list.mat'];
    IndexList = load(indexListPath);
    index_list = IndexList.index;
    len = length(index_list);
    
%     params_mat = load(Disdir);
%     params = params_mat.params;
%     min_size = min(size(params));
%     parameters = zeros(2,len);
    parfor index_it = 1:len
        index = index_list(index_it);
        
%         if min_size == 1
%             myp1 = params(index_it);
%             myp2 = params(index_it);
%             alpha = ALPHA;
%         elseif min_size == 2
%             myp1 = params(index_it,1);
%             myp2 = params(index_it,2);
%             alpha = ALPHA;
%         else
%             myp1 = params(index_it,1);
%             myp2 = params(index_it,2);
%             alpha = params(index_it,3);
%         end

        %% load data   
        cover_path = [BB_Dir, '/', num2str(index), '.jpg'];       
        pre_cost_path = [pre_cost_dir, '/', num2str(index), '.mat'];
        pre_grad_path = [pre_grad_dir, '/', num2str(index), '.mat'];
        pre_stego_path = [pre_stego_dir, '/', num2str(index), '.jpg'];
        
        cost_path = [cost_dir, '/', num2str(index), '.mat'];
        StegoPath = [stego_dir, '/', num2str(index), '.jpg'];
        
        params_path = [params_dir, '/', num2str(index), '.mat'];
     
        
        [grad, ~] = load_grad(pre_grad_path);
        [randChange, myp1, myp2, myalpha] = load_params(params_path, GENERATE_NUM);
        % parameters(:,index_it) = [myp1, myp2];
%         if pred(2) == 1
%             copyfile(pre_cost_path, cost_path);
%             copyfile(pre_stego_path, StegoPath);
%             continue;
%         end
%        
        if myp1 * myp2 == 0
            copyfile(pre_cost_path, cost_path);
            copyfile(pre_stego_path, StegoPath);
            continue;
        end
        
        [pre_rhoP1, pre_rhoM1] = load_cost(pre_cost_path);
        
        sign_grad = sign(grad);
               
        %% load cover hill stego   
        cover_struct = jpeg_read(cover_path);
        cover_coeffs = cover_struct.coef_arrays{1};

        %% preprocessing of grad and cost
%         s = (sign_grad<0);
%         l = (sign_grad>0);
        se = (sign_grad<=0);
        le = (sign_grad>=0);
            
        temp_grad = grad;
        flat_grad = reshape(temp_grad,1,IMAGE_SIZE*IMAGE_SIZE);
        [x_grad, ~] = sort(abs(flat_grad));
        
        
        temp_pre_rhoP1 = pre_rhoP1;
        temp_pre_rhoM1 = pre_rhoM1;

        flat_pre_rhoP1 = reshape(temp_pre_rhoP1,1,IMAGE_SIZE*IMAGE_SIZE);
        [x_P1, ~] = sort(flat_pre_rhoP1);
        flat_pre_rhoM1 = reshape(temp_pre_rhoM1,1,IMAGE_SIZE*IMAGE_SIZE);
        [x_M1, ~] = sort(flat_pre_rhoM1);
        

        %% calculate abs_grad abs_cost
        % smaller p grad
        high_grad = x_grad(max(1,round(IMAGE_SIZE*IMAGE_SIZE*myp1)));
        abs_grad = (abs(temp_grad)>high_grad);

        % smaller p cost 
        num_P1 = x_P1(max(1, round(IMAGE_SIZE*IMAGE_SIZE*myp2)));
        small_pre_rhoP1 = (temp_pre_rhoP1<num_P1);
        num_M1 = x_M1(max(1, round(IMAGE_SIZE*IMAGE_SIZE*myp2)));
        small_pre_rhoM1 = (temp_pre_rhoM1<num_M1);

        abs_bool_p1 = logical(abs_grad .* small_pre_rhoP1 .* le);
        abs_bool_m1 = logical(abs_grad .* small_pre_rhoM1 .* se);

        rhoP1 = pre_rhoP1;
        rhoM1 = pre_rhoM1;
        rhoP1(abs_bool_p1) = myalpha * rhoP1(abs_bool_p1);
        rhoM1(abs_bool_m1) = myalpha * rhoM1(abs_bool_m1);
%         rhoP1 = pre_rhoP1 + (s*0 + myalpha*le) .* abs_bool_p1;
%         rhoM1 = pre_rhoM1 + (l*0 + myalpha*se) .* abs_bool_m1;

        %% Get embedding costs & stego
        % inicialization
        wetCost = 10^8;
        nzAC = nnz(cover_coeffs)-nnz(cover_coeffs(1:8:end,1:8:end));

        % adjust embedding costs
        rhoP1(rhoP1 > wetCost) = wetCost;
        rhoP1(isnan(rhoP1)) = wetCost;
        rhoP1(cover_coeffs > 1023) = wetCost;

        rhoM1(rhoM1 > wetCost) = wetCost;
        rhoM1(isnan(rhoM1)) = wetCost;
        rhoM1(cover_coeffs < -1023) = wetCost;

        stego_coeffs = EmbeddingSimulator(cover_coeffs, rhoP1, rhoM1, round(payload * nzAC), randChange);

        stego_struct = cover_struct;
        stego_struct.coef_arrays{1} = stego_coeffs;

        jpeg_write(stego_struct, StegoPath);
        % save_cost(rhoP1, rhoM1, cost_path)
        
                    
    end
    % para = parameters;
    % save(parameter_path, 'para');

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

function [randChange, myp1, myp2, myalpha] = load_params_tt(params_path, GENERATE_NUM)
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

% select_index = partial_pred < 1;
% if sum(select_index) == 0
%     myp1 = 0;
%     myp2 = 0;
%     myalpha = 0;
%     
% else
    select_dis = partial_dis;
    select_p1 = partial_p1;
    select_p2 = partial_p2;


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


function save_cost(best_cost_p1, best_cost_m1, cost_path)
    
    rhoP1 = best_cost_p1;
    rhoM1 = best_cost_m1;
    save(cost_path, 'rhoP1', 'rhoM1');
    
end



function save_dis(disPath, dis_list, p1_list, p2_list)
    
    save(disPath, 'dis_list', 'p1_list', 'p2_list');

    
end



function [pre_rhoP1, pre_rhoM1] = load_cost(pre_cost_path)

    Pre_Rho = load(pre_cost_path);
    pre_rhoP1 = Pre_Rho.rhoP1;
    pre_rhoM1 = Pre_Rho.rhoM1;

end





function [grad, pred] = load_grad(pre_grad_path)

    Grad = load(pre_grad_path);
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
