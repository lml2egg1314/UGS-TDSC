
function flag = pgm2jpeg(QF)
    
  cover_dir = '/data1/lml/watermarking/BB-cover-resample-256'
  jpeg_dir = sprintf('%s-jpeg-%d', cover_dir, QF)
    if not(exist(jpeg_dir, 'dir'))
        mkdir(jpeg_dir)
    end

    parfor index = 1:20000

%         if index <= 10000
%             coverPath = [BossbassDir, '/', num2str(index), '.pgm'];
%         else
%             coverPath = [BowsDir, '/', num2str(index-10000), '.pgm'];
%         end
        coverPath = [cover_dir, '/', num2str(index), '.pgm'];
        stegoPath = [jpeg_dir, '/', num2str(index), '.jpg'];
%         costPath = [cost_dir, '/', num2str(index), '.mat'];
%         probPath = [prob_dir, '/', num2str(index), '.mat'];
        
    



    	%% Get embedding costs
    	% inicialization
    	cover = imread(coverPath);
        
    	imwrite(cover, stegoPath, 'quality', QF);

%     	save_cost(rhoP1, rhoM1, costPath);
%         save_prob_map(prob_map, probPath);

    end



    	flag = 'Finish';

end


function save_cost(rhoP1, rhoM1, costPath)
  	save(costPath, 'rhoP1', 'rhoM1');
end

function save_prob_map(prob_map, probPath)
    save(probPath, 'prob_map');
end


function [stego, prob_map] = embed_cmd_hill(cover, payload)

  params.w = 9;
  r_block = 2;
  c_block = 2;

  stego = f_sim_related_embed(cover, r_block, c_block, payload, params);

  hill_cost = f_cal_cost_HILL(cover);
  [tmp_stego, prob_map] = f_emb_filter(cover, hill_cost, payload, params);

end



function [Fstego] = f_sim_related_embed(cover,row_block,col_block,part_payload,params)

  cover=double(cover);
  stego=cover;%首先将携密图像等于载体图�???
  [k,l]=size(cover);
  trans_stego=zeros(k+2,l+2);%为了处理边界而做的填充处�???
  trans_cover=zeros(k+2,l+2);
  trans_stego(2:end-1,2:end-1)=stego;
  trans_cover(2:end-1,2:end-1)=cover;
  idex=1;
  for i=1:row_block
       if idex==1
            for j=1:col_block
                 HILL_cost=f_cal_cost_HILL(stego);
                 part_cost=HILL_cost(i:row_block:end,j:col_block:end);%取出部分cost�???
                 part_cover=cover(i:row_block:end,j:col_block:end);%取出载体图像的部�???
                 % params.seed=(i-1)*col_block+j;
                 if (i==1 && j==1)
                     part_stego = f_emb_filter(part_cover,part_cost,part_payload, params);%部分嵌入
                     stego(i:row_block:end,j:col_block:end)=part_stego; %更新图像
                     trans_stego(1+i:row_block:end-1,1+j:col_block:end-1)=part_stego;%
                 else
                     change=f_cla_diff_fromNeighbor(trans_stego,trans_cover,i,j,row_block,col_block);
                     part_stego = f_emb_filter2(part_cover,change,part_cost,part_payload,params);%部分嵌入
                     stego(i:row_block:end,j:col_block:end)=part_stego; %更新图像
                     trans_stego(1+i:row_block:end-1,1+j:col_block:end-1)=part_stego;
                 end
             end
             idex=0;
         else
             j=col_block;
             while (j>0)
                 % params.seed=(i-1)*col_block+j;
                 HILL_cost=f_cal_cost_HILL(stego);
                 part_cost=HILL_cost(i:row_block:end,j:col_block:end);%取出部分cost�???
                 part_cover=cover(i:row_block:end,j:col_block:end);%取出载体图像当前用于嵌入的部�???
                 change=f_cla_diff_fromNeighbor(trans_stego,trans_cover,i,j,row_block,col_block);
                 part_stego = f_emb_filter2(part_cover,change,part_cost,part_payload, params);%部分嵌入
                 stego(i:row_block:end,j:col_block:end)=part_stego; %更新图像
                 trans_stego(1+i:row_block:end-1,1+j:col_block:end-1)=part_stego;
                 j=j-1;
             end
             idex=1;
        end
  end
  Fstego=stego;

end


function [cost,r] = f_cal_cost_HILL(cover)

  %Get filter
  HF=[-1 2 -1;2 -4 2;-1 2 -1];
  H2 =  fspecial('average',[3 3]);
  %% Get cost
  cover=double(cover);
  sizeCover=size(cover);
  padsize=max(size(HF));
  coverPadded = padarray(cover, [padsize padsize], 'symmetric');% add padding
  R = conv2(coverPadded,HF, 'same');%mirror-padded convolution
  W=conv2(abs(R),H2,'same');
  % correct the W shift if filter size is even
   if mod(size(HF, 1), 2) == 0, W = circshift(W, [1, 0]); end;
   if mod(size(HF, 2), 2) == 0, W = circshift(W, [0, 1]); end;
    % remove padding
   W = W(((size(W, 1)-sizeCover(1))/2)+1:end-((size(W, 1)-sizeCover(1))/2), ((size(W, 2)-sizeCover(2))/2)+1:end-((size(W, 2)-sizeCover(2))/2));
   r=W;
   cost=1./(W+10^(-10));
   wetCost = 10^10;
  % compute embedding costs \rho
  rhoA = cost;
  rhoA(rhoA > wetCost) = wetCost; % threshold on the costs
  rhoA(isnan(rhoA)) = wetCost; % if all xi{} are zero threshold the cost

  HW =  fspecial('average', [15, 15]) ;
  cost = imfilter(rhoA, HW ,'symmetric','same');

end


function [stegoB,pChange] = f_emb_filter(cover, rho, payload, params)

  wetCost = 10^10;
  %% Get embedding costs
  % inicialization
  cover = double(cover);
  % seed = params.seed; %% seed for location selection
  rhoP1 = rho;
  rhoM1 = rho;
  rhoP1(cover==255) = wetCost; % do not embed +1 if the pixel has max value
  rhoM1(cover==0) = wetCost; % do not embed -1 if the pixel has min value
  % [stegoB,pChangeP1,pChangeM1]= f_EmbeddingSimulator_seed(cover, rhoP1, rhoM1, floor(payload*numel(cover)), seed);
  [stegoB,pChangeP1,pChangeM1]= f_EmbeddingSimulator_seed(cover, rhoP1, rhoM1, floor(payload*numel(cover)));
  pChange=pChangeP1+pChangeM1;

end


function [stegoB] = f_emb_filter2(cover,pixel_change, rho, payload, params)

  wetCost = 10^10;
  %% Get embedding costs
  % inicialization
  cover = double(cover);
  % seed = params.seed; %% seed for location selection
  rhoP1 = rho;
  rhoM1 = rho;
  rhoP1(cover==255) = wetCost; % do not embed +1 if the pixel has max value
  rhoM1(cover==0) = wetCost; % do not embed -1 if the pixel has min value
  rhoP1(pixel_change==1)=(rhoP1(pixel_change==1))./params.w;
  rhoM1(pixel_change==-1)=((rhoM1(pixel_change==-1)))./params.w;
  % stegoB = f_EmbeddingSimulator_seed(cover, rhoP1, rhoM1, floor(payload*numel(cover)), seed);
  stegoB = f_EmbeddingSimulator_seed(cover, rhoP1, rhoM1, floor(payload*numel(cover)));

end


function [y pChangeP1 pChangeM1] = f_EmbeddingSimulator_seed(x, rhoP1, rhoM1, m, seed)
%% --------------------------------------------------------------------------------------------------------------------------
% Embedding simulator simulates the embedding made by the best possible ternary coding method (it embeds on the entropy bound).
% This can be achieved in practice using "Multi-layered  syndrome-trellis codes" (ML STC) that are asymptotically aproaching the bound.
    n = numel(x);
    lambda = calc_lambda(rhoP1, rhoM1, m, n);
    pChangeP1 = (exp(-lambda .* rhoP1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
    pChangeM1 = (exp(-lambda .* rhoM1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
    % if nargin == 5
    %     RandStream.setGlobalStream(RandStream('mt19937ar','seed',seed));
    % else
    %     RandStream.setGlobalStream(RandStream('mt19937ar','Seed',sum(100*clock)));
    % end


    RandStream.setGlobalStream(RandStream('mt19937ar','Seed',sum(100*clock)));

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
        iterations = 0;
        alpha = double(message_length)/n;
        % limit search to 30 iterations
        % and require that relative payload embedded is roughly within 1/1000 of the required relative payload
        while  (double(m1-m3)/n > alpha/1000.0 ) && (iterations<300)
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
%         disp(iterations);
%         disp([message_length,  m2])
    end

    function Ht = ternary_entropyf(pP1, pM1)
        p0 = 1-pP1-pM1;
        P = [p0(:); pP1(:); pM1(:)];
        H = -((P).*log2(P));
        H((P<eps) | (P > 1-eps)) = 0;
        Ht = sum(H);
    end

end


function diff= f_cla_diff_fromNeighbor(paddStego,paddCover,rowIndex,colIndex,row_block,col_block)

  %%% 该函数的目的是统计当前即将进行信息嵌入像素的四邻
  %%% 域的+1 �???-1的个数�?�如�???+1多于-1 则将diff 置为+1，如�???+1
  %%% �???-1的数目相同则将diff，最后如�???+1少于-1，则diff�???-1�???
   Rightdiff=paddStego(1+rowIndex:row_block:end-1,2+colIndex:col_block:end)-paddCover(1+rowIndex:row_block:end-1,2+colIndex:col_block:end);
   Updiff=paddStego(rowIndex:row_block:end-2,1+colIndex:col_block:end-1)-paddCover(rowIndex:row_block:end-2,1+colIndex:col_block:end-1);
   Leftdiff=paddStego(1+rowIndex:row_block:end-1,colIndex:col_block:end-2)-paddCover(1+rowIndex:row_block:end-1,colIndex:col_block:end-2);
   Downdiff=paddStego(2+rowIndex:row_block:end,1+colIndex:col_block:end-1)-paddCover(2+rowIndex:row_block:end,1+colIndex:col_block:end-1);
   [k,l]=size(Rightdiff);
   diff=zeros(k,l);
   sum_matrix=Rightdiff+Updiff+Downdiff+Leftdiff;
   diff(sum_matrix>0)=1;
   diff(sum_matrix<0)=-1;

end

function [rhoP1, rhoM1] = hill_cost(cover, payload)

    cover = double(cover);
    wetCost = 10^8;
    [k,l] = size(cover);

    % compute embedding costs \rho	
    %Get filter
    HF1 = [-1, 2, -1; 2, -4, 2; -1, 2, -1];
    H2 = fspecial('average',[3 3]);

    %% Get cost
    sizeCover = size(cover);
    padsize = max(size(HF1));
    coverPadded = padarray(cover, [padsize padsize], 'symmetric');% add padding

    R1 = conv2(coverPadded, HF1, 'same');%mirror-padded convolution
    W1 = conv2(abs(R1), H2, 'same');

    if mod(size(HF1, 1), 2) == 0, W1= circshift(W1, [1, 0]); end;
    if mod(size(HF1, 2), 2) == 0, W1 = circshift(W1, [0, 1]); end;

    W1 = W1(((size(W1, 1) - sizeCover(1)) / 2) + 1 : end - ((size(W1, 1) - sizeCover(1)) / 2), ((size(W1, 2) - sizeCover(2)) / 2) + 1 : end - ((size(W1, 2) - sizeCover(2)) / 2));
    rho=1 ./ (W1 + 10 ^ (-10));

    HW =  fspecial('average', [15 15]);
    cost = imfilter(rho, HW , 'symmetric', 'same');
    rho = cost;

    % adjust embedding costs
    rho(rho > wetCost) = wetCost; % threshold on the costs
    rho(isnan(rho)) = wetCost; % if all xi{} are zero threshold the cost
    rhoP1 = rho;
    rhoM1 = rho;
    rhoP1(cover==255) = wetCost; % do not embed +1 if the pixel has max value
    rhoM1(cover==0) = wetCost; % do not embed -1 if the pixel has min value

end

