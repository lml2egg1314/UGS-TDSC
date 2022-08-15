function flag = gen_dct(params)

    addpath(fullfile('JPEG_Toolbox'));
    BB_Dir = params.cover_dir;
    stego_dir = params.output_stego_dir;
    stego_dct_dir = sprintf('%s-dct',stego_dir);


    listNum = params.listNum;
    if not(exist(stego_dct_dir,'dir'))
        mkdir(stego_dct_dir)
    end

   
%     if not(exist(Disdir,'dir'))
%         mkdir(Disdir)
%     end
    


    %% hyperparams
    IMAGE_SIZE = 256;
    RANDOM_NUM = 200;
    ALPHA = 2;
    L_P = 1.0;
    S_P = 0.01;


 
    
    
    
    %% load test index list
%     indexListPath = ['./index_list/', num2str(listNum), '/alaska_4w_test_list.mat'];
%     IndexList = load(indexListPath);
%     step2_list = IndexList.step2_list;

    indexListPath = ['../index_list/', num2str(listNum), '/alaska_list.mat'];
    IndexList = load(indexListPath);
    training_set = IndexList.step2_train;
    test_set = IndexList.step2_test;
    step2_list = IndexList.step2_list;
    len = length(step2_list);
    
%     params_mat = load(Disdir);
%     params = params_mat.params;
%     min_size = min(size(params));
    
    parfor index_it = 1:len
        index = step2_list(index_it);
        index_str = sprintf('%05d',index);

        %% load data   
 
        stego_path = [stego_dir, '/', index_str, '.jpg'];
        stego_dct_path = [stego_dir, '/', index_str, '.mat'];
       
        stego_struct = jpeg_read(stego_path);
        S_COEFFS = stego_struct.coef_arrays{1};
        save_dct2(S_COEFFS, stego_dct_path);

        
        
                    
    end
    

    flag = 'Finish';

end




function save_dct2(S_COEFFS, dct_path)
    save(dct_path, 'S_COEFFS');
end