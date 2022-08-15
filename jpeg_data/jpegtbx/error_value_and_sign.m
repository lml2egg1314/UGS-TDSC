
base_dir = '/data/lml/Boss256/BossBase-1.01-cover-resample-256/';
nan_rou_cof_dir = '/data/lml/jpeg_error/nan_rou_cof_75/';
signs_dir = '/data/lml/jpeg_error/signs_75/';
quanlity = 75;
t_75 = jpeg_qtable(75);
file_num = 10000;
for i = 1:file_num
    pgm_image = [base_dir, num2str(i),'.pgm'];
    nan_rou_cof_path = [nan_rou_cof_dir, num2str(i), '.mat'];
    signs_path = [signs_dir, num2str(i), '.mat'];
    image_data = double(imread(pgm_image));
    raw_cof = bdct(image_data);
    blksz = size(t_75);
    [v,r,c] = im2vec(raw_cof,blksz);
    
    nan_rou_cof = vec2im(v./repmat(qtable(:),1,size(v,2)),0,blksz,r,c);
   
    rou_cof = vec2im(round(v./repmat(qtable(:),1,size(v,2))),0,blksz,r,c);
    
    signs = sign(nan_rou_cof_ - rou_cof);
    
    save(nan_rou_cof_path,'nan_rou_cof'.'-v6');
    save(signs_path, 'signs', '-v6');
end
    
    