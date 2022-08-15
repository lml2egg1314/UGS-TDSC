%addpath('jpegtbx');
base_dir = '/data/lml/Boss512/BossBase-1.01-cover/';
nan_rou_cof_dir = '/data/lml/Boss512/jpeg_error/nan_rou_cof_75/';
ori_signs_dir = '/data/lml/Boss512/jpeg_error/ori_signs_75/';
norm_signs_dir = '/data/lml/Boss512/jpeg_error/norm_signs_75/';

if ~exist(nan_rou_cof_dir,'dir')
    mkdir(nan_rou_cof_dir)
end

if ~exist(ori_signs_dir, 'dir')
    mkdir(ori_signs_dir)
end

if ~exist(norm_signs_dir, 'dir')
    mkdir(norm_signs_dir)
end

quanlity = 75;
t = jpeg_qtable(75);
file_num = 10000;

for i = 1:file_num
    pgm_image = [base_dir, num2str(i),'.pgm'];
    nan_rou_cof_path = [nan_rou_cof_dir, num2str(i), '.mat'];
    ori_signs_path = [ori_signs_dir, num2str(i), '.mat'];
    norm_signs_path = [norm_signs_dir, num2str(i), '.mat'];
    
    ori_cover = double(imread(pgm_image));
    ori_cof = bdct(ori_cover-128);
    nan_round_cof = nan_round_quantize(ori_cof, t);
    round_cof = quantize(ori_cof, t);
    errors = nan_round_cof - round_cof;
    ori_signs = sign(errors);
    
    errors(abs(errors)<10^-10) = 0;
    norm_signs = sign(errors);
    
    save(nan_rou_cof_path,'nan_round_cof','-v6');
    save(ori_signs_path, 'ori_signs', '-v6');
    save(norm_signs_path, 'norm_signs', '-v6');
end
    
    