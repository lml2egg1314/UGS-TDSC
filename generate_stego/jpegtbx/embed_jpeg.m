addpath('JPEG_Toolbox')

base_dir = '/data/lml';
image_size = 256;
datasets = {'Boss','BossBase-1.01',10000; 'Bows','BOWS2',10000; 'Szu','SZU',40000; 'Alaska', 'alaska',80005};

steganography = 'bet-hill';
payload = single(0.4);
% cover_dir_format = '%s/%s256/%s-cover-resample-256';
% stego_dir_format = '%s/%s256/%s-%s-%s-resample-256';
% prob_map_dir_format = '%s/%s256/%s-%s-%s-pro-map-resample-256';

cover_dir_format = '%s/%s256/%s-cover-resample-256-jpeg-75';
stego_dir_format = '%s/%s256/%s-%s-%s-resample-256-jpeg-75';
prob_map_dir_format = '%s/%s256/%s-%s-%s-pro-map-resample-256-jpeg-75';

for i = 1:length(datasets)-1
    

    cover_dir = sprintf(cover_dir_format, base_dir, datasets{i,1},datasets{i,2});
    stego_dir = sprintf(stego_dir_format, base_dir, datasets{i,1},datasets{i,2},steganography, num2str(payload));
    pro_map_dir = sprintf(prob_map_dir_format, base_dir, datasets{i,1},datasets{i,2},steganography, num2str(payload));
    if ~exist(stego_dir, 'dir')
        mkdir(stego_dir)
    end
    if ~exist(pro_map_dir, 'dir')
        mkdir(pro_map_dir)
    end
    
    parfor j=1:datasets{i,3}
        cover_path = [cover_dir, num2str(j), '.jpg'];
        stego_path = [stego_dir, num2str(j), '.jpg'];
        prob_map_path = [prob_map_dir, num2str(j), '.mat'];
        [S_STRUCT, prob_map] = embed_bet_hill(cover_path, payload);

        jpeg_write(S_STRUCT, stego_path);
        save_prob_map(prob_map_path, 'prob_map');
    end
  
end
function save_prob_map(prob_map, prob_map_path)
  save(prob_map_path, 'prob_map', '-v6');
end