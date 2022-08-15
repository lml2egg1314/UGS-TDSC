
addpath('jpegtbx');

QFs = [75, 95];

for i = 1:2
    
quality = QFs(i);
for j = 1:4
    payload = j/10;
input_jpeg_dir_tpl = '/data/lml/jpeg_test/juni_%.1f_%d/stego/';
output_jpeg_dir_tpl = '/data/lml/jpeg_test/juni_%.1f_%d/stego-non-rounded/';

input_jpeg_dir = sprintf(input_jpeg_dir_tpl, payload, quality);
output_jpeg_dir = sprintf(output_jpeg_dir_tpl, payload, quality);


if ~exist(output_jpeg_dir, 'dir')
    mkdir(output_jpeg_dir);
end





for index = 1 : 20000
% for index = 5818 : 5818

  input_jpeg_path = [input_jpeg_dir, num2str(index), '.jpg'];
  output_jpeg_path = [output_jpeg_dir, num2str(index), '.mat'];

  c_struct = jpeg_read(input_jpeg_path);

  output_image = dct2spatial(c_struct);

  img = output_image;


  save(output_jpeg_path, 'img', '-v6');

end

end
end

function [s_spatial] = dct2spatial(s_struct)

  s_coef = s_struct.coef_arrays{1};
  s_quant = s_struct.quant_tables{1};

  dequntized_s_coef = dequantize(s_coef, s_quant);
  s_spatial = ibdct(dequntized_s_coef, 8) + 128;

end
