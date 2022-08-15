
addpath('JPEG_Toolbox')
addpath('jpegtbx');

file_num = 10000;
QF = params.QF;
base_dir = params.base_dir;
listNum = params.listNum;
payload = params.payload;
jpeg_cover_dir = params.cover_dir;

stego_dir = params.output_stego_dir;
extract_and_classify(QF, jpeg_cover_dir, stego_dir, params, payload);

function extract_and_classify(QF, jpeg_cover_dir, stego_dir, params, payload)
    
    total_start = tic;
    
    dataset_division_array_path = './dataset_division_array.mat';
    dataset_division_array_mat = load(dataset_division_array_path);
    dataset_division_array = dataset_division_array_mat.dataset_division_array;

    division_num = params.listNum;
    param_name = params.param_name;
    dataset_division = dataset_division_array{division_num};
    training_set = dataset_division.training_set;
    test_set = dataset_division.test_set;
    
    cover_set = {};
    stego_set = {};

    for index = 1 : 5000
        cover_set{end + 1} = sprintf('%s/%d.jpg', jpeg_cover_dir, training_set(index));
        stego_set{end + 1} = sprintf('%s/%d.jpg', stego_dir, training_set(index));
    end
    for index = 1 : 5000
        cover_set{end + 1} = sprintf('%s/%d.jpg', jpeg_cover_dir, test_set(index));
        stego_set{end + 1} = sprintf('%s/%d.jpg', stego_dir, test_set(index));
    end
   
    cover_features_path = sprintf('/data/lml/jpeg_test/gfr_cover_features_ln%d_QF%d.mat',params.listNum, QF);
    if params.start == 0
        fprintf('Extract GFR features ---- \n');
        cover_features = extract_gfr(cover_set, QF);
        save(cover_features_path, 'cover_features');
    else
        cover_features_mat = load(cover_features_path);
        cover_features = cover_features_mat.cover_features;
    end
        
     
        fprintf('Extract  GFR features ----- \n');
        stego_features = extract_gfr(stego_set, QF);

        test_acc = ensemble_classify(cover_features, stego_features, training_set, test_set);

        total_end = toc(total_start);

        fprintf('GFR and ensemble results ----- \n')
        fprintf('Test accuracy for # %s_%d_%s: %.4f\n', params.sp_dir, params.listNum, param_name, test_acc);

        file_id = fopen('acc_log_jpeg_comments.txt','a');
        fprintf(file_id,'GFR_%s_%s_%d: %.4f\n', params.sp_dir, param_name, params.listNum, test_acc);
        fclose(file_id);

        fprintf('Total time: %.2f seconds. \n', total_end);
        fprintf('------------------------- \n')
    end



function gfr_features = extract_gfr(image_set, QF)
    extract_start = tic;
    file_num = length(image_set);
    gfr_features = zeros(file_num, 17000);
 
    parfor i = 1:file_num
        image_item = image_set{i};
%         jpeg_path = [jpeg_dir, num2str(i), '.jpg'];
        %j_struct = jpeg_read(jpeg_path);
       
        %j_f = DCTR(j_struct, QF);
        j_f = GFR(image_item, 32, QF);
        
        gfr_features(i, :) = j_f;
        
    end
    extract_end = toc(extract_start);

    fprintf('GFR extracted %d cover images in %.2f seconds, in average %.2f seconds per image. \n\n', file_num, extract_end, extract_end / file_num);


end


function [test_acc] = ensemble_classify(cover_features, stego_features, training_set, testing_set)

  train_cover = cover_features(1:5000, :);
  train_stego = stego_features(1:5000, :);

  test_cover = cover_features(5001:10000, :);
  test_stego = stego_features(5001:10000, :);

  settings = struct('verbose', 2);

  train_start = tic;

  fprintf('Ensemble train start ----- \n');

  [trained_ensemble,results] = ensemble_training(train_cover, train_stego, settings);

  train_end = toc(train_start);


  fprintf('\n');

  test_start = tic;

  fprintf('Ensemble test start ----- \n');

  test_results_cover = ensemble_testing(test_cover, trained_ensemble);
  test_results_stego = ensemble_testing(test_stego, trained_ensemble);

  test_end = toc(test_start);


  % Predictions: -1 stands for cover, +1 for stego
  false_alarms = sum(test_results_cover.predictions ~= -1);
  missed_detections = sum(test_results_stego.predictions ~= +1);

  num_testing_samples = size(test_cover, 1) + size(test_stego, 1);

  testing_error = (false_alarms + missed_detections) / num_testing_samples;

  fprintf('Train time: %.2f seconds, Test time: %.2f seconds. \n\n', train_end, test_end);

  test_acc = 1 - testing_error;

end

