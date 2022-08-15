
payload = params.payload;
steganography = params.steganography;

file_num = 40000;

extract_and_classify(steganography, payload, file_num, params);


function extract_and_classify(steganography, payload, file_num, params)
    total_start = tic;
    listNum = params.listNum;
    QF = params.QF;
    sp_dir = params.sp_dir;
    fprintf('Extract cover SRM features ----- \n');
    cover_features_path = sprintf('/data/lml/jpeg_alaska/gfr_cover_features_uerd_0.4_%d_ln%d.mat',QF, listNum)
    ori_stego_features_path = sprintf('/data/lml/jpeg_alaska/gfr_stego_original_features_%s_ln%d.mat',sp_dir, listNum);
    stego_features_path = sprintf('/data/lml/jpeg_alaska/adv/gfr_stego_ugs_features_%s_ln%d.mat',sp_dir, listNum)
    
%     srm_cover_features_path = sprintf('/data/lml/jpeg_alaska/gfr_cover_features_ln%s_QF%d.mat', num2str(params.listNum), params.QF);

    cover_features_mat = load(cover_features_path);
    cover_features = cover_features_mat.cover_features;
    fprintf('Steganography: %s ----- \n\n', steganography);
    fprintf('Extract stego SRM features ----- \n');
%     ori_stego_features_path = sprintf('/data/lml/jpeg_alaska/gfr_stego_original_features_%s_ln%d_QF%d.mat',steganography, listNum, QF);
%     stego_features_path = sprintf('/data/lml/jpeg_alaska/gfr_stego_adv_features_%s_ln%d_QF%d.mat',steganography, listNum, QF);
%     
%     srm_stego_features_path = sprintf('/data/lml/jpeg_alaska/gfr_stego_original_features_uerd_ln%s_QF%d.mat', num2str(params.listNum), params.QF);
%     save(srm_stego_features_path, 'stego_features', '-v7.3');
% %     srm_stego_features_path = sprintf('/data/lml/spa_alaska/%s/srm_stego_mae%s.mat',params.sp_dir, num2str(params.listNum));
%     srm_stego_features_path = sprintf('/data/lml/spa_alaska/srm_stego_original_%s_%s.mat',params.sp_dir, num2str(params.listNum));
% %     save(srm_original_stego_features_path, 'stego_features', '-v7.3');
    stego_features_mat = load(stego_features_path);
    stego_features = stego_features_mat.stego_features;





    test_acc = ensemble_classify(cover_features, stego_features);

    total_end = toc(total_start);
  
  
  
  fprintf('SRM and ensemble results ----- \n')
  fprintf('Test accuracy for # dnet-adv-%s-%s #: %.4f \n', params.sp_dir, num2str(params.listNum), test_acc);

  file_id = fopen('acc_log_alaska_new.txt','a');
  fprintf(file_id,'%s  dnet-adv-%s-%s: %.4f\n', datestr(now), params.sp_dir, num2str(params.listNum), test_acc);
  fclose(file_id);

  fprintf('Total time: %.2f seconds. \n', total_end);
  fprintf('------------------------- \n')

end




function [features] = srm_extract(image_set)

  srm_start = tic;

  file_num = length(image_set);
  feature_num = 34671;

  features = zeros(file_num, feature_num);
  parfor i = 1:file_num
    image_item = image_set(i);
    feature_item = SRM(image_item);

    feature_item = struct2cell(feature_item);
    feature_item = [feature_item{:}];

    features(i, :) = feature_item;
  end

  srm_end = toc(srm_start);

  fprintf('SRM extracted %d images in %.2f seconds, in average %.2f seconds per image. \n\n', numel(image_set), srm_end, srm_end / numel(image_set));


end


function [test_acc] = ensemble_classify(cover_features, stego_features)

  train_cover = cover_features(1:15000, :);
  train_stego = stego_features(1:15000, :);

  test_cover = cover_features(20001:40000, :);
  test_stego = stego_features(20001:40000, :);

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

