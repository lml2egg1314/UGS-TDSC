

 %% dir
 steganography_set = {'juni','uerd'};
 QFs = [75, 95];

 for ln = 1:3
     params.listNum = ln;
     for i = 1:2
        params.start = 0;
        steganography = steganography_set{1};
        QF = QFs(i);
        params.QF = QF;
        for j = 4
            payload = j/10;
            sp_dir = sprintf('%s_%s_%d', steganography, num2str(payload), QF);
            params.sp_dir = sp_dir;
            base_dir = sprintf('/data/lml/jpeg_test/%s', sp_dir);
            params.base_dir = base_dir;
            grad_dir = sprintf('%s/%s', base_dir, num2str(params.listNum));
            

            output_dir = sprintf('%s/output_UGS', grad_dir);

            params.payload = payload;
            params.steganography = steganography;
            params.IMAGE_SIZE = 256;
     %%           
            params.cover_dir = sprintf('/data/lml/jpeg_test/BB-cover-resample-256-jpeg-%d', QF);
            params.filter_dir = sprintf('/data/lml/jpeg_test/filter-sets-jpeg-%d', QF);
            params.stego_dir = sprintf('%s/stego', base_dir);

            params.cost_dir = sprintf('%s/cost', base_dir);
           
            params.grad_dir = sprintf('%s/cover_grad', grad_dir);
                   
            params.params_dir = sprintf('%s/UGS', grad_dir);

            GENERATE_NUM = 2;
            params.GENERATE_NUM = GENERATE_NUM;
            params.output_stego_dir = sprintf('%s/stego', output_dir);
            params.output_cost_dir = sprintf('%s/cost', output_dir);

            disp(params);
            tic;

            flag = gen_final_stego_multiplication(params);
           	toc;
%             gfr_and_ensemble_from_image;
            toc;
                       
        end
     end
 end


   