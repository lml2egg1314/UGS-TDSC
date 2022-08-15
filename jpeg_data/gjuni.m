% payload = 0.4;
addpath('JPEG_Toolbox');
addpath('jpegtbx');
QFs = [75, 95];
for k = 1:2
    QF = QFs(k);
    
    for i = 1:4
        payload = i/10;
        tic;
        disp(datestr(now));
        

%         flag = juni_cost_embed(payload, QF);
        flag = uerd_cost_embed(payload, QF);

        disp(flag);
        toc;
    end
end

exit;