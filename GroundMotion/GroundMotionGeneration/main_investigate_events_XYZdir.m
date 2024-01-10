clear;
clc;
close;

path = load('events_Y.mat');
all_path = path.file_list;
path_all_events = [];

statictic_info = [];
invalid_case= [];
for i = 1:size(all_path,1)
    for j = 1:size(all_path,2)
        cell = all_path(i,j);
        if ~isempty(cell{1})
            path_all_events = [path_all_events;all_path(i,j)];
        end
    end
end

for k = 1:length(path_all_events)
    a = fns_import_time_data(path_all_events{k},'txt',[0,50]);
    [b,Fs] = fns_fft_data(a,100,false,false);
    if Fs/2 ==100     
        disp(num2str(k));
        GMG = cls_GM_generator(a, 100);
        
        [~,FRF_info,Time_info,GM_info] = GMG.generateTimeNonStaPesudoGMbyFit("Hu_S0",[200,0.01,0.5],[],[],false);   

        if FRF_info(1) == -1
            invalid_case = [invalid_case;k];            
        else
            Time_info(3) = (Time_info(3) - Time_info(2)) / (Time_info(4) - Time_info(2));
            Time_info(4) = (Time_info(4) - Time_info(2));
            Time_info(2) = (Time_info(2) - Time_info(1));
            statictic_info = [statictic_info ; [path_all_events(k),FRF_info,Time_info,GM_info]];
        end
       
    else
        disp(num2str(k));
        invalid_case = [invalid_case;k];
    end
    disp('---------------------------------------------------------------');
end

save('Statistic_info_Y.mat','statictic_info','-mat');


