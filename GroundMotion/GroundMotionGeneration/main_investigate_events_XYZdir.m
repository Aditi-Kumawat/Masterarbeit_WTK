clear;
clc;
close;

path = load('events_X.mat');
all_path = path.file_list;
path_all_events = [];

statictic_info_type1 = [];
statictic_info_type2 = [];
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
        
        [~,FRF_info,Time_info,GM_info] = GMG.generateTimeNonStaPesudoGMbyFit("Hu_S0",[200,0.01,0.1,GMG.PGA],[]);   

        if FRF_info(1) == -1
            invalid_case = [invalid_case;k];  
        else
            Time_info(3) = (Time_info(3) - Time_info(2)) / (Time_info(4) - Time_info(2));
            Time_info(4) = (Time_info(4) - Time_info(2));
            Time_info(2) = (Time_info(2) - Time_info(1));
            if FRF_info(3) >= 0.9
                GM_type = 1;
                statictic_info_type2 = [statictic_info_type2 ; [path_all_events(k),FRF_info,Time_info,GM_info,GM_type]];
            else
                GM_type = 0;
                statictic_info_type1 = [statictic_info_type1 ; [path_all_events(k),FRF_info,Time_info,GM_info,GM_type]];
            end
            
        end
       
    else
        disp(num2str(k));
        invalid_case = [invalid_case;k];
    end
    disp('---------------------------------------------------------------');
end


%a = fns_import_time_data(statictic_info_type2{1},'txt',[0,50]);
%[b,Fs] = fns_fft_data(a,100,false,true);
%save('Statistic_info_type1_X.mat','statictic_info_type1','-mat');
%save('Statistic_info_type2_X.mat','statictic_info_type2','-mat');

