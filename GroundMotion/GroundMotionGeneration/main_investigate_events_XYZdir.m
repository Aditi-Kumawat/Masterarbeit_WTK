clear;
clc;
close;

%path = load('events_Z.mat');
path = load('All_events_record_Y.mat');
%fit_type = 'Hu_S0_NoContraint';
fit_type = 'Hu_S0';
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
    a = fns_import_time_data(path_all_events{k}{1},'txt',[]);
    [b,Fs] = fns_fft_data(a,100,false,false);
    if Fs/2 ==100     
        disp(num2str(k));
        GMG = cls_GM_generator(a, 100);
        
        [~,FRF_info,Time_info,GM_info] = GMG.generateTimeNonStaPesudoGMbyFit(fit_type,[150,0.7,0.5,GMG.PGA],[]); 

        Time_info(3) = (Time_info(3) - Time_info(2)) / (Time_info(4) - Time_info(2));
        Time_info(4) = (Time_info(4) - Time_info(2));
        Time_info(2) = (Time_info(2) - Time_info(1));
            
        if FRF_info(1) == -1
            disp("Retry")
            %Rerun again
             [~,FRF_info,Time_info,GM_info] = GMG.generateTimeNonStaPesudoGMbyFit(fit_type,[150,0.7,0.5,GMG.PGA],[]); 
            Time_info(3) = (Time_info(3) - Time_info(2)) / (Time_info(4) - Time_info(2));
            Time_info(4) = (Time_info(4) - Time_info(2));
            Time_info(2) = (Time_info(2) - Time_info(1));
        end

        if FRF_info(1) == -1
            invalid_case = [invalid_case;k]; 
            disp(['FAIL case:',num2str(k),' ==============================='])
            continue;
        end

        % Z: 200
        if FRF_info(1) >= 200
            invalid_case = [invalid_case;k]; 
            continue;
        end
        
        % Z: 2
        if FRF_info(2) >= 2
            invalid_case = [invalid_case;k]; 
            continue;
        end

        % Z:7
        if FRF_info(3) >= 10
            invalid_case = [invalid_case;k]; 
            continue;
        end

%        if FRF_info(4) >= 0.01
%            invalid_case = [invalid_case;k]; 
%            continue;
%        end
%
%      if Time_info(1)>=10
%          invalid_case = [invalid_case;k]; 
%          continue;
%      end

%      if Time_info(2)>=10
%          invalid_case = [invalid_case;k]; 
%          continue;
%      end

%        if GM_info(2)>=80
%            invalid_case = [invalid_case;k]; 
%            continue;
%        end
        
        % Z: 2e-5
%        if GM_info(1)>= 2e-5
%            invalid_case = [invalid_case;k]; 
%            continue;
%        end
%
        if GM_info(1)>= 0.025
            invalid_case = [invalid_case;k]; 
            continue;
        end


        if strcmp(fit_type,"Hu_S0_NoContraint")
            if FRF_info(3) >= 0.99
                GM_type = 1;
            else
                GM_type = 0;
            end
            
            statictic_info_type2 = [statictic_info_type2 ; [path_all_events(k),FRF_info,Time_info,GM_info,GM_type]];
        
        else
            if FRF_info(3) >= 0.99
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
save('Statistic_info_type1_Y.mat','statictic_info_type1','-mat');
save('Statistic_info_type2_Y.mat','statictic_info_type2','-mat');

