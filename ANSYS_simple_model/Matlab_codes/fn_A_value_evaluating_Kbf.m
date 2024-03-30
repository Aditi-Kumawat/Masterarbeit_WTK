%% Code refer to DIN4150-2
%% Guildline values for evaluating human exposure to vibration 
%test = find_A_value("ReinesWohngebiet","night")

function A_value = find_A_value_evaluating_Kbf(location,time)
    % A_value = [Au,Ao,Ar], please check DIN 4150-2 page7 for detail
    
    time_ = ["day","night"];
    loc_ = ["Indistriegebiet","Gewerbegebiet","Kerngebiet","ReinesWohngebiet","Special"];
    day = {[0.4,6.0,0.2]; [0.3,6.0,0.15];[0.2,0.4,0.1]; [0.15,3.0,0.07]; [0.1,3.0,0.05]};
    night = {[0.3,0.6,0.15]; [0.2,0.4,0.1];[0.15,0.3,0.07]; [0.1,0.2,0.05]; [0.1,0.15,0.05]};
    T = table(day,night, 'RowNames', loc_, 'VariableNames', ["Day","Night"]); 

    %initialize
    A_value = [0,0,0] ;

    if ismember(location,loc_)  
        if ismember(time,time_)
            if strcmp(time,time_(1)) %case = day
                A_value = T.Day(location);
                A_value = A_value{1};
            else %case = night
                A_value = T.Night(location);
                A_value = A_value{1};
            end   
        else
            disp("ERROR! wrong input of time varaible");
            disp("please input:");
            disp("  day");
            disp("  night");
        end
    else
        disp("ERROR! wrong input of location varaible");
        disp("please input:");
        disp("    Indistriegebiet");
        disp("    Gewerbegebiet");
        disp("    Kerngebiet");
        disp("    ReinesWohngebiet");
        disp("    Special");
    end
end

