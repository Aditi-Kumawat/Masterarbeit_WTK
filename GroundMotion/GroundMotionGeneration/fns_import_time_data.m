function data = fns_import_time_data(file_path,data_type,time_cut_sec, plot_)
    data_type_list = {'txt','mat'};
    if ~ismember(data_type, data_type_list)
        fprintf('ERROR! Wrong input type, please input one of the type: [%s]\n', strjoin(data_type_list, ', '));
    end

    if strcmp(data_type,'txt') 
        data = readtable(file_path, 'ReadVariableNames',...
                false, 'HeaderLines', 5);
        time_ = data.Var1;
        ampl = data.Var2;
        %Make sure Acceleration at t=0 is 0, otherwise it might acumulate when 
        %integral to velocity and disp 
        
        if ~isempty(time_cut_sec)
            time = time_;
            time_cut_before = time_cut_sec(1);
            %time = time_(time_>=time_cut_before);
            %time = time - time_cut_before;
            %ampl = ampl(time_>=time_cut_before);

            time_cut_after = time_cut_sec(2);
            ampl(time>time_cut_after & abs(ampl)>= max(abs(ampl))/1000)...
                = ampl(time>time_cut_after & abs(ampl)>= max(abs(ampl))/1000)/1000;
            ampl(time<=time_cut_before & abs(ampl)>= max(abs(ampl))/1000)...
                = ampl(time<=time_cut_before & abs(ampl)>= max(abs(ampl))/1000)/1000;
            
        else
            time = time_;
        end
        ampl = ampl - mean(ampl);
        ampl(1) = 0;
        data = table(time,ampl);
        
    end


    if nargin < 4
        plot_ = false;  
    end

    if plot_ 
        figure;
        plot(time,ampl);
        title('Import Data');
        xlabel('time');
        ylabel('amplitude');
        grid on;
    end
end