function data = fns_import_time_data(file_path,data_type, plot_)
    data_type_list = {'txt','mat'};
    if ~ismember(data_type, data_type_list)
        fprintf('ERROR! Wrong input type, please input one of the type: [%s]\n', strjoin(data_type_list, ', '));
    end

    if strcmp(data_type,'txt') 
        data = readtable(file_path, 'ReadVariableNames',...
                false, 'HeaderLines', 5);
        time = data.Var1;
        ampl = data.Var2;
        data = table(time,ampl);
    end


    if nargin < 3
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