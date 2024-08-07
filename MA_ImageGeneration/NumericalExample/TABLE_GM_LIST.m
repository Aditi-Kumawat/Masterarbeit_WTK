clear;
clc;
close all;

dir = 'Z';
file_name = ['Statistic_info_type1_',dir,'.mat'];
path = load(file_name);
data = path.statictic_info_type1;
search_dir = 'C:\Users\v196m\Desktop\master_project\Ground Motion Model\All_signals\';

%% Import the data of depth
Depth_path = 'C:\Users\v196m\Desktop\master_project\Ground Motion Model\All_signals\Event_list_Insheim2.txt';
% Read the text file into a table
T = readtable(Depth_path);
Info_array = [];
for j = 1:length(T.Var1)
    % Convert the string to a datetime object
    dt = datetime(T.Var1(j), 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss');
    % Convert the datetime object to a formatted string
    T.Var7(j) = "GM_"+ sprintf('%d_%02d_%02d_%02d_%02d_%02d', ...
        year(dt), month(dt), day(dt), hour(dt), minute(dt), second(dt));
end

Record_info = data(:,1);
Spectral_info = data(:,2);
Time_info = data(:,3);
GM_info =   data(:,4);

%% Making Info Matrix
% MAT = ( M, D, R, S0, W_g, Damp_g, W_f_ratio, Duration, T_mid_ratio)
Info_matrix = zeros(length(Record_info),10);
Info_array = [];
for i = 1:length(Record_info)
    Record_info = data(:,1);
    Record_info_i = Record_info{i};
    
    % Use fileparts to extract the folder parts
    [path_record, ~, ~] = fileparts(Record_info_i{1});
    % Extract the directory names
    [~, current_dir, ~] = fileparts(path_record);
    [~, parent_dir, ~] = fileparts(fileparts(path_record)); 
    index = find(strcmp(T.Var7, parent_dir));
    [ID, dateTime] = separateString(current_dir);
    if ~isempty(index)
        D = T.Var4(index);
    else
        dips("ERROR! no depth found")
    end

    R = Record_info_i{2};
    M = Record_info_i{3};
    
    PGA         = GM_info{i}(1);
    W_g         = Spectral_info{i}(1)/(2*pi);
    Damp_g      = Spectral_info{i}(2);
    W_f_ratio   = Spectral_info{i}(3);
    Duration    = Time_info{i}(4);
    T_mid_ratio = Time_info{i}(3);


    %Info_matrix(i,1) = index;
    %Info_matrix(i,2) = ID;
    %Info_matrix(i,3) = dateTime;
    %Info_matrix(i,4) = M;
    %Info_matrix(i,5) = D;
    %Info_matrix(i,6) = R;
    %Info_matrix_Z(i,4) = PGA;
    %Info_matrix_Z(i,5) = W_g;
    %Info_matrix_Z(i,6) = Damp_g;
    %Info_matrix_Z(i,7) = W_f_ratio;
    %Info_matrix_Z(i,8) = Duration;
    %Info_matrix_Z(i,9) = T_mid_ratio;
    %Info_matrix_Z(i,10) = index;
    Info_array = [Info_array; length(Record_info)-i+1, index,{ID},{dateTime},M,D,R];
end
Info_array = flipud(Info_array);


VarsTable= table(Info_array(:,1), ...
                 Info_array(:,2),...
                 Info_array(:,3),...
                 Info_array(:,4),...
                 Info_array(:,5),...
                 Info_array(:,6),...
                 Info_array(:,7),...
   'VariableNames',{'Index','Event','Station','Time','M_L','D','R'});

% Specify the filename
filename = 'Table_all_event.csv';

% Write the table to a CSV file
writetable(VarsTable, filename);



function [ID, dateTime] = separateString(inputStr)
    % Split the string based on the underscore delimiter
    parts = strsplit(inputStr, '_');
    
    % Extract the ID and dateTime parts
    ID = parts{2}; % Second part is the ID (TMO55)
    dateTime = strjoin(parts(3:end), '_'); % Join the remaining parts to form the date and time
    dateTime = convertDateTime(dateTime);
    % Display the results
    disp(['ID: ', ID]);
    disp(['Date and Time: ', dateTime]);
end


function formattedDateTime = convertDateTime(inputDateTime)
    % Split the input string into components based on underscores
    parts = strsplit(inputDateTime, '_');
    
    % Extract date and time components
    year = parts{1};
    month = parts{2};
    day = parts{3};
    hour = parts{4};
    minute = parts{5};
    second = parts{6};
    
    % Format the date and time components
    formattedDateTime = sprintf('%s.%s.%s-%s:%s:%s', year, month, day, hour, minute, second);
end
