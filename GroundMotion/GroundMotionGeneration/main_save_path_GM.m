clear;
clc;
close;

%% Detect all the sub folder in the main folder
% Specify directory and file name
%main_dir = 'C:\Users\v196m\Desktop\master_project\Masterarbeit\GroundMotion\RecordData\'; 
main_dir = 'C:\Users\v196m\Desktop\master_project\Ground Motion Model\All_signals\';
dir_ = 'X';


%% Start the algorithm
% Use dir to get a list of all files and folders
contents = dir(main_dir);
% Extract only folders from the contents
folders = contents([contents.isdir]);
% Exclude items that are not folders
folders = folders(~ismember({folders.name}, {'.', '..'}));
% Extract folder names from the folder structure
folder_names = {folders.name};

if strcmp(dir_,'X')
    dir_num = 1;
elseif strcmp(dir_,'Y')
    dir_num = 2;
elseif strcmp(dir_,'Z')
    dir_num = 3;
else
    dir_num = 0;
end

file_list = {};
%Only 12 events
for i = 1:length(folder_names)
    %% Detect the second sub folder in the first sub folder
    sub_path = fullfile(main_dir,folder_names(i));

    % Use dir to get a list of all files and folders
    sub_contents = dir(sub_path{1});
    % Extract only folders from the contents
    sub_folders = sub_contents([sub_contents.isdir]);
    % Exclude items that are not folders
    sub_folders = sub_folders(~ismember({sub_folders.name}, {'.', '..'}));
    % Extract folder names from the folder structure
    sub_folders_names = {sub_folders.name};
    
    for j = 1: length(sub_folders_names)
        subsub_folders = fullfile(sub_path,sub_folders_names(j));
    
        
        %% Detect the first acceleration record
        subsub_contents = dir(subsub_folders{1});
        txt_files = ...
            subsub_contents(~[subsub_contents.isdir] & endsWith({subsub_contents.name}, '.txt'));
        txt_file_names = {txt_files.name};

        %index here 1-> X_dir acc, 2->Y_dir acc, 3->Z_dir acc
        first_acc_record = txt_file_names(dir_num);
        file_path = fullfile(subsub_folders{1},first_acc_record{1});
        
        %Get the info of R and M
        data = readtable(file_path, 'ReadVariableNames',...
                false, 'HeaderLines', 2);
        if ~isnan(data.Var6(1))
            if isa(data.Var6(1), 'numeric')
                R = data.Var6(1);
            else
                disp(['ERROR, R cannot found, case: ',file_path])
            end
        else
            disp(['ERROR, R cannot found, case: ',file_path])
        end

        if ~isnan(data.Var9(1))
            if isa(data.Var9(1), 'numeric')
                M = data.Var9(1);
            else
                disp(['ERROR, M cannot found, case: ',file_path])
            end
        else
            disp(['ERROR, M cannot found, case: ',file_path])
        end

        % i index represent the event , j index represent the stations
        file_list{i,j} = {file_path,R,M} ;
    end

    
end
save(['All_events_record_',dir_,'.mat'],'file_list','-mat');


