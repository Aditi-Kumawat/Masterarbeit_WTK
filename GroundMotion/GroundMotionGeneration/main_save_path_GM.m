clear;
clc;
close;

%% Detect all the sub folder in the main folder
% Specify directory and file name
main_dir = 'C:\Users\v196m\Desktop\master_project\Masterarbeit\GroundMotion\RecordData\'; 

% Use dir to get a list of all files and folders
contents = dir(main_dir);
% Extract only folders from the contents
folders = contents([contents.isdir]);
% Exclude items that are not folders
folders = folders(~ismember({folders.name}, {'.', '..'}));
% Extract folder names from the folder structure
folder_names = {folders.name};

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
        first_acc_record = txt_file_names(3);
        file_path = fullfile(subsub_folders{1},first_acc_record{1});

        % i index represent the event , j index represent the stations
        file_list{i,j} = file_path ;
    end

    
end
%save('events_Z.mat','file_list','-mat');


