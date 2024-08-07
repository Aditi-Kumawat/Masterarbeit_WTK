clear;
clc;
close all;
addpath(".\REALIZATION\")

Y     = load("TEST3_Y_SBAGM_V6_VAL_RAND_DOE_1000_DIR_Z.mat");
Val_1 = load("TEST3_Y_SBAGM_V6_VAL_FIX550_DOE_1000_DIR_Z.mat");
Val_2 = load("TEST3_Y_SBAGM_V6_VAL_FIX650_DOE_1000_DIR_Z.mat");
Val_3 = load("TEST3_Y_SBAGM_V6_VAL_FIX750_DOE_1000_DIR_Z.mat");
Val_4 = load("TEST3_Y_SBAGM_V6_VAL_FIX850_DOE_1000_DIR_Z.mat");
histogram((Y.Y(:,1)),'Normalization','pdf')
hold on 
histogram((Val_1.Y(:,1)),'Normalization','pdf')
histogram((Val_2.Y(:,1)),'Normalization','pdf')
histogram((Val_3.Y(:,1)),'Normalization','pdf')
histogram((Val_4.Y(:,1)),'Normalization','pdf')

legend('log(Y)', 'log(Y)|X = x1','log(Y)|X = x2','log(Y)|X = x3','log(Y)|X = x4');


clear;
clc;
close all;
addpath("C:\Users\v196m\Desktop\master_project\Masterarbeit\GroundMotion\GroundMotionGeneration")
addpath("C:\Users\v196m\Desktop\master_project\Masterarbeit\GroundMotion\GroundMotionGeneration\BaselineCorrection")


TEST_CASE = 1;
NumDOE = 200;
NumVar = 6;
dir = 'Z';
valid_point = 150;

%Learning_type = 'TR';
Learning_type = 'VAL';

% Only use for output, irrelvent to input
%output_type = "rand";
output_type = "fix";

DOE_file_name = sprintf('TEST%d_X_SBAGM_V%d_%s_DOE_%d_DIR_%s.mat',TEST_CASE,NumVar,Learning_type,NumDOE,dir);
DOE_path =['C:\Users\v196m\Desktop\master_project\Masterarbeit\StochasticPCE\InputData\',DOE_file_name];
X = load(DOE_path);

if strcmp(output_type, 'rand')
    output_file_name = sprintf('TEST%d_Y_SBAGM_V%d_%s_RAND_DOE_%d_DIR_%s.mat',TEST_CASE,NumVar,Learning_type,NumDOE,dir);
elseif strcmp(output_type, 'fix')
    output_file_name = sprintf('TEST%d_Y_SBAGM_V%d_%s_FIX%d_DOE_%d_DIR_%s.mat',TEST_CASE,NumVar,Learning_type,valid_point,NumDOE,dir);
end

output_path = ['.\REALIZATION\',output_file_name];

%% Import the data
file_name = ['Statistic_info_type1_',dir,'.mat'];
path = load(file_name);
data = path.statictic_info_type1;
search_dir = 'C:\Users\v196m\Desktop\master_project\Ground Motion Model\All_signals\';

% Import the data of depth
Depth_path = 'C:\Users\v196m\Desktop\master_project\Ground Motion Model\All_signals\Event_list_Insheim2.txt';
% Read the text file into a table
T = readtable(Depth_path);

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

    Info_matrix(i,1) = M;
    Info_matrix(i,2) = D;
    Info_matrix(i,3) = R;
    Info_matrix(i,4) = PGA;
    Info_matrix(i,5) = W_g;
    Info_matrix(i,6) = Damp_g;
    Info_matrix(i,7) = W_f_ratio;
    Info_matrix(i,8) = Duration;
    Info_matrix(i,9) = T_mid_ratio;
    Info_matrix(i,10) = index;

    Info_array = [Info_array; {parent_dir},{current_dir}, ...
         M,D,R,PGA,W_g,Damp_g,W_f_ratio,Duration,T_mid_ratio];
end


% Case 145 has too extreme value for Beta
% Case 36 has too extreme value for W_c

%% Fitting Scenario: Wc
% Fitting 
distance =  sqrt(power(Info_matrix(:,2),2)+ power(Info_matrix(:,3),2));
VarsTable= table(Info_matrix(:,10), Info_matrix(:,1),...
                 log(distance), distance, ...
                 log(Info_matrix(:,2)),Info_matrix(:,2),...
                 log(Info_matrix(:,3)),Info_matrix(:,3),...
                 log(Info_matrix(:,4)),...
                 (Info_matrix(:,5)),...
                 (Info_matrix(:,6)),...
                 (Info_matrix(:,7)),...
    'VariableNames',{'Event','M','LnDis','Dis','LnD','D','LnR','R','lnPGA','Wg','DRg','Wc'});

% Remove Outlier
VarsTable(200,:) = [];
VarsTable(153,:) = [];
VarsTable(107,:) = [];
VarsTable(105,:) = [];













