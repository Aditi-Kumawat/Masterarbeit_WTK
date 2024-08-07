clear;
clc;
close all;
addpath('C:\Users\v196m\Desktop\master_project\Masterarbeit\GroundMotion\GroundMotionGeneration')

dir = 'X';
file_name = ['Statistic_info_type1_',dir,'.mat'];
path = load(file_name);
data = path.statictic_info_type1;
search_dir = 'C:\Users\v196m\Desktop\master_project\Ground Motion Model\All_signals\';

%% Import the data of depth
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
Info_matrix_X = zeros(length(Record_info),10);
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

    Info_matrix_X(i,1) = M;
    Info_matrix_X(i,2) = D;
    Info_matrix_X(i,3) = R;
    Info_matrix_X(i,4) = PGA;
    Info_matrix_X(i,5) = W_g;
    Info_matrix_X(i,6) = Damp_g;
    Info_matrix_X(i,7) = W_f_ratio;
    Info_matrix_X(i,8) = Duration;
    Info_matrix_X(i,9) = T_mid_ratio;
    Info_matrix_X(i,10) = index;
end


%%

dir = 'Y';
file_name = ['Statistic_info_type1_',dir,'.mat'];
path = load(file_name);
data = path.statictic_info_type1;
search_dir = 'C:\Users\v196m\Desktop\master_project\Ground Motion Model\All_signals\';

%% Import the data of depth
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
Info_matrix_Y = zeros(length(Record_info),10);
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

    Info_matrix_Y(i,1) = M;
    Info_matrix_Y(i,2) = D;
    Info_matrix_Y(i,3) = R;
    Info_matrix_Y(i,4) = PGA;
    Info_matrix_Y(i,5) = W_g;
    Info_matrix_Y(i,6) = Damp_g;
    Info_matrix_Y(i,7) = W_f_ratio;
    Info_matrix_Y(i,8) = Duration;
    Info_matrix_Y(i,9) = T_mid_ratio;
    Info_matrix_Y(i,10) = index;

end

%%

dir = 'Z';
file_name = ['Statistic_info_type1_',dir,'.mat'];
path = load(file_name);
data = path.statictic_info_type1;
search_dir = 'C:\Users\v196m\Desktop\master_project\Ground Motion Model\All_signals\';

%% Import the data of depth
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
Info_matrix_Z = zeros(length(Record_info),10);
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

    Info_matrix_Z(i,1) = M;
    Info_matrix_Z(i,2) = D;
    Info_matrix_Z(i,3) = R;
    Info_matrix_Z(i,4) = PGA;
    Info_matrix_Z(i,5) = W_g;
    Info_matrix_Z(i,6) = Damp_g;
    Info_matrix_Z(i,7) = W_f_ratio;
    Info_matrix_Z(i,8) = Duration;
    Info_matrix_Z(i,9) = T_mid_ratio;
    Info_matrix_Z(i,10) = index;
end

pd = fitdist((Info_matrix_X(:,1)), 'Lognormal')
[h_chi2, p_chi2] = chi2gof((Info_matrix_X(:,1)), 'CDF', pd)
x_values = linspace(min(Info_matrix_X(:,1)), max(Info_matrix_X(:,1)), 100);
y_values = pdf(pd, x_values);

figure
subplot(3,1,1)
h = histogram(Info_matrix_X(:,1),10,'Normalization','probability','FaceColor',[0 0.4470 0.7410]);
hold on
nbins = max(h.Values);
%plot(x_values, 0.2*y_values/max(y_values), 'r-', 'LineWidth', 1);
xlabel('$M_{L}$', 'Interpreter', 'latex')
ylabel('Normalized frequency', 'Interpreter', 'latex')
subplot(3,1,2)
histogram(Info_matrix_X(:,2),10,'Normalization','probability','FaceColor',[0.8500 0.3250 0.0980])
xlabel('$D$', 'Interpreter', 'latex')
ylabel('Normalized frequency', 'Interpreter', 'latex')
subplot(3,1,3)
histogram(Info_matrix_X(:,3),10,'Normalization','probability','FaceColor',[0.9290 0.6940 0.1250])
xlabel('$R$', 'Interpreter', 'latex')
ylabel('Normalized frequency', 'Interpreter', 'latex')


figure 
h = histogram(Info_matrix_X(:,1),10,'Normalization','probability','FaceColor',[0 0.4470 0.7410]);
hold on
nbins = max(h.Values);
plot(x_values, 0.2*y_values/max(y_values), 'r-', 'LineWidth', 1);
xlabel('$M_{L}$', 'Interpreter', 'latex')
ylabel('Normalized frequency', 'Interpreter', 'latex')
grid on

%figure
%subplot(1,3,1)
%histogram(Info_matrix_X(:,3),8,'Normalization','probability','FaceColor',[0.9290 0.6940 0.1250])
%xlabel('Rupture distance, R_{x}')
%ylabel('Normalized frequency')
%subplot(1,3,2)
%histogram(Info_matrix_Y(:,3),8,'Normalization','probability','FaceColor',[0.9290 0.6940 0.1250])
%xlabel('Rupture distance, R_{y}')
%ylabel('Normalized frequency')
%subplot(1,3,3)
%histogram(Info_matrix_Z(:,3),8,'Normalization','probability','FaceColor',[0.9290 0.6940 0.1250])
%xlabel('Rupture distance, R_{z}')
%ylabel('Normalized frequency')