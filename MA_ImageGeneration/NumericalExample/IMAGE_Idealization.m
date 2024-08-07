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
x_values_1 = linspace(min(Info_matrix_X(:,1)), max(Info_matrix_X(:,1)), 100);
y_values_1 = pdf(pd, x_values_1);


pd = fitdist((Info_matrix_X(:,5)), 'Lognormal')
[h_chi2, p_chi2] = chi2gof((Info_matrix_X(:,5)), 'CDF', pd)
x_values_2 = linspace(min(Info_matrix_X(:,5)), max(Info_matrix_X(:,5)), 100);
y_values_2 = pdf(pd, x_values_2);


pd = fitdist((Info_matrix_Y(:,5)), 'Lognormal')
[h_chi2, p_chi2] = chi2gof((Info_matrix_Y(:,5)), 'CDF', pd)
x_values_3 = linspace(min(Info_matrix_Y(:,5)), max(Info_matrix_Y(:,5)), 100);
y_values_3 = pdf(pd, x_values_3);

pd = fitdist((Info_matrix_Z(:,5)), 'Lognormal')
[h_chi2, p_chi2] = chi2gof((Info_matrix_Z(:,5)), 'CDF', pd)
x_values_4 = linspace(min(Info_matrix_Z(:,5)), max(Info_matrix_Z(:,5)), 100);
y_values_4 = pdf(pd, x_values_4);

figure
h = histogram(Info_matrix_X(:,1),10,'Normalization','probability','FaceColor',[0 0.4470 0.7410]);
hold on
v = sort(h.Values);
nbins = mean(v(end-2:end));
plot(x_values_1, nbins*y_values_1/max(y_values_1), 'r-', 'LineWidth', 1);
xlabel('Moment magnitude, $M_{L}$', 'Interpreter', 'latex')
ylabel('Normalized frequency', 'Interpreter', 'latex')
grid on



figure
subplot(3,1,1)
h = histogram(Info_matrix_X(:,5),10,'Normalization','probability','FaceColor',[0 0.4470 0.7410]);
hold on
v = sort(h.Values);
nbins = mean(v(end-2:end));
plot(x_values_2, nbins*y_values_2/max(y_values_2), 'r-', 'LineWidth', 1);
xlabel('$\omega_{g,x}/2\pi$', 'Interpreter', 'latex')
ylabel('Normalized frequency', 'Interpreter', 'latex')
grid on

subplot(3,1,2)
histogram(Info_matrix_Y(:,5),10,'Normalization','probability','FaceColor',[0 0.4470 0.7410])
hold on
v = sort(h.Values);
nbins = mean(v(end-2:end));
plot(x_values_3, nbins*y_values_3/max(y_values_3), 'r-', 'LineWidth', 1);
xlabel('$\omega_{g,y}/2\pi$', 'Interpreter', 'latex')
ylabel('Normalized frequency', 'Interpreter', 'latex')
grid on

subplot(3,1,3)
histogram(Info_matrix_Z(:,5),10,'Normalization','probability','FaceColor',[0 0.4470 0.7410])
hold on
v = sort(h.Values);
nbins = mean(v(end-2:end));
plot(x_values_4, nbins*y_values_4/max(y_values_4), 'r-', 'LineWidth', 1);
xlabel('$\omega_{g,z}/2\pi$', 'Interpreter', 'latex')
ylabel('Normalized frequency', 'Interpreter', 'latex')
grid on

