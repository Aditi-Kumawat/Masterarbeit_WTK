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


% Create a figure with 6 subplots arranged in a 3x2 grid
figure;

% First subplot
pd = fitdist(log(Info_matrix_X(:,4)), 'Normal');
[h_chi2, p_chi2] = chi2gof(log(Info_matrix_X(:,4)), 'CDF', pd)
x_values = linspace(min(log(Info_matrix_X(:,4))), max(log(Info_matrix_X(:,4))), 100);
y_values = pdf(pd, x_values);
subplot(2, 2, 1); % 3 rows, 2 columns, first subplot
h = histogram(log(Info_matrix_X(:,4)),10,'Normalization','probability','FaceColor',[0 0.4470 0.7410]);
nbins = max(h.Values);
hold on
%plot(x_values, nbins*y_values/max(y_values), 'r-', 'LineWidth', 1);
%title('Peak ground acceleration', 'Interpreter', 'latex');
xlabel('log(PGA) ($m/s^{2}$)', 'Interpreter', 'latex');
ylabel('Normalized frequency', 'Interpreter', 'latex');
grid on;


% Second subplot
subplot(2, 2, 2); % 3 rows, 2 columns, second subplot
pd = fitdist((Info_matrix_X(:,5)), 'Normal');
[h_chi2, p_chi2] = chi2gof(Info_matrix_X(:,5), 'CDF', pd)
x_values = linspace(min((Info_matrix_X(:,5))), max((Info_matrix_X(:,5))), 100);
y_values = pdf(pd, x_values);
h = histogram((Info_matrix_X(:,5)),10,'Normalization','probability','FaceColor',[0 0.4470 0.7410]);
nbins = max(h.Values);
hold on
%plot(x_values, nbins*y_values/max(y_values), 'r-', 'LineWidth', 1);
%title('Predominant frequency of ground motion', 'Interpreter', 'latex');
xlabel('$\omega_{g}/2 \pi$ (Hz)', 'Interpreter', 'latex');
ylabel('Normalized frequency', 'Interpreter', 'latex');
grid on;

% Third subplot
subplot(2, 2, 3); % 3 rows, 2 columns, third subplot
pd = fitdist((Info_matrix_X(:,6)), 'Gamma');
[h_chi2, p_chi2] = chi2gof(Info_matrix_X(:,6),'Alpha',0.05 ,'CDF', pd)
x_values = linspace(min((Info_matrix_X(:,6))), max((Info_matrix_X(:,6))), 100);
y_values = pdf(pd, x_values);
h = histogram((Info_matrix_X(:,6)),10,'Normalization','probability','FaceColor',[0 0.4470 0.7410]);
nbins = max(h.Values);
hold on
%plot(x_values, nbins*y_values/max(y_values), 'r-', 'LineWidth', 1);
%title('Damping ratio', 'Interpreter', 'latex');
xlabel('$\zeta_{g}$', 'Interpreter', 'latex');
ylabel('Normalized frequency', 'Interpreter', 'latex');
grid on;

% Fourth subplot
subplot(2, 2, 4); % 3 rows, 2 columns, fourth subplot
pd = fitdist((Info_matrix_X(:,7)), 'Gamma');
[h_chi2, p_chi2] = chi2gof(Info_matrix_X(:,7),'Alpha',0.05, 'CDF', pd)
x_values = linspace(min((Info_matrix_X(:,7))), max((Info_matrix_X(:,7))), 100);
y_values = pdf(pd, x_values);
h = histogram((Info_matrix_X(:,7)),10,'Normalization','probability','FaceColor',[0 0.4470 0.7410]);
nbins = max(h.Values);
hold on
%plot(x_values, nbins*y_values/max(y_values), 'r-', 'LineWidth', 1);
%title('Ratio of high-pass filter', 'Interpreter', 'latex');
xlabel('$r_{f}$', 'Interpreter', 'latex');
ylabel('Normalized frequency', 'Interpreter', 'latex');
grid on;






% Step 2: Create a Q-Q plot for an exponential distribution
figure;
subplot(2, 2, 1)
qqplot(log(Info_matrix_X(:,4)), makedist('Normal'));
q1_data = quantile(log(Info_matrix_X(:,4)), 0.25);
q3_data = quantile(log(Info_matrix_X(:,4)), 0.75);
pd = makedist('Normal');
q1_theoretical = icdf(pd, 0.25);
q3_theoretical = icdf(pd, 0.75);
hold on;
plot(q1_theoretical, q1_data, 'ro', 'MarkerSize', 13, 'DisplayName', 'Q1');
plot(q3_theoretical, q3_data, 'ro', 'MarkerSize', 13, 'DisplayName', 'Q3');
title('log(PGA) ($m/s^{2}$)', 'Interpreter', 'latex');
ylabel('Quantiles of observed data', 'Interpreter', 'latex');
xlabel('Quantiles of normal Distribution', 'Interpreter', 'latex');
grid on;

subplot(2, 2, 2)
qqplot(Info_matrix_X(:,5), makedist('Normal'));
q1_data = quantile((Info_matrix_X(:,5)), 0.25);
q3_data = quantile((Info_matrix_X(:,5)), 0.75);
pd = makedist('Normal');
q1_theoretical = icdf(pd, 0.25);
q3_theoretical = icdf(pd, 0.75);
hold on;
plot(q1_theoretical, q1_data, 'ro', 'MarkerSize', 13, 'DisplayName', 'Q1');
plot(q3_theoretical, q3_data, 'ro', 'MarkerSize', 13, 'DisplayName', 'Q3');
title('$\omega_{g}/2 \pi$ (Hz)', 'Interpreter', 'latex');
ylabel('Quantiles of observed data', 'Interpreter', 'latex');
xlabel('Quantiles of normal Distribution', 'Interpreter', 'latex');
grid on;

subplot(2, 2, 3)
qqplot(Info_matrix_X(:,6), makedist('Normal'));
q1_data = quantile((Info_matrix_X(:,6)), 0.25);
q3_data = quantile((Info_matrix_X(:,6)), 0.75);
pd = makedist('Normal');
q1_theoretical = icdf(pd, 0.25);
q3_theoretical = icdf(pd, 0.75);
hold on;
plot(q1_theoretical, q1_data, 'ro', 'MarkerSize', 13, 'DisplayName', 'Q1');
plot(q3_theoretical, q3_data, 'ro', 'MarkerSize', 13, 'DisplayName', 'Q3');
title('$\zeta_{g}$', 'Interpreter', 'latex');
ylabel('Quantiles of observed data', 'Interpreter', 'latex');
xlabel('Quantiles of normal Distribution', 'Interpreter', 'latex');
grid on;

subplot(2, 2, 4)
qqplot(Info_matrix_X(:,7), makedist('Normal'));
q1_data = quantile((Info_matrix_X(:,7)), 0.25);
q3_data = quantile((Info_matrix_X(:,7)), 0.75);
pd = makedist('Normal');
q1_theoretical = icdf(pd, 0.25);
q3_theoretical = icdf(pd, 0.75);
hold on;
plot(q1_theoretical, q1_data, 'ro', 'MarkerSize', 13, 'DisplayName', 'Q1');
plot(q3_theoretical, q3_data, 'ro', 'MarkerSize', 13, 'DisplayName', 'Q3');
title('$r_{f}$', 'Interpreter', 'latex');
ylabel('Quantiles of observed data', 'Interpreter', 'latex');
xlabel('Quantiles of normal Distribution', 'Interpreter', 'latex');
grid on;

% Add labels and title
%xlabel('Theoretical Quantiles');

%title('Q-Q Plot: Data vs. Exponential Distribution');

% Create an invisible axis that spans the entire figure
ax = axes('Position', [0 0 1 1], 'Visible', 'off');
% Set the common labels
axes(ax);  % Activate the invisible axis
xlabel('Common X Label');
ylabel('Common Y Label');
% Optional: Customize the plot





%% First subplot
%subplot(3, 2, 1); % 3 rows, 2 columns, first subplot
%histogram(log(Info_matrix_X(:,4)),10,'Normalization','probability','FaceColor',[0 0.4470 0.7410])
%title('Peak ground acceleration', 'Interpreter', 'latex');
%xlabel('log(PGA) ($m/s^{2}$)', 'Interpreter', 'latex');
%ylabel('Normalized frequency');
%grid on;
%% Second subplot
%subplot(3, 2, 2); % 3 rows, 2 columns, second subplot
%histogram((Info_matrix_X(:,5)),10,'Normalization','probability','FaceColor',[0 0.4470 0.7410])
%title('Predominant frequency of ground motion', 'Interpreter', 'latex');
%xlabel('$\omega_{g}/2 \pi$ (Hz)', 'Interpreter', 'latex');
%ylabel('Normalized frequency');
%grid on;
%% Third subplot
%subplot(3, 2, 3); % 3 rows, 2 columns, third subplot
%histogram((Info_matrix_X(:,6)),10,'Normalization','probability','FaceColor',[0 0.4470 0.7410])
%title('Damping ratio', 'Interpreter', 'latex');
%xlabel('$\zeta_{g}$', 'Interpreter', 'latex');
%ylabel('Normalized frequency');
%grid on;
%% Fourth subplot
%subplot(3, 2, 4); % 3 rows, 2 columns, fourth subplot
%histogram((Info_matrix_X(:,7)),10,'Normalization','probability','FaceColor',[0 0.4470 0.7410])
%title('Ratio of high-pass filter', 'Interpreter', 'latex');
%xlabel('$r_{f}$', 'Interpreter', 'latex');
%ylabel('Normalized frequency');
%grid on;
%% Fifth subplot
%subplot(3, 2, 5); % 3 rows, 2 columns, fifth subplot
%histogram((Info_matrix_X(:,8)),10,'Normalization','probability','FaceColor',[0 0.4470 0.7410])
%title('Duration', 'Interpreter', 'latex');
%xlabel('$t_{d}$ (sec)', 'Interpreter', 'latex');
%ylabel('Normalized frequency');
%grid on;
%% Sixth subplot
%subplot(3, 2, 6); % 3 rows, 2 columns, sixth subplot
%histogram((Info_matrix_X(:,9)),10,'Normalization','probability','FaceColor',[0 0.4470 0.7410])
%title('Time ratio of PGA', 'Interpreter', 'latex');
%xlabel('$r_{mid}$', 'Interpreter', 'latex');
%ylabel('Normalized frequency');
%grid on;


%figure
%histogram(log(Info_matrix_X(:,4)),10,'Normalization','probability','FaceColor',[0 0.4470 0.7410])
%xlabel('PGA')
%ylabel('Normalized frequency')
%figure
%histogram(Info_matrix_X(:,5),10,'Normalization','probability','FaceColor',[0 0.4470 0.7410])
%xlabel('WG')
%ylabel('Normalized frequency')
%figure
%histogram(Info_matrix_X(:,6),10,'Normalization','probability','FaceColor',[0 0.4470 0.7410])
%xlabel('Damp')
%ylabel('Normalized frequency')
%figure
%histogram(Info_matrix_X(:,7),10,'Normalization','probability','FaceColor',[0 0.4470 0.7410])
%xlabel('Wf')
%ylabel('Normalized frequency')
%figure
%histogram(Info_matrix_X(:,8),10,'Normalization','probability','FaceColor',[0 0.4470 0.7410])
%xlabel('Duration')
%ylabel('Normalized frequency')
%figure
%histogram(Info_matrix_X(:,9),10,'Normalization','probability','FaceColor',[0 0.4470 0.7410])
%xlabel('Rmid')
%ylabel('Normalized frequency')


