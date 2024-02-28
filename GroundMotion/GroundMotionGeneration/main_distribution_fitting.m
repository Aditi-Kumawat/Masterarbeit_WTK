clear;
clc;
close all;

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

%% Fitting distribution

data = (Info_matrix(:,4));
%data = log(distance)/max(log(distance);

% Fit the data with gamma, beta, and lognormal distributions
gamma_dist = fitdist(data, 'gamma');
lognormal_dist = fitdist(data, 'lognormal');
gaussian_dist = fitdist(data, 'normal');
%beta_dist = fitdist(data, 'beta');

% Perform Kolmogorov-Smirnov goodness-of-fit tests
%[~, p_beta] = kstest(data, 'CDF', beta_dist);
[~, p_gamma] = kstest(data, 'CDF', gamma_dist);
[~, p_lognormal] = kstest(data, 'CDF', lognormal_dist);% Fit the data with uniform and Gaussian distributions

% Perform Kolmogorov-Smirnov goodness-of-fit tests
%[~, p_uniform] = kstest(data, 'CDF', uniform_dist);
[~, p_gaussian] = kstest(data, 'CDF', gaussian_dist);


% Display p-values
disp(['Gamma p-value: ', num2str(p_gamma)]);
disp(['Lognormal p-value: ', num2str(p_lognormal)]);
%disp(['Beta p-value: ', num2str(p_beta)]);
disp(['Gaussian p-value: ', num2str(p_gaussian)]);

%% Plot the histogram
h = histogram(data, 'Normalization', 'probability', 'DisplayName', 'Data');


hold on;
%gaussian_dist.mu = -8.25;
%gaussian_dist.sigma = 1.4;
% Plot the fitted curves
x_values = linspace(min(data), max(data), 100);
plot(x_values, max(h.Values)*pdf(gamma_dist, x_values)/max(pdf(gamma_dist, x_values)), 'LineWidth', 2, 'DisplayName', 'Gamma Fit');
plot(x_values, max(h.Values)*pdf(lognormal_dist, x_values)/max(pdf(lognormal_dist, x_values)), 'LineWidth', 2, 'DisplayName', 'Lognormal Fit');
plot(x_values, max(h.Values)*pdf(gaussian_dist, x_values)/max(pdf(gaussian_dist, x_values)), 'LineWidth', 2, 'DisplayName', 'Gaussian Fit');

% Add labels and legend
xlabel('Value');
ylabel('Probability Density');
title('Histogram and Fitted Curves');
legend('show');

% Hold off to end the plot
hold off;





%% Perform optimization using fminunc (BFGS)
%[x, fval, exitflag, output] = fminunc(@(x)  MLE_func(x, Y, Vars), x0, options);
%%
%%% Display the result
%disp('Optimal solution:');
%disp(x);
%disp('Optimal function value:');
%disp(fval);
%disp('Exit flag:');
%disp(exitflag);
%disp('Output information:');
%disp(output);
%%figure
%%scatter(Info_matrix(:,1),Y)
%%figure
%%scatter(Info_matrix(:,2),Y)
%
%qqplot(Info_matrix(:,3));
%
%% Plot the 3D scatter plot
%%scatter3(Info_matrix(:,1), sqrt(power(Info_matrix(:,2),2)+ power(Info_matrix(:,3),2)), Y, 'filled');
%%xlabel('X');
%%ylabel('Y');
%%zlabel('Z');
%%title('3D Scatter Plot');
%
%function f = MLE_func(Coeffs, Y, X)
%    in_event_stddev = Coeffs(1);
%    betw_event_stddev = Coeffs(2);
%   
%    % Get the unique elements in the list
%    unique_elements = unique(X(:,end));
%    % Initialize the matrix
%    Sigma_mat = zeros(length(X(:,end)));
%    % Loop through the unique elements
%    for k = 1:length(unique_elements)
%        % Find the indices where the current element occurs in the list
%        indices = find(X(:,end) == unique_elements(k));
%        % Fill the diagonal block with ones for the current element
%        Sigma_mat(indices, indices) = power(in_event_stddev,2);
%    end
%    
%    Distance = sqrt(power(X(:,3),2)+ power(X(:,4),2));
%    Sigma_mat = Sigma_mat + power(betw_event_stddev,2)*eye(length(Sigma_mat));
%
%    %X_mat  = [X(:,1), X(:,2), power(X(:,2),2),  Distance , power( Distance,2)  ];
%    X_mat  = [X(:,1), X(:,2) ,X(:,3), X(:,4) ];
%    
%    mu = (Y - X_mat*transpose(Coeffs(3:end)));
% 
%    % log MLE
%    f = -(-0.5*log(det(Sigma_mat)) - 0.5*((transpose(mu)/Sigma_mat)*mu));
%    %f = -(exp(-0.5*((transpose(mu)/Sigma_mat)*mu)))/sqrt(det(Sigma_mat));
%
%end











