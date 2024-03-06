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

VarsTable= table(Info_matrix(:,10), Info_matrix(:,1), log(distance), distance, Info_matrix(:,2),Info_matrix(:,3),log(Info_matrix(:,4)),...
            log(Info_matrix(:,5)),log(Info_matrix(:,6)),log(Info_matrix(:,7)),...
    'VariableNames',{'Event','M','LnDis','Dis','D','R','lnPGA','Wg','DRg','Wc'});
VarsTable(36,:) = [];
num = length(VarsTable.Event);

LME_lnPGA = fitlme(VarsTable,'lnPGA ~  M + LnDis + (LnDis|Event) ',"StartMethod","random");
LME_lnWg = fitlme(VarsTable,'Wg     ~  M + Dis + ( Dis|Event)',"StartMethod","random");
% Wc should be fixed 
LME_lnWc = fitlme(VarsTable,'Wc     ~  M + Dis + ( Dis|Event)',"StartMethod","random");
% Beta should be fixed 
LME_lnBeta = fitlme(VarsTable,'DRg  ~  M + Dis + ( Dis|Event)',"StartMethod","random");


% Concatenate the residuals into a single matrix
all_residuals = [LME_lnPGA.residuals, LME_lnWg.residuals, LME_lnWc.residuals, LME_lnBeta.residuals];

% Compute the correlation matrix
correlation_matrix = corrcoef(all_residuals);
% Compute the covariance matrix
covariance_matrix = cov(all_residuals);
% Compute the Cholesky decomposition of the covariance matrix
chol_matrix = chol(covariance_matrix, 'lower');
mu = [0,0,0,0];
R = mvnrnd(mu,covariance_matrix ,num);

%Plot the correlation matrix
figure;
corrplot(all_residuals)

% Display the correlation matrix
disp('Correlation Matrix of Residuals:');
disp(correlation_matrix);

% Display the covariance matrix
disp('Covariance Matrix of Residuals:');
disp(covariance_matrix);

% Display the Cholesky matrix
disp('Cholesky Decomposition of the Covariance Matrix:');
disp(chol_matrix);


%fns_plotResidual_events(LME_lnPGA)

%fns_visualized_fitting(LME_lnWg, VarsTable, VarsTable.Wg, [0.5, 1, 1.5, 1.2, 0.5], [7.0, 9.5, 12.0, 8.0, 10])

a = exp(predict(LME_lnWc)+R(:,2));
histogram((Info_matrix(:,7)),100)
hold on 
histogram(a, 100)



















