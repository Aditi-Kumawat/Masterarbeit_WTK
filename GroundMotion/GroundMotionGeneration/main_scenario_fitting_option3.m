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


% Case 145 has too extreme value for Beta
% Case 36 has too extreme value for W_c

%% Fitting Scenario: Wc
% Fitting 
distance =  sqrt(power(Info_matrix(:,2),2)+ power(Info_matrix(:,3),2));

%VarsTable= table(Info_matrix(:,10), Info_matrix(:,1), log(distance), distance, (Info_matrix(:,2)),(Info_matrix(:,3)),log(Info_matrix(:,4)),...
%            Info_matrix(:,5),Info_matrix(:,6),Info_matrix(:,7),...
%    'VariableNames',{'Event','M','LnDis','Dis','D','R','lnPGA','Wg','DRg','Wc'});
VarsTable= table(Info_matrix(:,10), Info_matrix(:,1),...
                 log(distance), distance, ...
                 log(Info_matrix(:,2)),Info_matrix(:,2),...
                 log(Info_matrix(:,3)),Info_matrix(:,3),...
                 log(Info_matrix(:,4)),...
                 (Info_matrix(:,5)),...
                 (Info_matrix(:,6)),...
                 (Info_matrix(:,7)),...
    'VariableNames',{'Event','M','LnDis','Dis','LnD','D','LnR','R','lnPGA','Wg','DRg','Wc'});

VarsTable(200,:) = [];
VarsTable(153,:) = [];
VarsTable(107,:) = [];
VarsTable(105,:) = [];

num = length(VarsTable.Event);

%LME_lnPGA = fitlme(VarsTable,'lnPGA ~  M + LnDis + (LnDis|Event)');
%LME_lnPGA = fitlme(VarsTable,'lnPGA ~  M + R + D + (R|Event)')
%LME_lnPGA = fitlme(VarsTable,'lnPGA ~  M + R + D + (R|Event)');
LME_lnWc = fitlme(VarsTable, 'Wc     ~  R +  Wg + ( R|Event)+  ( Wg|Event)')
% Beta should be fixed 
LME_lnBeta = fitlme(VarsTable,'DRg  ~  D');

%% Residuals and correaltion 
% Concatenate the residuals into a single matrix
all_residuals = [LME_lnPGA.residuals, LME_lnWc.residuals];
%all_residuals = [VarsTable.lnPGA, VarsTable.Wc,  VarsTable.DRg];
% Compute the correlation matrix
correlation_matrix = corrcoef(all_residuals);
% Compute the covariance matrix
covariance_matrix = cov(all_residuals);
% Compute the Cholesky decomposition of the covariance matrix
chol_matrix = chol(covariance_matrix, 'lower');
% correalated error
mu = [0,0];
%R = mvnrnd(mu,covariance_matrix ,num);

%% Plot and displace
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


%fns_plotResidual_events(LME_lnBeta)

fns_visualized_fitting(LME_lnPGA, VarsTable, VarsTable.PGA, [0.5, 1, 1.5, 1.2, 0.5], [7.0, 9.5, 12.0, 8.0, 10])

%% X dir:
% Wg ~normal
% M ~ lognoraml
% Dis ~normal (not good)

%% Y dir:
% Wg ~normal
% M ~ lognoraml
% Dis ~normal (not good)

%% Z dir:
% Wg ~lognoraml (outlier)
% M ~ lognoraml
% Dis ~normal (not good)

%figure
%probplot('weibull',VarsTable.Wg)
%figure
%probplot('lognormal',VarsTable.Wg)
%figure
%probplot('normal',VarsTable.Wg)


%a = exp(predict(LME_lnWc)+R(:,2));
%histogram((Info_matrix(:,7)),50)
%hold on 
%histogram(a, 50)

Testing= table(Info_matrix(:,10), Info_matrix(:,1), log(distance), distance, Info_matrix(:,2),Info_matrix(:,3),log(Info_matrix(:,4)),...
            Info_matrix(:,5),log(Info_matrix(:,6)),log(Info_matrix(:,7)),...
    'VariableNames',{'Event','M','LnDis','Dis','D','R','lnPGA','Wg','DRg','Wc'});

%num_1 = length(Info_matrix(:,10));
%Testing.M = 0.3 * ones(num_1,1);
%Testing.Dis = 8 * ones(num_1,1);
%Testing.Wg = 30 * ones(num_1,1);
%Testing.LnDis = log(4) * ones(num_1,1);
%R = mvnrnd(mu,covariance_matrix ,num_1);
%
%a = exp(predict(LME_lnWc,Testing)+R(:,2));


%Realization= table(1, 1, 1, 1, 1,1,1,...
%            1,1,1,...
%    'VariableNames',{'Event','M','LnDis','Dis','D','R','lnPGA','Wg','DRg','Wc'});
%
%Realization.Wg = 40;
%Wc_list = zeros(500,1);
%i = 1;
%len_real = 0;
%total = 0;
%while len_real ~= 500
%    total = total + 1;
%    R = mvnrnd(mu,covariance_matrix ,1);
%    Wc = predict(LME_lnWc,Realization)+R(2);
%    if Wc > 0.01 && Wc <= 1
%        Wc_list(i) = Wc;
%        len_real = i;
%        i = i+1;
%    end
%
%end

%len_real = 0;
%Error_list = zeros(500,1);
%mean_value = exp(predict(LME_lnWc,Realization));
%while len_real ~= 500
%    total = total + 1;
%    R = mvnrnd(mu,covariance_matrix ,1);
%    upper = 1- mean_value;
%    lower = mean_value - 0;
%    range = (min(upper,lower));
%    if abs(exp(R(2))) <= range
%        Error_list(i) = exp(R(2));
%        len_real = i;
%        i = i+1;
%    end
%
%end
%Wc_list = mean_value *ones(500,1) + Error_list;


%mean_v = predict(LME_lnWc,Realization);

%R = mvnrnd(mu,covariance_matrix ,1000);
%Wc = exp(mean_v*ones(1000,1)+R(:,2));
%Wc = Wc(Wc<=1);



%fns_generateGM_Scenario(LME_lnPGA,LME_lnWc,VarsTable.M,VarsTable.Dis,VarsTable.Wg,0.3)



