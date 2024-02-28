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


%% Fitting Scenario: PGA 
% Fitting 
Y = log(Info_matrix(:,4));
distance =  sqrt(power(Info_matrix(:,2),2)+ power(Info_matrix(:,3),2));
VarsTable = table(Info_matrix(:,10), Info_matrix(:,1), log(distance), Info_matrix(:,2),Info_matrix(:,3),log(Info_matrix(:,4)),...
            Info_matrix(:,5),log(Info_matrix(:,6)),log(Info_matrix(:,7)),...
    'VariableNames',{'Event','M','Dis','D','R','lnPGA','Wg','DRg','Wc'});

lme = fitlme(VarsTable,'lnPGA ~ M + Dis +  (Dis|Event) ');

% Plot in-event and between event residual
residuals = table(lme.Residuals.Raw,Info_matrix(:,1),log(distance), lme.Variables.Event,'VariableNames',{'res','M','Dis','Event'} );
mean_r_event = grpstats(residuals, 'Event', {'mean','std'});
overal_mean_r = mean(residuals.res);

with_in = mean_r_event.mean_res - overal_mean_r;
between = overal_mean_r  - residuals.res;

figure
subplot(211)
scatter(residuals.M, between);
hold on 
scatter(mean_r_event.mean_M, with_in, 'filled')
yline(0,'Color','[0.15,0.15,0.15]','LineStyle','--')
xlabel("M")
ylabel("Residuals")
legend("with-in events","between events")
ylim([-5,5])

subplot(212)
scatter(exp(lme.Variables.Dis),between, 'filled')
yline(0,'Color','[0.15,0.15,0.15]','LineStyle','--')
ylim([-5,5])
xlabel("distance")
ylabel("Residuals")
figure
histogram(lme.residuals)






%lme = fitlme(VarsTable, 'Wc ~  Wg ')
%lme = fitlme(VarsTable, 'DRg ~  Wg + Dis     ')

%VarsTable = [Info_matrix(:,1), log(distance),log(Info_matrix(:,5))];






%figure
%scatter(Info_matrix(:,5), Info_matrix(:,6))
%len = length(Info_matrix(:,6));
%TestTable_1 = table(10*ones(len,1),1.0*ones(len,1), 1.5*ones(len,1), 5*ones(len,1),5*ones(len,1),5*ones(len,1),...
%            30*ones(len,1),5*ones(len,1),5*ones(len,1),...
%             'VariableNames',{'Event','M','Dis','D','R','lnPGA','Wg','DRg','Wc'});
%
%TestTable_2 = table(10*ones(len,1),0.8*ones(len,1), 2.5*ones(len,1), 5*ones(len,1),5*ones(len,1),5*ones(len,1),...
%            5*ones(len,1),5*ones(len,1),5*ones(len,1),...
%             'VariableNames',{'Event','M','Dis','D','R','lnPGA','Wg','DRg','Wc'});
%
%TestTable_3 = table(10*ones(len,1),1.7*ones(len,1), 2.0*ones(len,1), 5*ones(len,1),5*ones(len,1),5*ones(len,1),...
%            5*ones(len,1),5*ones(len,1),5*ones(len,1),...
%             'VariableNames',{'Event','M','Dis','D','R','lnPGA','Wg','DRg','Wc'});
%
%
%figure
%plotResiduals(lme)
%%p = predict(lme,TestTable);
%error_1 = random(lme,TestTable_1);
%error_2 = random(lme,TestTable_2);
%error_3 = random(lme,TestTable_3);
%figure
%scatter3(log(distance), Info_matrix(:,1), log(Info_matrix(:,4)),'blue')
%hold on 
%scatter3(1.5*ones(len,1)      , 1.0*ones(len,1), error_1,'red')
%scatter3(2.5*ones(len,1)      , 0.8*ones(len,1), error_2,'cyan')
%scatter3(2.0*ones(len,1)      , 1.7*ones(len,1), error_3,'magenta')


%exp(random(lme,TestTable_1))








