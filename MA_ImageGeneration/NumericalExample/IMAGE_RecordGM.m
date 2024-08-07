
clear;
clc;
close all;

addpath("C:\Users\v196m\Desktop\master_project\Masterarbeit\GroundMotion\GroundMotionGeneration")


%path = load('events_Z.mat');
path = load('All_events_record_Z.mat');
%fit_type = 'Hu_S0_NoContraint';
fit_type = 'Hu_S0';
all_path = path.file_list;
path_all_events = [];


for i = 1:size(all_path,1)
    for j = 1:size(all_path,2)
        cell = all_path(i,j);
        if ~isempty(cell{1})
            path_all_events = [path_all_events;all_path(i,j)];
        end
    end
end

%%

dir = 'X';
file_name = ['Statistic_info_type1_',dir,'.mat'];
path = load(file_name);
data = path.statictic_info_type1;
search_dir = 'C:\Users\v196m\Desktop\master_project\Ground Motion Model\All_signals\';

a = fns_import_time_data(data{1}{1},'txt',[]);


[b,Fs] = fns_fft_data(a,100,false,false);
fix_freq_list = transpose(0:(b.Freq(3)-b.Freq(2)):100);

all_recored = zeros(length(fix_freq_list),length(data));

%figure
%for i = 1:length(data)
%    a = fns_import_time_data(data{i}{1},'txt',[]);
%    plot(a.time,a.ampl);
%   hold on
%end
%hold off

figure 
subplot(3, 1, 1); 
for i = 1:length(data)
    a = fns_import_time_data(data{i}{1},'txt',[]);
    [b,Fs] = fns_fft_data(a,100,false,false);
    vq1 = interp1(b.Freq,b.Ampl, fix_freq_list );
    all_recored(:,i) = vq1;
    %
    %hold on 
end
mean_1 = mean(all_recored,2);
plot(fix_freq_list,mean_1,'Color',[0.8500 0.3250 0.0980],'LineWidth',1)
hold on
windowSize = 250; % Window size for the moving average
y_smooth = movmean(mean_1, windowSize);
plot(fix_freq_list,y_smooth,'Color',[0 0.4470 0.7410],'LineWidth',2)

plot(fix_freq_list,all_recored,'Color',[0.9290 0.6940 0.1250])
plot(fix_freq_list,mean_1,'Color',[0.8500 0.3250 0.0980],'LineWidth',1)
plot(fix_freq_list,y_smooth,'Color',[0 0.4470 0.7410],'LineWidth',2)

ylim([0,0.02])
legend("Average","Average smoothing","Recorded GMs in x-direction")
xlabel("Frequency (Hz)", 'Interpreter', 'latex')
ylabel("$Acc_{x}$ $(m/s^{2})$", 'Interpreter', 'latex')

%%

dir = 'Y';
file_name = ['Statistic_info_type1_',dir,'.mat'];
path = load(file_name);
data = path.statictic_info_type1;
search_dir = 'C:\Users\v196m\Desktop\master_project\Ground Motion Model\All_signals\';

a = fns_import_time_data(data{1}{1},'txt',[]);
[b,Fs] = fns_fft_data(a,100,false,false);
fix_freq_list = transpose(0:(b.Freq(3)-b.Freq(2)):100);

all_recored = zeros(length(fix_freq_list),length(data));

%figure
%for i = 1:length(data)
%    a = fns_import_time_data(data{i}{1},'txt',[]);
%    plot(a.time,a.ampl);
%   hold on
%end
%hold off


subplot(3, 1, 2); 
for i = 1:length(data)
    a = fns_import_time_data(data{i}{1},'txt',[]);
    [b,Fs] = fns_fft_data(a,100,false,false);
    vq1 = interp1(b.Freq,b.Ampl, fix_freq_list );
    all_recored(:,i) = vq1;
    %
    %hold on 
end
mean_2 = mean(all_recored,2);
plot(fix_freq_list,mean_2,'Color',[0.8500 0.3250 0.0980],'LineWidth',1)
hold on
windowSize = 250; % Window size for the moving average
y_smooth = movmean(mean_2, windowSize);
plot(fix_freq_list,y_smooth,'Color',[0 0.4470 0.7410],'LineWidth',2)

plot(fix_freq_list,all_recored,'Color',[0.9290 0.6940 0.1250])
plot(fix_freq_list,mean_2,'Color',[0.8500 0.3250 0.0980],'LineWidth',1)
plot(fix_freq_list,y_smooth,'Color',[0 0.4470 0.7410],'LineWidth',2)

ylim([0,0.02])
legend("Average","Average smoothing","Recorded GMs in y-direction")
xlabel("Frequency (Hz)", 'Interpreter', 'latex')
ylabel("$Acc_{y}$ $(m/s^{2})$", 'Interpreter', 'latex')

%%

dir = 'Z';
file_name = ['Statistic_info_type1_',dir,'.mat'];
path = load(file_name);
data = path.statictic_info_type1;
search_dir = 'C:\Users\v196m\Desktop\master_project\Ground Motion Model\All_signals\';

a = fns_import_time_data(data{1}{1},'txt',[]);
[b,Fs] = fns_fft_data(a,100,false,false);
fix_freq_list = transpose(0:(b.Freq(3)-b.Freq(2)):100);

all_recored = zeros(length(fix_freq_list),length(data));

%figure
%for i = 1:length(data)
%    a = fns_import_time_data(data{i}{1},'txt',[]);
%    plot(a.time,a.ampl);
%   hold on
%end
%hold off

 
subplot(3, 1, 3); 
for i = 1:length(data)
    a = fns_import_time_data(data{i}{1},'txt',[]);
    [b,Fs] = fns_fft_data(a,100,false,false);
    vq1 = interp1(b.Freq,b.Ampl, fix_freq_list );
    all_recored(:,i) = vq1;
    %
    %hold on 
end
mean_3 = mean(all_recored,2);
plot(fix_freq_list,mean_3,'Color',[0.8500 0.3250 0.0980],'LineWidth',1)
hold on
windowSize = 250; % Window size for the moving average
y_smooth = movmean(mean_3, windowSize);
plot(fix_freq_list,y_smooth,'Color',[0 0.4470 0.7410],'LineWidth',2)

plot(fix_freq_list,all_recored,'Color',[0.9290 0.6940 0.1250])
plot(fix_freq_list,mean_3,'Color',[0.8500 0.3250 0.0980],'LineWidth',1)
plot(fix_freq_list,y_smooth,'Color',[0 0.4470 0.7410],'LineWidth',2)

ylim([0,0.02])
legend("Average","Average smoothing","Recorded GMs in z-direction")
xlabel("Frequency (Hz)", 'Interpreter', 'latex')
ylabel("$Acc_{z}$ $(m/s^{2})$", 'Interpreter', 'latex')





