
clear;
clc;
close all;

addpath("C:\Users\v196m\Desktop\master_project\Masterarbeit\GroundMotion\GroundMotionGeneration")



%%

dir = 'X';
file_name = ['Statistic_info_type1_',dir,'.mat'];
path = load(file_name);
data = path.statictic_info_type1;
search_dir = 'C:\Users\v196m\Desktop\master_project\Ground Motion Model\All_signals\';

a = fns_import_time_data(data{1}{1},'txt',[]);

k = fns_import_time_data(data{2}{1},'txt',[]);

figure 
plot(k.time, k.ampl,'Color',[0 0.4470 0.7410],'LineWidth',1)
legend("Recorded GM from INS3, 21:47:07, 9$^{th}$ Nov.,2020", 'Interpreter', 'latex')
xlabel("Time (sec)", 'Interpreter', 'latex')
ylabel("$Acc_{x}$ $(m/s^{2})$", 'Interpreter', 'latex')
grid on


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


for i = 1:length(data)
    a = fns_import_time_data(data{i}{1},'txt',[]);
    [b,Fs] = fns_fft_data(a,100,false,false);
    vq1 = interp1(b.Freq,b.Ampl, fix_freq_list );
    all_recored(:,i) = vq1;
    %
    %hold on 
end



figure 

event = 2;
plot(fix_freq_list,all_recored(:,event),'Color',[0 0.4470 0.7410],'LineWidth',1)
hold on

plot(fix_freq_list,all_recored,'Color',[0.9290 0.6940 0.1250])
plot(fix_freq_list,all_recored(:,event),'Color',[0 0.4470 0.7410],'LineWidth',1)

%plot(fix_freq_list,mean_1,'Color',[0.8500 0.3250 0.0980],'LineWidth',1)
%plot(fix_freq_list,mean_1,'Color',[0.8500 0.3250 0.0980],'LineWidth',1)
%plot(fix_freq_list,y_smooth,'Color',[0 0.4470 0.7410],'LineWidth',2)

legend("Recorded GM from INS3, 21:47:07, 9$^{th}$ Nov.,2020","All Recorded GMs in x-direction", 'Interpreter', 'latex')
xlabel("Frequency (Hz)", 'Interpreter', 'latex')
ylabel("$Acc_{x}$ $(m/s^{2})$", 'Interpreter', 'latex')