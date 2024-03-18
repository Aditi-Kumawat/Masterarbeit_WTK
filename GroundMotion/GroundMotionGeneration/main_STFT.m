clear;
clc;
close all;

addpath("BaselineCorrection\")
path = load('Statistic_info_type1_Z.mat');
all_path = path.statictic_info_type1;
data_t = fns_import_time_data(all_path{2}{1},'txt',[]);
data_f = fns_fft_data(data_t,100,false);

[s,f,t]  = stft(data_t.ampl,200,Window=kaiser(100,5),OverlapLength=60,FFTLength=length(data_t.ampl));

sdb = mag2db(abs(s));
mesh(t,f,sdb);

cc = max(sdb(:))+[-60 0];
ax = gca;
ax.CLim = cc;
xlim([0,t(end)])
view(2)
colorbar