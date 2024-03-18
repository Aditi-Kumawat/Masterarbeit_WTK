clear;
clc;
close all;

addpath("BaselineCorrection\")
%path = 'C:\Users\v196m\Desktop\master_project\Masterarbeit\GroundMotion\RecordData\GM_2016_02_12_06_26_04\GM_TMO57_2016_02_12_06_26_04\a_2_TMO57_2016_02_12_06_26_04.txt';
%path = load('events.mat');
%all_path = path.file_list;
%data_t = fns_import_time_data(path,'txt',[1,25]);



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

%c = cls_GM_generator(1,data_t,100);
%
%
%[GM,FRF_info,Time_info,GM_info] = c.generateTimeNonStaPesudoGMbyFit("Hu_S0",[200,0.5,0.5,c.PGA],[],false);
%
%
%[Time, Acc, ~, Disp] = fns_generateGM_Params(1, [], 100 ,"Hu_S0", FRF_info(1:3), [1, 1.5 ,5, 10], GM_info(1));
%
%
%
%
%%b = fns_fft_data(data_t,100,false,true);
%%[c,f] = periodogram(data_t.ampl,[],b.Freq,200);
%%
%%psd = (1/(200*length(data_t.ampl))) *power(b.Ampl,2);
%%figure
%%plot(f,c)
%%hold on 
%%plot(b.Freq, psd,'--')
%%rng(2);
%
%
%%GMG = cls_GM_generator(a, 100);
%
%
%%GMG.HuKTmodel(50,0.5,0.211,2);
%%c = GMG.fit_HuKTmodel_S0([100,0.1,1,GMG.PGA],true);
%%GMG.fit_GM_model("Hu_S0",[100,0.01,0.5,GMG.PGA],true);
%%GMG.fit_HuKTmodel_S0([200,0.1,1],true)
%%GMG.GMmodel("KT_S0",filter_params,[],true);
%%GMG.generateWhiteNoise(true);
%%g = GMG.generateNormStaPesudoGM("CP_S0",[200,0.01,50,0.01,GMG.PGA],[],'time',true,true);
%%time = GMG.Time;
%%ampl = GMG.PGA*g.ampl;
%%ampl(1) = 0 ;
%%PesuedoGM = table(time,ampl);
%
%%time_percentiles = GMG.getPercentileInfo(true);
%%q=GMG.generateTimeModFunc(time_percentiles,[],[],1,true);
%
%
%
%
%%%fns_fft_data(PesuedoGM,100,false,true);
%%%fns_fft_data(a,100,false,true);
%%%
%%
%%[time_,acc,vel,disp_] = fns_generateGM_Params(2,GM.time(GM.time<=30),100,"Hu_S0",...
%%                          FRF_info,...
%%                          Time_info,...
%%                          GMG.AriasIntensity);
%%
%ampl = GM.ampl;
%time = GM.time;
%
%%%
%%%
%[vel, disp] = newmarkIntegrate(time, ampl, 0.5, 0.25);
%[nomDR, nomAR] = computeDriftRatio(time, disp, 'ReferenceAccel', ampl);
%[accel_A3, vel_A3, disp_A3] = baselineCorrection(time, ampl, 'AccelFitOrder', 3);
%[A3DR, A3AR] = computeDriftRatio(time, disp_A3, 'ReferenceAccel', ampl);
%vel_ = cumtrapz(time,ampl);
%dis = cumtrapz(time,vel_);
%
%
%figure;
%% Plot the first subplot
%subplot(2, 1, 1);
%plot(time,accel_A3, '-.','LineWidth',1.);
%hold on 
%plot(Time, Acc, '--');
%title('Acceleration');
%xlabel('t (sec)');
%ylabel('Acc(t)');
%legend('Baseline Correction')
%
%% Plot the second subplot
%subplot(2, 1, 2);
%
%plot(time, disp_A3, '-.','LineWidth',1.);
%hold on 
%plot(Time, Disp, '--');
%title('Displacement');
%xlabel('t (sec)');
%ylabel('U(t)');
%legend('Baseline Correction')
%[T_o,Spa_o,Spv_o,Sd_o] = fns_response_spectra(data_t.time(3)-data_t.time(2),data_t.ampl,5,9.81,10);
%[T_p,Spa_p,Spv_p,Sd_p] = fns_response_spectra(data_t.time(3)-data_t.time(2),accel_A3,5,9.81,10);
%
%%% Plot Spectra
%figure;
%subplot(2,1,1)
%%figure('Name','Spectral Displacement','NumberTitle','off')
%semilogx(T_o,Sd_o,'LineWidth',2.)
%hold on 
%semilogx(T_p,Sd_p,'LineWidth',2.)
%grid on
%xlabel('Period (sec)','FontSize',13);
%ylabel('Sd (mm)','FontSize',13);
%title('Displacement Spectrum','FontSize',13)
%legend('Recorded','Simulated');
%
%subplot(2,1,2)
%%figure('Name','Pseudo Acceleration Spectrum','NumberTitle','off')
%semilogx(T_o,Spa_o,'LineWidth',2.)
%hold on 
%semilogx(T_p,Spa_p,'LineWidth',2.)
%grid on
%xlabel('Period (sec)','FontSize',13);
%ylabel('Spa (g)','FontSize',13);
%title('Pseudo Acceleration Spectrum','FontSize',13);
%legend('Recorded','Simulated');


