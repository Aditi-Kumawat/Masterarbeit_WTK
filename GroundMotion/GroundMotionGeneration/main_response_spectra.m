clear;
clc;
close;

seed = 2;
rng(seed);
path = load('events.mat');
all_path = path.file_list;

%Parameter evaluation
data_t = fns_import_time_data(all_path{12},'txt',[1,50]);
GMG = cls_GM_generator(data_t, 100);
[~,FRF_info,Time_info,GM_info] = GMG.generateTimeNonStaPesudoGMbyFit("Hu_S0",[200,0.01,0.5,GMG.PGA],[],[],false,false,true);

%Generate Pesuedo Ground Motioin based on those parameters
PesuedoGM = fns_generateGM_Params(seed,data_t.time,100,"Hu_S0",...
                          FRF_info,...
                          Time_info,...
                          GMG.AriasIntensity,1);

%Perfrom response spectra analysis
[T_o,Spa_o,Spv_o,Sd_o] = fns_response_spectra(data_t.time(3)-data_t.time(2),data_t.ampl,5,9.81,10);
[T_p,Spa_p,Spv_p,Sd_p] = fns_response_spectra(data_t.time(3)-data_t.time(2),PesuedoGM.ampl,5,9.81,10);

%% Plot Spectra
figure;
subplot(2,1,1)
%figure('Name','Spectral Displacement','NumberTitle','off')
semilogx(T_o,Sd_o,'LineWidth',2.)
hold on 
semilogx(T_p,Sd_p,'LineWidth',2.)
grid on
xlabel('Period (sec)','FontSize',13);
ylabel('Sd (mm)','FontSize',13);
title('Displacement Spectrum','FontSize',13)
legend('Recorded','Simulated');

subplot(2,1,2)
%figure('Name','Pseudo Acceleration Spectrum','NumberTitle','off')
semilogx(T_o,Spa_o,'LineWidth',2.)
hold on 
semilogx(T_p,Spa_p,'LineWidth',2.)
grid on
xlabel('Period (sec)','FontSize',13);
ylabel('Spa (g)','FontSize',13);
title('Pseudo Acceleration Spectrum','FontSize',13);
legend('Recorded','Simulated');