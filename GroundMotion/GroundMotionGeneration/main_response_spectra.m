clear;
clc;
close;

path = load('Statistic_info_type_all_Z_nine.mat');
all_path = path.statictic_info_type2;

%Parameter evaluation
data_t = fns_import_time_data(all_path{1}{1},'txt',[]);
GMG = cls_GM_generator(data_t, 100);
[GM,FRF_info,Time_info,GM_info] = GMG.generateTimeNonStaPesudoGMbyFit("Hu_S0",[200,1,0.5,GMG.PGA],[]);

%Generate Pesuedo Ground Motioin based on those parameters
%[time,ampl,vel,disp] = fns_generateGM_Params([],data_t.time,100,"Hu_S0",...
%                          FRF_info,...
%                          Time_info,...
%                          GMG.AriasIntensity);

%Perfrom response spectra analysis
[T_o,Spa_o,Spv_o,Sd_o] = fns_response_spectra(data_t.time(3)-data_t.time(2),data_t.ampl,5,9.81,10);
[T_p,Spa_p,Spv_p,Sd_p] = fns_response_spectra(GM.time(3)-GM.time(2),GM.ampl,5,9.81,10);

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