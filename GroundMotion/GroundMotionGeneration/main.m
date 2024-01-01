clear;
clc;
close;

path = "C:\Users\v196m\Desktop\master_project\Masterarbeit\GroundMotion\RecordData\GM_2012_11_12_11_15_04\GM_INSH_2012_11_12_11_15_04\a_1_INSH_2012_11_12_11_15_04.txt";
a = fns_import_time_data(path,'txt');

%b = fns_fft_data(a,100,false,true);
%rng(1);
%s = rng;

GMG = cls_GM_generator(a, 100);


%GMG.HuKTmodel(50,0.5,0.211,2);
%c = GMG.fit_CPmodel([100,0.1,50,0.2]);
%c = GMG.fit_GM_model("CP",[100,0.2,20,0.2],true,true);
%c = GMG.fit_KTmodel([50,0.3],true);
%GMG.GMmodel("CP",c);
%GMG.generateWhiteNoise(true);
g = GMG.generateStaPesudoGM("CP",[100,0.2,20,0.1],'time',false,false);
time_percentiles = GMG.getPercentileInfo();
%q=GMG.generateTimeModFunc(time_percentiles);

PesuedoGM = GMG.generateTimeNonStaPesudoGM("CP",[100,0.2,20,0.1],true);


[T_o,Spa_o,Spv_o,Sd_o] = fns_response_spectra(a.time(3)-a.time(2),a.ampl,5,9.81,10);

[T_p,Spa_p,Spv_p,Sd_p] = fns_response_spectra(a.time(3)-a.time(2),PesuedoGM.ampl,5,9.81,10);


%% Plot Spectra
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