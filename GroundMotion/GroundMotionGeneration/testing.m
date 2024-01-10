clear;
clc;
close;
rng('shuffle');

%path = 'C:\Users\v196m\Desktop\master_project\Masterarbeit\GroundMotion\RecordData\GM_2012_11_12_11_15_04\GM_INS2_2012_11_12_11_15_04\a_1_INS2_2012_11_12_11_15_04.txt';
path = load('events.mat');
all_path = path.file_list;

%good
a = fns_import_time_data(all_path{12},'txt',[2,25],true);
GMG = cls_GM_generator(a, 100);
%GMG.fit_GM_model("CP_S0",[200,0.01,100,0.01,GMG.PGA],true,true);

%intense
%a = fns_import_time_data(all_path{2},'txt');
%GMG = cls_GM_generator(a, 100);
%GMG.fit_GM_model("CP_S0",[100,0.01,20,0.01,GMG.PGA],true,true);

%intense
%a = fns_import_time_data(all_path{3},'txt');
%GMG = cls_GM_generator(a, 100);
%GMG.fit_GM_model("CP_S0",[200,0.05,30,0.05,GMG.PGA],true,true);

%intense
%a = fns_import_time_data(all_path{4},'txt');
%GMG = cls_GM_generator(a, 100);
%GMG.fit_GM_model("CP_S0",[200,0.05,30,0.05,GMG.PGA],true,true);

%good
%a = fns_import_time_data(all_path{5},'txt');
%GMG = cls_GM_generator(a, 100);
%GMG.fit_GM_model("CP_S0",[200,0.05,30,0.05,GMG.PGA],true,true);

%good
%a = fns_import_time_data(all_path{6},'txt');
%GMG = cls_GM_generator(a, 100);
%GMG.fit_GM_model("CP_S0",[200,0.05,30,0.05,GMG.PGA],true,true);

%good
%a = fns_import_time_data(all_path{7},'txt');
%GMG = cls_GM_generator(a, 100);
%GMG.fit_GM_model("CP_S0",[200,0.05,30,0.05,GMG.PGA],true,true);

%good
%a = fns_import_time_data(all_path{8},'txt');
%GMG = cls_GM_generator(a, 100);
%GMG.fit_GM_model("CP_S0",[200,0.05,30,0.05,GMG.PGA],true,true);

%good
%a = fns_import_time_data(all_path{9},'txt');
%GMG = cls_GM_generator(a, 100);
%GMG.fit_GM_model("CP_S0",[200,0.05,50,0.05,GMG.PGA],true,true);

%good
%a = fns_import_time_data(all_path{10},'txt');
%GMG = cls_GM_generator(a, 100);
%GMG.fit_GM_model("CP_S0",[200,0.05,50,0.05,GMG.PGA],true,true);

%bad
%a = fns_import_time_data(all_path{11},'txt');
%GMG = cls_GM_generator(a, 100);
%GMG.fit_GM_model("CP_S0",[150,0.01,75,0.01,GMG.PGA],true,true);

%bad
%a = fns_import_time_data(all_path{12},'txt');
%GMG = cls_GM_generator(a, 100);
%GMG.fit_GM_model("CP_S0",[150,0.01,75,0.01,GMG.PGA],true,true);

%b = fns_fft_data(a,100,false,true);
%rng(2);


%GMG = cls_GM_generator(a, 100);


%GMG.HuKTmodel(50,0.5,0.211,2);
%c = GMG.fit_HuKTmodel_S0([100,0.1,1,PGA],true)
%filter_params = GMG.fit_GM_model("CP_S0",[200,0.01,100,0.01,PGA],true,true);
%GMG.fit_HuKTmodel_S0([200,0.1,1],true)
%GMG.GMmodel("KT_S0",filter_params,[],true);
%GMG.generateWhiteNoise(true);
%g = GMG.generateNormStaPesudoGM("CP_S0",[200,0.01,50,0.01,GMG.PGA],[],'time',true,true);
%time = GMG.Time;
%ampl = GMG.PGA*g.ampl;
%ampl(1) = 0 ;
%PesuedoGM = table(time,ampl);

%time_percentiles = GMG.getPercentileInfo(true);
%q=GMG.generateTimeModFunc(time_percentiles,[],[],1,true);

PesuedoGM = GMG.generateTimeNonStaPesudoGM("Hu_S0",[200,0.01,50],[],[],false,true,true);


%fns_fft_data(PesuedoGM,100,false,true);
%fns_fft_data(a,100,false,true);
%
%p = fns_generateGM_Params(2,a.time,100,"CP",...
%                          filter_params,...
%                          PGA,...
%                          time_percentiles,...
%                          AriasIntensity,1);
%

%figure;
%plot(PesuedoGM.time,PesuedoGM.ampl);
%hold on 
%plot(p.time,p.ampl,'--');


[T_o,Spa_o,Spv_o,Sd_o] = fns_response_spectra((a.time(3)-a.time(2)),a.ampl,5,9.81,10);
[T_p,Spa_p,Spv_p,Sd_p] = fns_response_spectra((a.time(3)-a.time(2)),PesuedoGM.ampl,5,9.81,10);

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
