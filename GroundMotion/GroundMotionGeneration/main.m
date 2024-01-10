clear;
clc;
close;
rng(2);

%path = 'C:\Users\v196m\Desktop\master_project\Masterarbeit\GroundMotion\RecordData\GM_2012_11_12_11_15_04\GM_INS2_2012_11_12_11_15_04\a_1_INS2_2012_11_12_11_15_04.txt';
path = load('events.mat');
all_path = path.file_list;

a = fns_import_time_data(all_path{12},'txt',[1,25]);
GMG = cls_GM_generator(a, 100);
time = GMG.getPercentileInfo(true);
q = GMG.generateTimeModFunc(time,[],[],1,true);
[PesuedoGM,FRF_info,Time_info,GM_info] = GMG.generateTimeNonStaPesudoGMbyFit("Hu_S0",[200,0.01,0.5,GMG.PGA],[],[],false,false,true);

%a = fns_import_time_data(all_path{2},'txt',[7,20],true);
%GMG = cls_GM_generator(a, 100);
%[~,FRF_info,Time_info,GM_info] = GMG.generateTimeNonStaPesudoGMbyFit("Hu_S0",[200,0.01,10],[],[],false,true,true);

%a = fns_import_time_data(all_path{3},'txt',[2.5,20],true);
%GMG = cls_GM_generator(a, 100);
%[~,FRF_info,Time_info,GM_info] = GMG.generateTimeNonStaPesudoGMbyFit("Hu_S0",[200,0.01,10],[],[],false,true,true);

%a = fns_import_time_data(all_path{4},'txt',[1,15],true);
%GMG = cls_GM_generator(a, 100);
%[~,FRF_info,Time_info,GM_info] = GMG.generateTimeNonStaPesudoGMbyFit("Hu_S0",[200,0.01,10],[],[],false,true,true);

%a = fns_import_time_data(all_path{5},'txt',[],true);
%GMG = cls_GM_generator(a, 100);
%[~,FRF_info,Time_info,GM_info] = GMG.generateTimeNonStaPesudoGMbyFit("Hu_S0",[200,0.01,10],[],[],false,true,true);

%a = fns_import_time_data(all_path{6},'txt',[],false);
%GMG = cls_GM_generator(a, 100);
%[~,FRF_info,Time_info,GM_info] = GMG.generateTimeNonStaPesudoGMbyFit("Hu_S0",[200,0.01,10],[],[],false,true,true);

%a = fns_import_time_data(all_path{7},'txt',[],false);
%GMG = cls_GM_generator(a, 100);
%[~,FRF_info,Time_info,GM_info] = GMG.generateTimeNonStaPesudoGMbyFit("Hu_S0",[200,0.01,10],[],[],false,true,true);

%a = fns_import_time_data(all_path{8},'txt',[],false);
%GMG = cls_GM_generator(a, 100);
%[~,FRF_info,Time_info,GM_info] = GMG.generateTimeNonStaPesudoGMbyFit("Hu_S0",[200,0.01,10],[],[],false,true,true);

% Cannot work
%a = fns_import_time_data(all_path{9},'txt',[],false);
%GMG = cls_GM_generator(a, 100);
%[~,FRF_info,Time_info,GM_info] = GMG.generateTimeNonStaPesudoGMbyFit("Hu_S0",[200,0.01,10],[],[],false,true,true);

%a = fns_import_time_data(all_path{10},'txt',[0,52],false);
%GMG = cls_GM_generator(a, 100);
%[~,FRF_info,Time_info,GM_info] = GMG.generateTimeNonStaPesudoGMbyFit("Hu_S0",[200,0.01,10],[],[],false,true,true);

% Cannot work
%a = fns_import_time_data(all_path{11},'txt',[],false);
%GMG = cls_GM_generator(a, 100);
%[~,FRF_info,Time_info,GM_info] = GMG.generateTimeNonStaPesudoGMbyFit("Hu_S0",[200,0.01,10],[],[],false,true,true);

%a = fns_import_time_data(all_path{12},'txt',[2.5,25],false);
%GMG = cls_GM_generator(a, 100);
%[~,FRF_info,Time_info,GM_info] = GMG.generateTimeNonStaPesudoGMbyFit("Hu_S0",[200,0.01,10],[],[],false,true,true);


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




%fns_fft_data(PesuedoGM,100,false,true);
%fns_fft_data(a,100,false,true);
%


%p = fns_generateGM_Params(2,a.time,100,"Hu_S0",...
%                          FRF_info,...
%                          Time_info,...
%                          GMG.AriasIntensity,1);
%
%
%figure;
%plot(PesuedoGM.time,PesuedoGM.ampl);
%hold on 
%plot(p.time,p.ampl,'--');


