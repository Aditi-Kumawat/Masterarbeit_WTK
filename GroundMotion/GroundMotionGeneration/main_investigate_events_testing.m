clear;
clc;
close;

path = load('events.mat');
all_path = path.file_list;

%Events 9, 11 cannot work

statictic_info = [];

a = fns_import_time_data(all_path{1},'txt',[]);
GMG = cls_GM_generator(a, 100);
[~,FRF_info,Time_info,GM_info] = GMG.generateTimeNonStaPesudoGMbyFit("Hu_S0",[200,0.01,10],[],[],false);
statictic_info = [statictic_info ; [FRF_info,Time_info,GM_info]];

a = fns_import_time_data(all_path{2},'txt',[7,20]);
GMG = cls_GM_generator(a, 100);
[~,FRF_info,Time_info,GM_info] = GMG.generateTimeNonStaPesudoGMbyFit("Hu_S0",[200,0.01,10],[],[],false);
statictic_info = [statictic_info ; [FRF_info,Time_info,GM_info]];

a = fns_import_time_data(all_path{3},'txt',[2.5,20]);
GMG = cls_GM_generator(a, 100);
[~,FRF_info,Time_info,GM_info] = GMG.generateTimeNonStaPesudoGMbyFit("Hu_S0",[200,0.01,10],[],[],false);
statictic_info = [statictic_info ; [FRF_info,Time_info,GM_info]];

a = fns_import_time_data(all_path{4},'txt',[1,15]);
GMG = cls_GM_generator(a, 100);
[~,FRF_info,Time_info,GM_info] = GMG.generateTimeNonStaPesudoGMbyFit("Hu_S0",[200,0.01,10],[],[],false);
statictic_info = [statictic_info ; [FRF_info,Time_info,GM_info]];

a = fns_import_time_data(all_path{5},'txt',[]);
GMG = cls_GM_generator(a, 100);
[~,FRF_info,Time_info,GM_info] = GMG.generateTimeNonStaPesudoGMbyFit("Hu_S0",[200,0.01,10],[],[],false);
statictic_info = [statictic_info ; [FRF_info,Time_info,GM_info]];

a = fns_import_time_data(all_path{6},'txt',[],false);
GMG = cls_GM_generator(a, 100);
[~,FRF_info,Time_info,GM_info] = GMG.generateTimeNonStaPesudoGMbyFit("Hu_S0",[200,0.01,10],[],[],false);
statictic_info = [statictic_info ; [FRF_info,Time_info,GM_info]];

a = fns_import_time_data(all_path{7},'txt',[],false);
GMG = cls_GM_generator(a, 100);
[~,FRF_info,Time_info,GM_info] = GMG.generateTimeNonStaPesudoGMbyFit("Hu_S0",[200,0.01,10],[],[],false);
statictic_info = [statictic_info ; [FRF_info,Time_info,GM_info]];

a = fns_import_time_data(all_path{8},'txt',[],false);
GMG = cls_GM_generator(a, 100);
[~,FRF_info,Time_info,GM_info] = GMG.generateTimeNonStaPesudoGMbyFit("Hu_S0",[200,0.01,10],[],[],false);
statictic_info = [statictic_info ; [FRF_info,Time_info,GM_info]];

a = fns_import_time_data(all_path{10},'txt',[0,52],false);
GMG = cls_GM_generator(a, 100);
[~,FRF_info,Time_info,GM_info] = GMG.generateTimeNonStaPesudoGMbyFit("Hu_S0",[200,0.01,10],[],[],false);
statictic_info = [statictic_info ; [FRF_info,Time_info,GM_info]];

a = fns_import_time_data(all_path{12},'txt',[2.5,25],false);
GMG = cls_GM_generator(a, 100);
[~,FRF_info,Time_info,GM_info] = GMG.generateTimeNonStaPesudoGMbyFit("Hu_S0",[200,0.01,10],[],[],false);
statictic_info = [statictic_info ; [FRF_info,Time_info,GM_info]];



