clear;
clc;
close all;

Y = load("TESTING5_PRED_SDOF_SBAGM_Y_rand_1000.mat");

Val_1 = load("TESTING5_PRED_SDOF_SBAGM_Y_fix_200_Num_1000.mat");

Val_2 = load("TESTING5_PRED_SDOF_SBAGM_Y_fix_400_Num_1000.mat");
Val_3 = load("TESTING5_PRED_SDOF_SBAGM_Y_fix_600_Num_1000.mat");
Val_4 = load("TESTING5_PRED_SDOF_SBAGM_Y_fix_800_Num_1000.mat");
histogram((Y.Y(:,1)),'Normalization','pdf')
hold on 
histogram((Val_1.Y(:,1)),'Normalization','pdf')
histogram((Val_2.Y(:,1)),'Normalization','pdf')
histogram((Val_3.Y(:,1)),'Normalization','pdf')
histogram((Val_4.Y(:,1)),'Normalization','pdf')

legend('log(Y)', 'log(Y)|X = x1','log(Y)|X = x2','log(Y)|X = x3','log(Y)|X = x4');











