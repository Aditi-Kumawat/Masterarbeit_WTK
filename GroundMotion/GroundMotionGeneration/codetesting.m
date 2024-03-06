clear;
clc;
close all;

Y = load("SDOF_SBAGM_Y_Pred_1000.mat");

Val_1 = load("SDOF_SBAGM_Y_Pred_200_Num_1000.mat");

Val_2 = load("SDOF_SBAGM_Y_Pred_400_Num_1000.mat");
Val_3 = load("SDOF_SBAGM_Y_Pred_600_Num_1000.mat");
Val_4 = load("SDOF_SBAGM_Y_Pred_800_Num_1000.mat");
histogram((Y.Y(:,1)),'Normalization','pdf')
hold on 
histogram((Val_1.Y(:,1)),'Normalization','pdf')
histogram((Val_2.Y(:,1)),'Normalization','pdf')
histogram((Val_3.Y(:,1)),'Normalization','pdf')
histogram((Val_4.Y(:,1)),'Normalization','pdf')

legend('log(Y)', 'log(Y)|X = x1','log(Y)|X = x2','log(Y)|X = x3','log(Y)|X = x4');











