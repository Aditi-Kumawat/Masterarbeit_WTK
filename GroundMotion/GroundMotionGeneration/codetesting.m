clear;
clc;
close all;

Y = load("SDOF_AGM_Y_1000.mat");

Val_1 = load("SDOF_AGM_Y_Valid_200_Num_1000.mat");

Val_2 = load("SDOF_AGM_Y_Valid_400_Num_1000.mat");
Val_3 = load("SDOF_AGM_Y_Valid_600_Num_1000.mat");
Val_4 = load("SDOF_AGM_Y_Valid_800_Num_1000.mat");

histogram((Y.Y(:,1)),'Normalization','pdf')
hold on 
histogram(log(Val_1.Y(:,1)),'Normalization','pdf')
histogram(log(Val_2.Y(:,1)),'Normalization','pdf')
histogram(log(Val_3.Y(:,1)),'Normalization','pdf')
histogram(log(Val_4.Y(:,1)),'Normalization','pdf')

legend('log(Y)', 'log(Y)|X = x1','log(Y)|X = x2','log(Y)|X = x3','log(Y)|X = x4');









