clear;
clc;
close all;

Y     = load("TEST3_Y_SBAGM_V6_VAL_RAND_DOE_1000_DIR_Z.mat");
Val_1 = load("TEST3_Y_SBAGM_V6_VAL_FIX550_DOE_1000_DIR_Z.mat");
Val_2 = load("TEST3_Y_SBAGM_V6_VAL_FIX650_DOE_1000_DIR_Z.mat");
Val_3 = load("TEST3_Y_SBAGM_V6_VAL_FIX750_DOE_1000_DIR_Z.mat");
Val_4 = load("TEST3_Y_SBAGM_V6_VAL_FIX850_DOE_1000_DIR_Z.mat");
histogram(exp(Y.Y(:,1)),'Normalization','pdf')
hold on 
histogram(exp(Val_1.Y(:,1)),'Normalization','pdf')
histogram(exp(Val_2.Y(:,1)),'Normalization','pdf')
histogram(exp(Val_3.Y(:,1)),'Normalization','pdf')
histogram(exp(Val_4.Y(:,1)),'Normalization','pdf')

legend('log(Y)', 'log(Y)|X = x1','log(Y)|X = x2','log(Y)|X = x3','log(Y)|X = x4');