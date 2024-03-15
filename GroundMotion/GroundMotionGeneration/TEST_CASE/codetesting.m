clear;
clc;
close all;
addpath(".\REALIZATION\")

Y     = load("TEST1_Y_SBAGM_V6_VAL_RAND_DOE_1000_DIR_Z.mat");
Val_1 = load("TEST1_Y_SBAGM_V6_VAL_FIX200_DOE_1000_DIR_Z.mat");
Val_2 = load("TEST1_Y_SBAGM_V6_VAL_FIX400_DOE_1000_DIR_Z.mat");
Val_3 = load("TEST1_Y_SBAGM_V6_VAL_FIX600_DOE_1000_DIR_Z.mat");
Val_4 = load("TEST1_Y_SBAGM_V6_VAL_FIX800_DOE_1000_DIR_Z.mat");
histogram((Y.Y(:,1)),'Normalization','pdf')
hold on 
histogram((Val_1.Y(:,1)),'Normalization','pdf')
histogram((Val_2.Y(:,1)),'Normalization','pdf')
histogram((Val_3.Y(:,1)),'Normalization','pdf')
histogram((Val_4.Y(:,1)),'Normalization','pdf')

legend('log(Y)', 'log(Y)|X = x1','log(Y)|X = x2','log(Y)|X = x3','log(Y)|X = x4');













