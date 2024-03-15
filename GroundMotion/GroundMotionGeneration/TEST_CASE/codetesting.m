clear;
clc;
close all;
%addpath(".\REALIZATION\")
%Y = load("TEST0_Y_SBAGM_V5_VAL_RAND_DOE_1000_DIR_Z.mat");
%
%Val_1 = load("TEST0_Y_SBAGM_V5_VAL_FIX200_DOE_1000_DIR_Z.mat");
%Val_2 = load("TEST0_Y_SBAGM_V5_VAL_FIX400_DOE_1000_DIR_Z.mat");
%Val_3 = load("TEST0_Y_SBAGM_V5_VAL_FIX600_DOE_1000_DIR_Z.mat");
%Val_4 = load("TEST0_Y_SBAGM_V5_VAL_FIX800_DOE_1000_DIR_Z.mat");
%histogram((Y.Y(:,1)),'Normalization','pdf')
%hold on 
%histogram((Val_1.Y(:,1)),'Normalization','pdf')
%histogram((Val_2.Y(:,1)),'Normalization','pdf')
%histogram((Val_3.Y(:,1)),'Normalization','pdf')
%histogram((Val_4.Y(:,1)),'Normalization','pdf')
%
%legend('log(Y)', 'log(Y)|X = x1','log(Y)|X = x2','log(Y)|X = x3','log(Y)|X = x4');

X = 7.5*rand(50000,1)+8;
Y = 7.5*rand(50000,1)+8;
dis = sqrt(power(X,2) + power(Y,2));











