%% tesing of fn_rms_kb
clear;
clc;
close;

t = linspace(0,15,3001);
Fs = (t(end)-t(1))/length(t);

%% convolution validation
Conv_thoery = 0.5.*(exp(-t) + sin(t) - cos(t));
f1_valid = exp(-(t));
f2_valid = sin(t);
Conv_valid = conv(f1_valid,f2_valid)*Fs;

%% function: fn_rms_kb verificaiton
tau = 0.125;
f1 = exp(-(t)/tau);
f2 = sin(t);
KB_test = conv(f1,power(f2,2))*Fs;
KB_test_rms = sqrt(KB_test/tau);

%Import the function
KB_fun = fn_rms_kb(t, transpose(sin(t)), tau);


figure;
plot(t, Conv_valid(1:length(t)),'LineWidth',2);
hold on
plot(t, Conv_thoery,'--','LineWidth',2);
title('convolution validation')
legend('reference','conv result')


figure;
plot(t,KB_fun,'LineWidth',2);
hold on 
plot(t,KB_test_rms(1:length(t)),'--','LineWidth',2);
title('fn_rms_kb verificaiton')
legend('reference','fn rms kb')





