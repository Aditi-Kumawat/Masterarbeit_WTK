clear;
clc;
close all;

%% Import time domain data
path = load('events.mat');
all_path = path.file_list;
data_t = fns_import_time_data(all_path{12},'txt',[1,25]);
data_f = fns_fft_data(data_t,100,false,false);

%% Initialize class
GMG = cls_GM_generator(data_t, 100);

%% FFT <-> PSD or ESD 
% Compute PSD by function: periodogram
    Fs = 200; %Sampling rate
    
    %Freq list, same as the one extracted from fns_fft_data()
    %If the length of list increase, the density will reduce, since it have to
    %divided with more frequency range
    Freq_list = data_f.Freq; 
    
    [psd_func,freq] = periodogram(data_t.ampl,[],Freq_list,Fs);

% Compute PSD by FFT
    %Compute the factor
    FFT_PSD_Fac = (1/(Fs*length(data_t.ampl)));

    % Using the Parseval's theorem
    % energy spectral density
    ESD = power(data_f.Ampl,2);

    % Power spectral density
    psd =  FFT_PSD_Fac * ESD;

figure;
plot(data_f.Freq,psd)
hold on 
plot(freq,psd_func,'-.')

xlabel('Freq (Hz)');
ylabel('PSD');
title('Compare PSD generated be two different method');
legend('periodogram()', 'FFT');

%% Autocorrealtion function and ESD    

%Compute autocorrealtion function by xcorr()
    auto_corr_ref= xcorr(data_t.ampl);
%Compute autocorrealtion function by convolution
    auto_corr_conv = conv(data_t.ampl, flipud(data_t.ampl));

figure;
% ESD not PSD here
plot(auto_corr_ref);
hold on 
plot(auto_corr_conv,'-.' )
xlabel('index');
ylabel('R(tau)');
title('Compare Autocorrealtion function ');
legend('reference', 'convolution');

% Theory: the FFT of autocorraltion function = ESD
    time = data_t.time;
    L = length(auto_corr_conv);
    
    ESD_auto_corr = fft(auto_corr_conv);
    HalfPost_ESD = ESD_auto_corr(1:floor(L/2+1));
    resample_freq = Fs*(0:(L/2))/L;

figure;
% ESD not PSD here
plot(data_f.Freq,power(data_f.Ampl,2));
hold on 
plot(resample_freq ,abs(HalfPost_ESD),'-.' )
xlabel('Freq (Hz)');
ylabel('PSD');
title('Compare PSD ');
legend('Amplitude ** 2', 'FFT(R(tau))');

