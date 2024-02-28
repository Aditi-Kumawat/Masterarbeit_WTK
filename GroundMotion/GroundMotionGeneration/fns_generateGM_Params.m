function [time, accel_A3, vel_A3, disp_A3] = fns_generateGM_Params(seed,Time_array,CutOffFreq,GM_model,GM_params,Time_Percentiles,PGA) 
%% Generate Ground Motion By Input Paramter
    %% Input data:
    %   0.Seed            : Specific random seed, If [], then using 'shuffle'.
    %   1.Time_array      : Time series, size = [N,1];
    %   2.CutOffFreq      : Cut off frequency, It should be half of the Sampling
    %                     : Rate, CutOffFreq = Fs/2.
    %   3.GM_model        : "KT" = Kanai-Tajimi model.
    %                     : "CP" = Clough-Penzien model.
    %                     : "Hu" = Hu-Zhou model.
    %   4.GM_params       : if "KT", parameters should be [Omega_g, Damping_g]; 
    %                     : if "CP", parameters should be [Omega_g, Damping_g, Omega_c, Damping_c];
    %                     : if "Hu", parameters should be [Omega_g, Damping_g, Omega_c];
    %   5.S_init          : 1 
    %   6.Time_Percentiles: Time Percentiles of ground motion = 
    %                       [sec(0.01%),sec(5%),sec(45%),sec(95%)].
    %   7.AriasIntensity  : Arias Intensity = pi*trapz(power(Amplitude_t,2))/(2*9.81);
    %% Output data:
    %   1.output_GM       : Output table, time(var1) = Time series,
    %                                     ampl(var2) = Amplitude.
    addpath('BaselineCorrection\');

    %%Initialized Seed for White Noise
    %if isempty(seed)
    %    rng('shuffle');
    %else
    %    rng(seed);
    %end
    

    %Initialized and generateing pesuedo Data_t 
    
    if ~isempty(Time_array)
        time = Time_array;
    else
        time = transpose(0:0.005:15);
    end
    
    check_size = size(time);
    if check_size(1) == 1 && check_size(2) == length(time)
            time = transpose(time);
    end

    ampl = ones(length(time),1);
    pesuedoData_t = table(time,ampl);
    
    %Initialized class
    GMG = cls_GM_generator(seed,pesuedoData_t, CutOffFreq);

    %Generate filter
    filter = GMG.GMmodel(GM_model,GM_params);
    FRF = filter;

    
    %Generate White Noise and transform to Freq domain
    noise = GMG.generateWhiteNoise;
    noise_FFT = fft(noise);
    P1 = noise_FFT(1:floor(length(noise)/2+1));
    
    %Apply FRF on White Noise
    PesudoGM_freq = transpose(P1).*FRF;
    PesudoGM_freq(1:0) = 0;

    %IFFT, transform to time domain
    L = length(noise);
    P1_pad = [PesudoGM_freq; conj(flipud(PesudoGM_freq(2:end-1,:)))];
    P1_ifft = ifft(P1_pad*GMG.Fs, L, 1, 'symmetric');
    data_IFFT = P1_ifft(1:L,:)/GMG.Fs;
    time = GMG.Time;

    %normalized 
    original_variance = var(data_IFFT);
    data_IFFT = (data_IFFT - mean(data_IFFT))/(3*sqrt(original_variance));
    ampl = PGA*data_IFFT;

    %Generate Time Modulating Function, making it time non-stationary.
    q = GMG.generateTimeModFunc(Time_Percentiles,[]);
    ampl = q.*ampl;

    %Baseline correction
    %[~, disp] = newmarkIntegrate(time, ampl, 0.5, 0.25);
    %[DR, AR] = computeDriftRatio(time, disp, 'ReferenceAccel', ampl);
    
    [accel_A3, vel_A3, disp_A3] = baselineCorrection(time, ampl, 'AccelFitOrder', 3);  
    %[A3DR, A3AR] = computeDriftRatio(time, disp_A3, 'ReferenceAccel', ampl);=
    
end