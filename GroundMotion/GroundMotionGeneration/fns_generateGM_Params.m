function output_GM = fns_generateGM_Params(seed,Time_array,CutOffFreq,GM_model,GM_params,Time_Percentiles,AriasIntensity,energy_factor) 
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
    %   8.energy_factor   : Scaling the Arias Intensity. This is used for
    %                       matching specific ground motion. Defalut = 1. 
    %% Output data:
    %   1.output_GM       : Output table, time(var1) = Time series,
    %                                     ampl(var2) = Amplitude.

    %Initialized Seed for White Noise
    if isempty(seed)
        rng('shuffle');
    else
        rng(seed);
    end
    
    %Initialized and generateing pesuedo Data_t 
    ampl = ones(length(Time_array),1);
    time = Time_array;
    pesuedoData_t = table(time,ampl);
    
    %Initialized class
    GMG = cls_GM_generator(pesuedoData_t, CutOffFreq);

    %Generate filter
    S_init = 1;
    filter = GMG.GMmodel(GM_model,GM_params,S_init);
    FRF = sqrt(filter/S_init);
    norm_FRF = FRF;
    
  

    %Generate White Noise and transform to Freq domain
    noise = GMG.generateWhiteNoise;
    noise_FFT = fft(noise);
    P1 = noise_FFT(1:floor(length(noise)/2+1));

    
    %Apply FRF on White Noise
    PesudoGM_freq = transpose(P1).*norm_FRF;


    %IFFT, transform to time domain
    L = length(noise);
    P1_pad = [PesudoGM_freq; conj(flipud(PesudoGM_freq(2:end-1,:)))];
    P1_ifft = ifft(P1_pad*GMG.Fs, L, 1, 'symmetric');
    data_IFFT = P1_ifft(1:L,:)/GMG.Fs;
    time = GMG.Time;

    %normalized 
    original_variance = var(data_IFFT);
    data_IFFT = data_IFFT/(sqrt(original_variance));%max(abs(data_IFFT));
    ampl = data_IFFT - mean(data_IFFT);
    ampl(1) = 0;

    %Generate Time Modulating Function, making it time non-stationary.
    q = GMG.generateTimeModFunc(Time_Percentiles,[],AriasIntensity,energy_factor,false);
    ampl = q.*ampl; 

    %Output result
    output_GM = table(time,ampl);
end