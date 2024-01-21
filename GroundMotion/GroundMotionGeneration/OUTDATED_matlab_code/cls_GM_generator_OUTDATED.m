classdef cls_GM_generator
%% Ground Motion Generator
    %% Initialization: 
    % 1. Input data: the data type should be table with size [N,2],
    %    N is the length of time series. Could be extract from function: 
    %    fns_import_time_domain().
    % 2. CutOffFreq: It should be half of the Sampling Rate, 
    %                CutOffFreq = Fs/2.
    %% Method:
    % 1.generateWhiteNoise(plot_):
    %   Generating Gaussain White Noise.
    %   Input:
    %       1. plot_: bool. Plot the white noise in time domain.
    %   Output:
    %       1. Table: (time,amplitude), size [N,2].
    %
    % 2.generateStaPesudoGM(GM_model,init_Guess,S_init,domain_type,show_info_, plot_)
    %   Generate Simulated Stationary Ground Motion. 
    %   Input:
    %       1.GM_model:  if "KT" = Kanai-Tajimi model.
    %                    if "CP" = Clough-Penzien model.
    %                    if "Hu" = Hu-Zhou model.
    %       2.GM_params: if "KT", parameters should be [Omega_g, Damping_g]; 
    %                    if "CP", parameters should be [Omega_g, Damping_g, Omega_c, Damping_c];
    %                    if "Hu", parameters should be [Omega_g, Damping_g, Omega_c];
    %                    Good Guess: Omega_g = arg(max(Amplitude_freq)),
    %                                Damping_g = 0.3,
    %                                Omega_c = Omega_g/5,
    %                                Damping_g = 0.3.  
    %       3.S_init: Peak Ground Acceleration (m/s^2). if [], then using
    %                 the PGA of recorded ground motion.
    %       4.domain_type: 'time' or 'freq' 
    %       5.show_info_ and plot_: bool.
    %   Output:
    %       1. if domain_type = 'time', output Table: (time,amplitude), size [N,2].
    %          if domain_type = 'freq', output Table: (freq,complexvalue), size [N,2].
    %   Reference: Chen (2022) Power spectral models of stationary earthquake-induced ground
    %              motion process considering site characteristics
    %
    % 3.generateTimeNonStaPesudoGM(GM_model,init_Guess,S_init,AriasIntensity,energy_corrected_bool,show_info_, plot_)
    %   Generate Simulated non-Stationary Ground Motion. (Only Time non-stationary).
    %   Input:
    %       1. GM_model, init_Guess, S_init: same as parameters for obj.generateStaPesudoGM()
    %       2. AriasIntensity: Arias Intensity = pi*trapz(power(Amplitude_t,2))/(2*9.81);
    %                          if [], then automatically compute from recorded Ground motion.
    %       3. energy_corrected_bool: bool, if true, then it will automatically
    %                                 compute then scale the generated ground
    %                                 motion. if false, then factor = 1.
    %                                 Reference:  Broccardo (2017) A spectral-based stochastic ground motion model
    %                                             with a non-parametric time-modulating function
    %       4. show_info_ and plot_: bool.
    %   Output: 
    %       1. output Table: (time,amplitude), size [N,2].

        
    properties
        Time;
        Ampl_t;
        PGA;

        Freq;
        Real_f;
        Imag_f;
        Cmlx_f;
        Ampl_f;

        Fs;
        KT_filter;
        CutOffFreq;
        Rad_Freq;

        PowerSpectrum;
        AriasIntensity
    end
    
    methods
        function obj = cls_GM_generator(data_t, cut_off_freq)

            [data_f,Fs] = fns_fft_data(data_t,cut_off_freq);  
            obj.CutOffFreq = cut_off_freq;

            obj.Time = data_t.time;
            obj.Ampl_t = data_t.ampl;
            obj.PGA = max(abs(data_t.ampl));

            obj.Freq = data_f.Freq;
            obj.Rad_Freq = 2*pi*obj.Freq;
            obj.Real_f = data_f.Real;
            obj.Imag_f = data_f.Imag;
            obj.Cmlx_f = data_f.Cmlx;
            obj.Ampl_f = data_f.Ampl;
            obj.Fs = Fs;
            %obj.PowerSpectrum = (1/(200*length(obj.Ampl_t)))*power(obj.Ampl_f,2); 
            %                  = periodogram(obj.Ampl_t,[],obj.Freq,obj.Fs); 
            obj.PowerSpectrum = power(obj.Ampl_f,2);  
            obj.AriasIntensity = pi*trapz(power(obj.Ampl_t,2))/(2*9.81);
        end
    
        function noise = generateWhiteNoise(obj,plot_)

            if nargin < 2
                plot_ = false;
            end         
            mean_value = 0;  % Set your desired mean
            std_deviation = 1;  % Set your desired standard deviation

            % Generate white Gaussian noise
            noise = mean_value + std_deviation * randn(1,length(obj.Time));

            %% Design a low-pass filter with a cutoff frequency
            %cutoff_frequency = obj.CutOffFreq / (0.5 * obj.Fs)-0.01; % Normalize the cutoff frequency
            %filter_order = 50;
            %low_pass_filter = fir1(filter_order, cutoff_frequency);
            %
            %% Apply the filter to the white noise
            %bandlimited_noise = filter(low_pass_filter, 1, noise);
            %  
            %% Scale and offset the noise to have a maximum amplitude in [-1, 1]
            %noise = bandlimited_noise;%/max(abs(bandlimited_noise));

            if plot_
                figure;
                plot(obj.Time, noise);
                title('Band-Limited White Gaussian Noise');
                xlabel('Time (seconds)');
                ylabel('Amplitude');
            end
        end

        function [output_GM,coeffs] = generateNormStaPesudoGM(obj,GM_model,init_Guess,S_init,domain_type,show_info_, plot_)
            domain_type_list = {'time','freq'};
            if ~ismember(domain_type,  domain_type_list)
                fprintf('ERROR! Wrong input type, please input one of the type: [%s]\n', strjoin(data_type_list, ', '));
            end

            if nargin < 6
                show_info_ = false;
            end

            if nargin < 7
                plot_ = false;
            end            
            
            %Fitting model
            [coeffs,Flag] = fit_GM_model(obj,GM_model,init_Guess,show_info_,false);

            % if Flag ~= 1, not convegence
            if Flag ~= 1
               coeffs(1) = -1; 
            end

            if isempty(S_init)
                filter = GMmodel(obj,GM_model,coeffs,[]);
                if strcmp(GM_model,"Hu") || strcmp(GM_model,"KT") || strcmp(GM_model,"CP") || strcmp(GM_model,"Hu_S0")
                    S_init = obj.PGA;
                else
                    S_init = coeffs(end);
                end
            else
                filter = GMmodel(obj,GM_model,coeffs,S_init);

            end

            %Convert to omega_f for representation
            if  strcmp(GM_model,"Hu_S0")
                coeffs(3) = coeffs(3);%*coeffs(1);
            end

            %Generate white noise
            noise = obj.generateWhiteNoise;
            noise_FFT = fft(noise);
            P1 = noise_FFT(1:floor(length(noise)/2+1));
            freq = obj.Fs*(0:(length(noise)/2))/length(noise);

            %Generate FRF
            FRF = sqrt(filter/S_init);
            norm_FRF = FRF;
            
   
            try
                if length(FRF)~=length(P1)
                    error('Lengths of FRF and Noise_FFT are not equal.');
                end
    
                if length(P1)~=length(freq)
                    error('Lengths of Freq and Noise_FFT are not equal.');
                end

                %Generate ground motion by appling FRF on white noise.
                PesudoGM_freq = transpose(P1).*norm_FRF;
                PesudoGM_freq(1:10) = 0;


                if strcmp(domain_type,'freq')
                    output_GM = table(freq,cmplx);

                    if plot_
                        figure;
                        plot(freq,norm_FRF);
                        title('Fitting result');
                        xlabel('Freq (Hz)');
                        ylabel('S(w)/S0');
                        legend("fitted filter");
                        figure;
                        plot(freq,abs(PesudoGM_freq));
                        title('Pesudo Ground Motion in freq-domain');
                        xlabel('Freq (Hz)');
                        ylabel('Amplitude');
                    end
                end
                
                if strcmp(domain_type,'time')
                    %IFFT transform back to time domain.
                    L = length(noise);
                    P1_pad = [PesudoGM_freq; conj(flipud(PesudoGM_freq(2:end-1,:)))];
                    P1_ifft = ifft(P1_pad*obj.Fs, L, 1, 'symmetric');
                    data_IFFT = P1_ifft(1:L,:)/obj.Fs;

                    time = obj.Time;
                    %normalized 
                    original_variance = var(data_IFFT);
                    data_IFFT = data_IFFT/(sqrt(original_variance));%max(abs(data_IFFT));
                    ampl = data_IFFT - mean(data_IFFT);
                    ampl(1) = 0;
                    output_GM = table(time,ampl);

                    if plot_
                        figure;
                        plot(obj.Time,data_IFFT);
                        title('Pesudo Ground Motion in time-domain');
                        xlabel('time (s)');
                        ylabel('Amplitude');   

                        figure;
                        plot(freq,norm_FRF);
                        title('Fitting result');
                        xlabel('Freq (Hz)');
                        ylabel('Frequency Response Function (Hz)');
                        legend("fitted filter");
                        figure;
                        plot(freq,abs(PesudoGM_freq));
                        title('Pesudo Ground Motion in freq-domain');
                        xlabel('Freq (Hz)');
                        ylabel('Amplitude');
                    end
                end



            catch exception
                disp(['Error: ' exception.message]);
            end
            
        end

        function [output_GM,frequency_coeffs,time_coeffs,GM_info] = generateTimeNonStaPesudoGMbyFit(obj,GM_model,init_Guess,S_init,AriasIntensity,energy_corrected_bool,show_info_, plot_)
            if isempty(AriasIntensity)
                AriasIntensity = obj.AriasIntensity;
            end

            if nargin < 6
                energy_corrected_bool = false;
            end
            
            if nargin < 7
                show_info_ = false;
            end

            if nargin < 8
                plot_ = false;
            end
            
            if isempty(S_init)
                [stationaryGM,coeffs] = generateNormStaPesudoGM(obj,GM_model,init_Guess,[],"time",show_info_, plot_);
            else
                [stationaryGM,coeffs] = generateNormStaPesudoGM(obj,GM_model,init_Guess,S_init,"time",show_info_, plot_);
            end
            %Get Percentile from recorded ground motion.
            time_percentiles = getPercentileInfo(obj,false);
            %Generate Time Modulating Function.
            q = generateTimeModFunc(obj,time_percentiles,[],AriasIntensity,1,false);
 
            time = obj.Time;
            %Generate simulated ground motion
            ampl_ = q.*stationaryGM.ampl;

            %Correct Arias Intensity or not (Broccardo, 2017)
            if energy_corrected_bool
                simulate_AI = pi*trapz(power(ampl_,2))/(2*9.81);

                %Correct Energy
                energy_factor = AriasIntensity/simulate_AI;
                K = energy_factor*ones(length(ampl_),1);

                %Recompute the Time Modulating Function.
                q = generateTimeModFunc(obj,time_percentiles,[],AriasIntensity,energy_factor,false);
                ampl = q.*stationaryGM.ampl;           
                output_GM = table(time,ampl,K);
            else
                ampl = ampl_;
                energy_factor = 1;
                K = energy_factor*ones(length(ampl),1);
                output_GM = table(time,ampl,K);
            end

            %Output coeffs
            frequency_coeffs = coeffs;
            time_coeffs = time_percentiles;
            GM_info = [AriasIntensity,obj.Time(end),obj.Fs];

            if show_info_
                disp('----------Time Modulating Function-----------');
                disp(['AriasIntensity = ',num2str(AriasIntensity)]);
                disp(['EnergyFactor,K = ',num2str(energy_factor)]);
                disp(['Time      1%,P = ',num2str(time_percentiles(1)),' sec']);
                disp(['Time      5%,P = ',num2str(time_percentiles(2)),' sec']);
                disp(['Time     45%,P = ',num2str(time_percentiles(3)),' sec']);
                disp(['Time     95%,P = ',num2str(time_percentiles(4)),' sec']);
            end

            if plot_
                figure;
                plot(obj.Time,obj.Ampl_t,'Color',[0.2, 0.4, 0.8, 0.5]);
                hold on 
                plot(obj.Time,ampl,'Color',[0.9290 0.6940 0.1250,0.3]);
                title('Non-Stationary Pesudo Ground Motion in Time-domain');
                xlabel('Time (s)');
                ylabel('Amplitude');
                legend("Recored Ground Motion","Non-Stationary Ground Motion");

                plot_time = obj.Time(obj.Time<=time_percentiles(4)+10);

                figure;
                plot(plot_time,pi*cumtrapz(...
                    power(obj.Ampl_t(obj.Time<=time_percentiles(4)+10),2))/(2*9.81));
                hold on 
                plot(plot_time,pi*cumtrapz(...
                    power(ampl(obj.Time<=time_percentiles(4)+10),2))/(2*9.81));
                title('Cumulative Arias Intensity Fitting');
                xlabel('Time (s)');
                ylabel('Arias Intensity');
                legend("Recorded Arias Intensity",...
                    ['Simulated Arias Intensity, K =',num2str(energy_factor)]);

            end
        end

        function filter = GMmodel(obj,model_type,filter_para,S_init,plot_)
            %   1.model_type      : "KT" = Kanai-Tajimi model.
            %                     : "CP" = Clough-Penzien model.
            %                     : "Hu" = Hu-Zhou model.
            %   2.filter_para     : if "KT", parameters should be [Omega_g, Damping_g]; 
            %                     : if "CP", parameters should be [Omega_g, Damping_g, Omega_c, Damping_c];
            %                     : if "Hu", parameters should be [Omega_g, Damping_g, Omega_c];
            %   3.S_init          : Peak Ground Acceleration (m/s^2).
            %   4.plot_           : Bool, plot or not.
            %
            %   Reference: Chen(2022),Power spectral models of stationary earthquake-induced ground
            %                         motion process considering site characteristics

            model_type_list = {'Hu','KT','CP','Hu_S0','KT_S0','CP_S0'};
            if ~ismember(model_type, model_type_list)
                fprintf('ERROR! Wrong input type, please input one of the type: [%s]\n', strjoin(data_type_list, ', '));
            end
            
            if nargin < 4 || isempty(S_init)
                S_init = obj.PGA;
            end

            if strcmp(model_type,'Hu') 
                Omega_g=filter_para(1);
                Beta_g=filter_para(2);
                Omega_c=filter_para(3);

                filter = HuKTmodel(obj,Omega_g,Beta_g,S_init,Omega_c);
            end

            if strcmp(model_type,'CP') 
                Omega_g=filter_para(1);
                Beta_g=filter_para(2);
                Omega_f=filter_para(3);
                Beta_f=filter_para(4);

                filter = CPmodel(obj,Omega_g,Beta_g,S_init,Omega_f,Beta_f);
            end

            if strcmp(model_type,'KT')
                Omega_g=filter_para(1);
                Beta_g=filter_para(2);

                filter = KTmodel(obj,Omega_g,Beta_g,S_init);
            end

            if strcmp(model_type,'Hu_S0') 
                Omega_g=filter_para(1);
                Beta_g=filter_para(2);
                Omega_c=filter_para(3)*filter_para(1);

                filter = HuKTmodel(obj,Omega_g,Beta_g,S_init,Omega_c);
            end

            if strcmp(model_type,'CP_S0') 
                Omega_g=filter_para(1);
                Beta_g=filter_para(2);
                Omega_f=filter_para(3);
                Beta_f=filter_para(4);
                if isempty(S_init)
                    S_init = filter_para(5);
                end

                filter = CPmodel(obj,Omega_g,Beta_g,S_init,Omega_f,Beta_f);
            end

            if strcmp(model_type,'KT_S0')
                Omega_g=filter_para(1);
                Beta_g=filter_para(2);
                if isempty(S_init)
                    S_init = filter_para(3);
                end

                filter = KTmodel(obj,Omega_g,Beta_g,S_init);
            end

            if nargin < 5
                plot_ = false;  
            end

            if plot_
                figure;
                plot(obj.Rad_Freq,filter);
                title('Fitting result');
                xlabel('radius Freq');
                ylabel('S(w)');
                legend("Unscaling PowerSpectrum")
            end
        end

        function filter = HuKTmodel(obj,Omega_g, Beta_g, S_init, Omega_c)
            rad_Freq = 2*pi*obj.Freq;
            HighPass_filter = power(rad_Freq,6)./(power(rad_Freq,6)+power(Omega_c,6));
            filter = S_init*HighPass_filter.*...
                (power(Omega_g,4) + power(2*Omega_g*Beta_g*rad_Freq,2))./...
                (power(power(Omega_g,2)-power(rad_Freq,2),2)+power(2*Omega_g*Beta_g*rad_Freq,2));
        end

        function filter = CPmodel(obj,Omega_g, Beta_g, S_init, Omega_f,Beta_f)
            rad_Freq = 2*pi*obj.Freq;
            HighPass_filter = power(rad_Freq,4)./...
                (power(power(Omega_f,2)-power(rad_Freq,2),2)+power(2*Omega_f*Beta_f*rad_Freq,2));
            filter = S_init*HighPass_filter.*...
                (power(Omega_g,4) + power(2*Omega_g*Beta_g*rad_Freq,2))./...
                (power(power(Omega_g,2)-power(rad_Freq,2),2)+power(2*Omega_g*Beta_g*rad_Freq,2));
            
        end

        function filter = KTmodel(obj,Omega_g, Beta_g, S_init)
            rad_Freq = 2*pi*obj.Freq;
            filter = S_init*(power(Omega_g,4) + power(2*Omega_g*Beta_g*rad_Freq,2))./...
                (power(power(Omega_g,2)-power(rad_Freq,2),2)+power(2*Omega_g*Beta_g*rad_Freq,2));
        end

        function coeffs = fit_GM_model_OutOfDate(obj,model_type,init_Guess,show_info_,plot_)
            model_type_list = {'Hu','KT','CP','Hu_S0','KT_S0','CP_S0'};
            if ~ismember(model_type, model_type_list)
                fprintf('ERROR! Wrong input type, please input one of the type: [%s]\n', strjoin(data_type_list, ', '));
            end

            if nargin < 4
                show_info_ = false;
            end

            if nargin < 5
                plot_ = false;
            end

            if strcmp(model_type,'Hu') 
                [coeffs,interval] = fit_HuKTmodel(obj,init_Guess,plot_);
                if show_info_ 
                    disp('----------------Fitting result---------------');
                    disp(['S_init  = ',num2str(obj.PGA),', PGA, exract from recorded GM directly']);
                    disp(['Omega_g = ',num2str(coeffs(1)),', 95% confidence intervals: ',...
                        '(',num2str(interval(1,1)),', ',num2str(interval(2,1)),').']);
                    disp(['Beta_g  = ',num2str(coeffs(2)),', 95% confidence intervals: ',...
                        '(',num2str(interval(1,2)),', ',num2str(interval(2,2)),').']);
                    disp(['Omega_c = ',num2str(coeffs(3)),', 95% confidence intervals: ',...
                        '(',num2str(interval(1,3)),', ',num2str(interval(2,3)),').']);
                end
            end

            if strcmp(model_type,'CP')
                [coeffs,interval] = fit_CPmodel(obj,init_Guess,plot_);

                if show_info_ 
                    disp('----------------Fitting result---------------');
                    disp(['S_init  = ',num2str(obj.PGA),', PGA, exract from recorded GM directly']);
                    disp(['Omega_g = ',num2str(coeffs(1)),', 95% confidence intervals: ',...
                        '(',num2str(interval(1,1)),', ',num2str(interval(2,1)),').']);
                    disp(['Beta_g  = ',num2str(coeffs(2)),', 95% confidence intervals: ',...
                        '(',num2str(interval(1,2)),', ',num2str(interval(2,2)),').']);
                    disp(['Omega_f = ',num2str(coeffs(3)),', 95% confidence intervals: ',...
                        '(',num2str(interval(1,3)),', ',num2str(interval(2,3)),').']);
                    disp(['Beta_f  = ',num2str(coeffs(4)),', 95% confidence intervals: ',...
                        '(',num2str(interval(1,4)),', ',num2str(interval(2,4)),').']);
                end
            end

            if strcmp(model_type,'KT')
                [coeffs,interval] = fit_KTmodel(obj,init_Guess,plot_);

                if show_info_ 
                    disp('----------------Fitting result---------------');
                    disp(['S_init  = ',num2str(obj.PGA),', PGA, exract from recorded GM directly']);
                    disp(['Omega_g = ',num2str(coeffs(1)),', 95% confidence intervals: ',...
                        '(',num2str(interval(1,1)),', ',num2str(interval(2,1)),').']);
                    disp(['Beta_g  = ',num2str(coeffs(2)),', 95% confidence intervals: ',...
                        '(',num2str(interval(1,2)),', ',num2str(interval(2,2)),').']);
                end
            end

            if strcmp(model_type,'Hu_S0') 
                [coeffs,interval] = fit_HuKTmodel_S0(obj,init_Guess,plot_);
                if show_info_ 
                    disp('----------------Fitting result---------------');
                    disp(['PGA     = ',num2str(obj.PGA),', PGA, exract from recorded GM directly']);
                    disp(['Omega_g = ',num2str(coeffs(1)),', 95% confidence intervals: ',...
                        '(',num2str(interval(1,1)),', ',num2str(interval(2,1)),').']);
                    disp(['Beta_g  = ',num2str(coeffs(2)),', 95% confidence intervals: ',...
                        '(',num2str(interval(1,2)),', ',num2str(interval(2,2)),').']);
                    disp(['Omega_c = ',num2str(coeffs(3)),', 95% confidence intervals: ',...
                        '(',num2str(interval(1,3)),', ',num2str(interval(2,3)),').']);
                    disp(['S_init  = ',num2str(coeffs(4)),', 95% confidence intervals: ',...
                        '(',num2str(interval(1,4)),', ',num2str(interval(2,4)),').']);
                end
            end

            if strcmp(model_type,'CP_S0')
                [coeffs,interval] = fit_CPmodel_S0(obj,init_Guess,plot_);

                if show_info_ 
                    disp('----------------Fitting result---------------');
                    disp(['PGA     = ',num2str(obj.PGA),', PGA, exract from recorded GM directly']);
                    disp(['Omega_g = ',num2str(coeffs(1)),', 95% confidence intervals: ',...
                        '(',num2str(interval(1,1)),', ',num2str(interval(2,1)),').']);
                    disp(['Beta_g  = ',num2str(coeffs(2)),', 95% confidence intervals: ',...
                        '(',num2str(interval(1,2)),', ',num2str(interval(2,2)),').']);
                    disp(['Omega_f = ',num2str(coeffs(3)),', 95% confidence intervals: ',...
                        '(',num2str(interval(1,3)),', ',num2str(interval(2,3)),').']);
                    disp(['Beta_f  = ',num2str(coeffs(4)),', 95% confidence intervals: ',...
                        '(',num2str(interval(1,4)),', ',num2str(interval(2,4)),').']);
                    disp(['S_init  = ',num2str(coeffs(5)),', 95% confidence intervals: ',...
                        '(',num2str(interval(1,5)),', ',num2str(interval(2,5)),').']);                   
                end
            end

            if strcmp(model_type,'KT_S0')
                [coeffs,interval] = fit_KTmodel_S0(obj,init_Guess,plot_);

                if show_info_ 
                    disp('----------------Fitting result---------------');
                    disp(['PGA     = ',num2str(obj.PGA),', PGA, exract from recorded GM directly']);
                    disp(['Omega_g = ',num2str(coeffs(1)),', 95% confidence intervals: ',...
                        '(',num2str(interval(1,1)),', ',num2str(interval(2,1)),').']);
                    disp(['Beta_g  = ',num2str(coeffs(2)),', 95% confidence intervals: ',...
                        '(',num2str(interval(1,2)),', ',num2str(interval(2,2)),').']);
                    disp(['S_init  = ',num2str(coeffs(3)),', 95% confidence intervals: ',...
                        '(',num2str(interval(1,3)),', ',num2str(interval(2,3)),').']);
                end
            end
        end
    
        function [coeffs,Flag] = fit_GM_model(obj,model_type,init_Guess,show_info_,plot_)
            model_type_list = {'Hu','KT','CP','Hu_S0','KT_S0','CP_S0'};
            if ~ismember(model_type, model_type_list)
                fprintf('ERROR! Wrong input type, please input one of the type: [%s]\n', strjoin(data_type_list, ', '));
            end

            if nargin < 4
                show_info_ = false;
            end

            if nargin < 5
                plot_ = false;
            end

            Flag = 0;

            if strcmp(model_type,'Hu') 
                [coeffs,interval] = fit_HuKTmodel(obj,init_Guess,plot_);
                if show_info_ 
                    disp('----------------Fitting result---------------');
                    disp(['S_init  = ',num2str(obj.PGA),', PGA, exract from recorded GM directly']);
                    disp(['Omega_g = ',num2str(coeffs(1)),', 95% confidence intervals: ',...
                        '(',num2str(interval(1,1)),', ',num2str(interval(2,1)),').']);
                    disp(['Beta_g  = ',num2str(coeffs(2)),', 95% confidence intervals: ',...
                        '(',num2str(interval(1,2)),', ',num2str(interval(2,2)),').']);
                    disp(['Omega_c = ',num2str(coeffs(3)),', 95% confidence intervals: ',...
                        '(',num2str(interval(1,3)),', ',num2str(interval(2,3)),').']);
                end
            end

            if strcmp(model_type,'CP')
                [coeffs,interval] = fit_CPmodel(obj,init_Guess,plot_);

                if show_info_ 
                    disp('----------------Fitting result---------------');
                    disp(['S_init  = ',num2str(obj.PGA),', PGA, exract from recorded GM directly']);
                    disp(['Omega_g = ',num2str(coeffs(1)),', 95% confidence intervals: ',...
                        '(',num2str(interval(1,1)),', ',num2str(interval(2,1)),').']);
                    disp(['Beta_g  = ',num2str(coeffs(2)),', 95% confidence intervals: ',...
                        '(',num2str(interval(1,2)),', ',num2str(interval(2,2)),').']);
                    disp(['Omega_f = ',num2str(coeffs(3)),', 95% confidence intervals: ',...
                        '(',num2str(interval(1,3)),', ',num2str(interval(2,3)),').']);
                    disp(['Beta_f  = ',num2str(coeffs(4)),', 95% confidence intervals: ',...
                        '(',num2str(interval(1,4)),', ',num2str(interval(2,4)),').']);
                end
            end

            if strcmp(model_type,'KT')
                [coeffs,interval] = fit_KTmodel(obj,init_Guess,plot_);

                if show_info_ 
                    disp('----------------Fitting result---------------');
                    disp(['S_init  = ',num2str(obj.PGA),', PGA, exract from recorded GM directly']);
                    disp(['Omega_g = ',num2str(coeffs(1)),', 95% confidence intervals: ',...
                        '(',num2str(interval(1,1)),', ',num2str(interval(2,1)),').']);
                    disp(['Beta_g  = ',num2str(coeffs(2)),', 95% confidence intervals: ',...
                        '(',num2str(interval(1,2)),', ',num2str(interval(2,2)),').']);
                end
            end

            if strcmp(model_type,'Hu_S0') 
                [coeffs,Flag] = fit_HuKTmodel_S0(obj,init_Guess,plot_);
                if show_info_ 
                    disp('----------------Fitting result---------------');
                    disp(['PGA     = ',num2str(obj.PGA),', PGA, exract from recorded GM directly']);
                    disp(['Omega_g = ',num2str(coeffs(1))]);
                    disp(['Beta_g  = ',num2str(coeffs(2))]);
                    disp(['Omega_c = ',num2str(coeffs(3)*coeffs(1))]);
                    disp(['S_init  = ',num2str(obj.PGA)]);
                end
            end

            if strcmp(model_type,'CP_S0')
                [coeffs,interval] = fit_CPmodel_S0(obj,init_Guess,plot_);

                if show_info_ 
                    disp('----------------Fitting result---------------');
                    disp(['PGA     = ',num2str(obj.PGA),', PGA, exract from recorded GM directly']);
                    disp(['Omega_g = ',num2str(coeffs(1)),', 95% confidence intervals: ',...
                        '(',num2str(interval(1,1)),', ',num2str(interval(2,1)),').']);
                    disp(['Beta_g  = ',num2str(coeffs(2)),', 95% confidence intervals: ',...
                        '(',num2str(interval(1,2)),', ',num2str(interval(2,2)),').']);
                    disp(['Omega_f = ',num2str(coeffs(3)),', 95% confidence intervals: ',...
                        '(',num2str(interval(1,3)),', ',num2str(interval(2,3)),').']);
                    disp(['Beta_f  = ',num2str(coeffs(4)),', 95% confidence intervals: ',...
                        '(',num2str(interval(1,4)),', ',num2str(interval(2,4)),').']);
                    disp(['S_init  = ',num2str(coeffs(5)),', 95% confidence intervals: ',...
                        '(',num2str(interval(1,5)),', ',num2str(interval(2,5)),').']);                   
                end
            end

            if strcmp(model_type,'KT_S0')
                [coeffs,interval] = fit_KTmodel_S0(obj,init_Guess,plot_);

                if show_info_ 
                    disp('----------------Fitting result---------------');
                    disp(['PGA     = ',num2str(obj.PGA),', PGA, exract from recorded GM directly']);
                    disp(['Omega_g = ',num2str(coeffs(1)),', 95% confidence intervals: ',...
                        '(',num2str(interval(1,1)),', ',num2str(interval(2,1)),').']);
                    disp(['Beta_g  = ',num2str(coeffs(2)),', 95% confidence intervals: ',...
                        '(',num2str(interval(1,2)),', ',num2str(interval(2,2)),').']);
                    disp(['S_init  = ',num2str(coeffs(3)),', 95% confidence intervals: ',...
                        '(',num2str(interval(1,3)),', ',num2str(interval(2,3)),').']);
                end
            end
        end

        function [CP_coeffs,CP_coeffs_interval] = fit_CPmodel(obj,init_Guess,plot_)
            rad_Freq = obj.Rad_Freq;
            S_init = obj.PGA;
            curve_fittype = fittype('(power(rad_Freq,4))*(power(Omega_g,4)+ power(2*Omega_g*Beta_g*rad_Freq,2))./((power(power(Omega_f,2)-power(rad_Freq,2),2)+power(2*Omega_f*Beta_f*rad_Freq,2))*(power(power(Omega_g,2)-power(rad_Freq,2),2)+power(2*Omega_g*Beta_g*rad_Freq,2)))',...
                        'independent',{'rad_Freq'},...
                        'coefficients',{'Omega_g','Beta_g','Omega_f','Beta_f'});
            options = fitoptions('Method', 'NonlinearLeastSquares', 'StartPoint', init_Guess,'Lower', [0,0,0,0],'MaxFunEvals',1200,'MaxIter',1200);
            fit_curve = fit(rad_Freq,obj.PowerSpectrum/S_init,curve_fittype,options);
            CP_coeffs = coeffvalues(fit_curve);
            CP_coeffs_interval = confint(fit_curve);
            norm_fac = 1/S_init;

            if nargin < 3
                plot_ = false;  
            end
            
            if plot_
                figure
                plot(rad_Freq,obj.PowerSpectrum*norm_fac);
                hold on
                plot(rad_Freq,feval(fit_curve, rad_Freq))
                title('Fitting result');
                xlabel('radius Freq');
                ylabel('S(w)/S0');
                legend("Scaling PowerSpectrum","fitted C-P filter")
            end
            
        end

        function [HuKT_coeffs,HuKT_coeffs_interval] = fit_HuKTmodel(obj,init_Guess,plot_)
            rad_Freq = obj.Rad_Freq;
            S_init = obj.PGA;
            curve_fittype = fittype('(power(rad_Freq,6))*(power(Omega_g,4) + power(2*Omega_g*Beta_g*rad_Freq,2))./((power(rad_Freq,6)+power(Omega_c,6))*(power(power(Omega_g,2)-power(rad_Freq,2),2)+power(2*Omega_g*Beta_g*rad_Freq,2)))',...
                        'independent',{'rad_Freq'},...
                        'coefficients',{'Omega_g','Beta_g','Omega_c'});
            options = fitoptions('Method', 'NonlinearLeastSquares', 'StartPoint', init_Guess,'Lower', [0,0,0]);
            fit_curve = fit(rad_Freq,obj.PowerSpectrum/S_init,curve_fittype,options);
            HuKT_coeffs = coeffvalues(fit_curve);
            HuKT_coeffs_interval = confint(fit_curve);
            norm_fac = 1/S_init;

            if nargin < 3
                plot_ = false;  
            end
            
            if plot_
                figure
                plot(rad_Freq,obj.PowerSpectrum*norm_fac);
                hold on
                plot(rad_Freq,feval(fit_curve, rad_Freq))
                title('Fitting result');
                xlabel('radius Freq');
                ylabel('S(w)/S0');
                legend("Scaling PowerSpectrum","fitted Hu-KT filter")
            end
        end

        function [KT_coeffs,KT_coeffs_interval] = fit_KTmodel(obj,init_Guess,plot_)
            rad_Freq = obj.Rad_Freq;
            S_init = obj.PGA;
            curve_fittype = fittype('(power(Omega_g,4) + power(2*Omega_g*Beta_g*rad_Freq,2))./((power(power(Omega_g,2)-power(rad_Freq,2),2)+power(2*Omega_g*Beta_g*rad_Freq,2)))',...
                        'independent',{'rad_Freq'},...
                        'coefficients',{'Omega_g','Beta_g'});
            options = fitoptions('Method', 'NonlinearLeastSquares', 'StartPoint',init_Guess,'Lower', [0,0]);
            fit_curve = fit(rad_Freq,obj.PowerSpectrum/S_init,curve_fittype,options);
            KT_coeffs = coeffvalues(fit_curve);
            KT_coeffs_interval = confint(fit_curve);
            norm_fac = 1/S_init;

            if nargin < 3
                plot_ = false;  
            end
            
            if plot_
                figure
                plot(rad_Freq,obj.PowerSpectrum*norm_fac);
                hold on
                plot(rad_Freq,feval(fit_curve, rad_Freq))
                title('Fitting result');
                xlabel('radius Freq');
                ylabel('S(w)/S0');
                legend("Scaling PowerSpectrum","fitted K-T filter")
            end
            
        end

        function [CP_coeffs,CP_coeffs_interval] = fit_CPmodel_S0(obj,init_Guess,plot_)
            rad_Freq = obj.Rad_Freq;
            %S_init = obj.PGA;
            curve_fittype = fittype('S_init*(power(rad_Freq,4))*(power(Omega_g,4)+ power(2*Omega_g*Beta_g*rad_Freq,2))./((power(power(Omega_f,2)-power(rad_Freq,2),2)+power(2*Omega_f*Beta_f*rad_Freq,2))*(power(power(Omega_g,2)-power(rad_Freq,2),2)+power(2*Omega_g*Beta_g*rad_Freq,2)))',...
                        'independent',{'rad_Freq'},...
                        'coefficients',{'Omega_g','Beta_g','Omega_f','Beta_f','S_init'});
            options = fitoptions('Method', 'NonlinearLeastSquares', 'StartPoint', init_Guess,'Lower', [0,0,0,0,0]);
            fit_curve = fit(rad_Freq,obj.PowerSpectrum,curve_fittype,options);
            CP_coeffs = coeffvalues(fit_curve);
            CP_coeffs_interval = confint(fit_curve);
            norm_fac = 1/CP_coeffs(end);

            if nargin < 3
                plot_ = false;  
            end
            
            if plot_
                figure
                plot(rad_Freq,obj.PowerSpectrum*norm_fac);
                hold on
                plot(rad_Freq,feval(fit_curve, rad_Freq)*norm_fac)
                title('Fitting result');
                xlabel('radius Freq');
                ylabel('S(w)/S0');
                legend("Scaling PowerSpectrum","fitted C-P filter")
            end
            
        end

        function CP_coeffs = fit_CPmodel_S0_trial(obj,init_Guess)
            rad_Freq = obj.Rad_Freq;
            %S_init = obj.PGA;


            fun = @(params) params(5)*(power(rad_Freq,4)).*(power(params(1),4)+ power(2*params(1)*params(2).*rad_Freq,2))./((power(power(params(3),2)-power(rad_Freq,2),2)+power(2*params(3)*params(4).*rad_Freq,2)).*(power(power(params(1),2)-power(rad_Freq,2),2)+power(2*params(1)*params(2).*rad_Freq,2)))-obj.PowerSpectrum;
            lb = [0,0,0,0,0];
            options = optimoptions('lsqnonlin','Display','iter');
            %x0 = [100,0.01,50,0.01,obj.PGA];
            x0 = init_Guess;
            [CP_coeffs,~,~,~,~] = lsqnonlin(fun,x0,lb,[],options);
            
        end

        function [HuKT_coeffs,HuKT_coeffs_interval] = fit_HuKTmodel_S0_OutOfDate(obj,init_Guess,plot_)
            rad_Freq = obj.Rad_Freq;
            %S_init = obj.PGA;
            curve_fittype = fittype('S_init*(power(rad_Freq,6))*(power(Omega_g,4) + power(2*Omega_g*Beta_g*rad_Freq,2))./((power(rad_Freq,6)+power(Omega_c,6))*(power(power(Omega_g,2)-power(rad_Freq,2),2)+power(2*Omega_g*Beta_g*rad_Freq,2)))',...
                        'independent',{'rad_Freq'},...
                        'coefficients',{'Omega_g','Beta_g','Omega_c','S_init'});
            options = fitoptions('Method', 'NonlinearLeastSquares', 'StartPoint', init_Guess,'Lower', [0,0,0,0]);
            fit_curve = fit(rad_Freq,obj.PowerSpectrum,curve_fittype,options);
            HuKT_coeffs = coeffvalues(fit_curve);
            HuKT_coeffs_interval = confint(fit_curve);
            norm_fac = 1/HuKT_coeffs(end);

            if nargin < 3
                plot_ = false;  
            end
            
            if plot_
                figure
                plot(rad_Freq,obj.PowerSpectrum*norm_fac);
                hold on
                plot(rad_Freq,feval(fit_curve, rad_Freq)*norm_fac)
                title('Fitting result');
                xlabel('radius Freq');
                ylabel('S(w)/S0');
                legend("Scaling PowerSpectrum","fitted Hu-KT filter")
            end
        end

        function [HuKT_coeffs,Flag] = fit_HuKTmodel_S0_(obj,init_Guess,plot_)
            %Use without parameter: S0
            
            rad_Freq = obj.Rad_Freq;
            S_init = obj.PGA;
            fitting_fun = @(para) S_init.*(power(rad_Freq,6)).*(power(para(1),4) + power(2*para(1)*para(2).*rad_Freq,2))./((power(rad_Freq,6)+power(para(3)*para(1),6)).*(power(power(para(1),2)-power(rad_Freq,2),2)+power(2*para(1)*para(2).*rad_Freq,2)))- obj.PowerSpectrum;
            lb = [1,0,0];
            ub = [inf,inf,inf];

            %options = optimoptions('lsqnonlin','Display','iter','FunctionTolerance',1e-16,'StepTolerance',1e-16);
            options = optimoptions('lsqnonlin','FunctionTolerance',1e-16,'StepTolerance',1e-16);
            x0 = init_Guess;
            [HuKT_coeffs,~,~,Flag,~] = lsqnonlin(fitting_fun,x0,lb,ub,options);

            plot_fun =  @(para) S_init.*(power(rad_Freq,6)).*(power(para(1),4) + power(2*para(1)*para(2).*rad_Freq,2))./((power(rad_Freq,6)+power(para(3)*para(1),6)).*(power(power(para(1),2)-power(rad_Freq,2),2)+power(2*para(1)*para(2).*rad_Freq,2)));
            if nargin < 3
                plot_ = false;  
            end
            
            if plot_
                figure
                plot(rad_Freq,obj.PowerSpectrum);
                hold on
                plot(rad_Freq,plot_fun(HuKT_coeffs))
                title('Fitting result');
                xlabel('radius Freq');
                ylabel('S(w)/S0');
                legend("Scaling PowerSpectrum","fitted Hu-KT filter")
            end
        end

        function [HuKT_coeffs,Flag] = fit_HuKTmodel_S0(obj,init_Guess,plot_)
            %Use with parameter: S0

            rad_Freq = obj.Rad_Freq;
            %S_init = obj.PGA;
            fitting_fun = @(para) obj.PowerSpectrum - para(4).*(power(rad_Freq,6)).*(power(para(1),4) + power(2*para(1)*para(2).*rad_Freq,2))./((power(rad_Freq,6)+power(para(3)*para(1),6)).*(power(power(para(1),2)-power(rad_Freq,2),2)+power(2*para(1)*para(2).*rad_Freq,2)));
            lb = [1,0,0,0];
            ub = [300,2.0,1,1];

           
            %options = optimoptions('lsqnonlin','Display','iter','FunctionTolerance',1e-16,'StepTolerance',1e-16);
            options = optimoptions('lsqnonlin','FunctionTolerance',1e-16,'StepTolerance',1e-16);
            x0 = init_Guess;
            [HuKT_coeffs,~,~,Flag,~] = lsqnonlin(fitting_fun,x0,lb,ub,options);

            HuKT_coeffs = HuKT_coeffs(1:3);
            plot_fun =  @(para) para(4).*(power(rad_Freq,6)).*(power(para(1),4) + power(2*para(1)*para(2).*rad_Freq,2))./((power(rad_Freq,6)+power(para(3)*para(1),6)).*(power(power(para(1),2)-power(rad_Freq,2),2)+power(2*para(1)*para(2).*rad_Freq,2)));
            
            
            if nargin < 4
                plot_ = false;  
            end
            
            if plot_
                figure
                plot(rad_Freq,obj.PowerSpectrum);
                hold on
                plot(rad_Freq,plot_fun(HuKT_coeffs))
                title('Fitting result');
                xlabel('radius Freq');
                ylabel('S(w)/S0');
                legend("Scaling PowerSpectrum","fitted Hu-KT filter")
            end
        end

        function [KT_coeffs,KT_coeffs_interval] = fit_KTmodel_S0(obj,init_Guess,plot_)
            rad_Freq = obj.Rad_Freq;
            %S_init = obj.PGA;
            curve_fittype = fittype('S_init*(power(Omega_g,4) + power(2*Omega_g*Beta_g*rad_Freq,2))./((power(power(Omega_g,2)-power(rad_Freq,2),2)+power(2*Omega_g*Beta_g*rad_Freq,2)))',...
                        'independent',{'rad_Freq'},...
                        'coefficients',{'Omega_g','Beta_g','S_init'});
            options = fitoptions('Method', 'NonlinearLeastSquares', 'StartPoint',init_Guess,'Lower', [0,0,0]);
            fit_curve = fit(rad_Freq,obj.PowerSpectrum,curve_fittype,options);
            KT_coeffs = coeffvalues(fit_curve);
            KT_coeffs_interval = confint(fit_curve);
            norm_fac = 1/(KT_coeffs(end));

            if nargin < 3
                plot_ = false;  
            end
            
            if plot_
                figure
                plot(rad_Freq,obj.PowerSpectrum*norm_fac);
                hold on
                plot(rad_Freq,feval(fit_curve, rad_Freq)*norm_fac);
                title('Fitting result');
                xlabel('radius Freq');
                ylabel('S(w)/S0');
                legend("Scaling PowerSpectrum","fitted K-T filter")
            end
            
        end

        function time_percentiles = getPercentileInfo_by_fitting(obj,plot_)  
            % Calculate Arias Intensity
            cdf = pi*cumtrapz(power(obj.Ampl_t,2))/(2*9.81);

            %target_percentiles = [1,5,45,95];

            % Interpolate time values corresponding to target percentiles
            %time_percentiles = interp1(cdf / max(cdf), obj.Time, target_percentiles / 100);

            valid_time = obj.Time;%(obj.Time>=time_percentiles(1));
            %valid_time = valid_time(obj.Time<=time_percentiles(4)+5);

            valid_ampl_t = obj.Ampl_t;%(obj.Time>=time_percentiles(1));
            %valid_ampl_t = valid_ampl_%(obj.Time<=time_percentiles(4)+5);

            cumulative_AI_valid = pi*cumtrapz(power(valid_ampl_t ,2))/(2*9.81);
            AI = pi*trapz(power(valid_ampl_t ,2))/(2*9.81);

            curve_fittype = fittype('gamcdf(valid_time,alpha, beta)',...
            'independent',{'valid_time'},...
            'coefficients',{'alpha','beta'});

            options = fitoptions('Method', 'NonlinearLeastSquares', 'StartPoint', [6,2],'Lower', [0,0],'MaxFunEvals',10000,'MaxIter',10000);
            fit_para = fit(valid_time,cumulative_AI_valid/AI,curve_fittype,options);
            Gam_coeffs = coeffvalues(fit_para);

            %fit_gam_curve = gamcdf(obj.Time,Gam_coeffs(1),Gam_coeffs(2));
            per1 = gaminv(0.01,Gam_coeffs(1),Gam_coeffs(2));
            per5 = gaminv(0.05,Gam_coeffs(1),Gam_coeffs(2));
            per45 = gaminv(0.45,Gam_coeffs(1),Gam_coeffs(2));
            per95 = gaminv(0.95,Gam_coeffs(1),Gam_coeffs(2));
            time_percentiles = [per1,per5,per45,per95];

            if nargin < 2
                plot_ = false;  
            end
            
            if plot_
                figure
                plot(obj.Time,...
                    cdf/obj.AriasIntensity);
                hold on 
                plot(obj.Time,...
                    gamcdf(obj.Time,Gam_coeffs(1),Gam_coeffs(2)));
                xline(time_percentiles(1),'r');
                xline(time_percentiles(2),'g');
                xline(time_percentiles(3),'c');
                xline(time_percentiles(4),'m');

                xline(per5,'--');
                xline(per45,'--');
                xline(per95,'--');
                title('Cumulative Arias Intensity');
                xlabel('Time (s)');
                ylabel('Arias Intensity');
                %legend("Cumlative AI","Fitting Gamma CDF","1 percentiles",...
                %    "5 percentiles","45 percentiles","95 percentiles");
            end
        end

        function time_percentiles = getPercentileInfo(obj,plot_)  
            % Calculate Arias Intensity
            cdf = pi*cumtrapz(power(obj.Ampl_t,2))/(2*9.81);

            % Find indices of duplicate x-values
            [unique_x, idx_unique] = unique(cdf, 'stable');
            %duplicate_indices = setdiff(1:length(cdf), idx_unique);
            
            % Remove duplicates from x and corresponding y values
            cdf_unique = unique_x;
            time = obj.Time(idx_unique);

            target_percentiles = [1,5,45,95];

            % Interpolate time values corresponding to target percentiles
            time_percentiles = interp1(cdf_unique / max(cdf_unique), time, target_percentiles / 100);

            if nargin < 2
                plot_ = false;  
            end
            
            if plot_
                figure
                plot(obj.Time,...
                    cdf/obj.AriasIntensity);
                xline(time_percentiles(1),'r');
                xline(time_percentiles(2),'g');
                xline(time_percentiles(3),'c');
                xline(time_percentiles(4),'m');
                title('Cumulative Arias Intensity');
                xlabel('Time (s)');
                ylabel('Arias Intensity');
                legend("Cumlative AI","1 percentiles",...
                    "5 percentiles","45 percentiles","99 percentiles");
            end
        end

        function q = generateTimeModFunc(obj,time_percentiles,gamma_params,AriasIntensity,energy_factor,plot_)
            if isempty(AriasIntensity)
                AriasIntensity = obj.AriasIntensity;
            end
            
            if isempty(energy_factor)
                energy_factor = 1;  
            end


            if isempty(gamma_params)
                t_mid = (time_percentiles(4)+time_percentiles(2))/2 - time_percentiles(2);
                duration = time_percentiles(4)-time_percentiles(2);
                % Define the objective function
                objectiveFunction = @(params) (power(gamcdf(t_mid, params(1), params(1)/params(2)) - 0.50,2)+ ...
                                               power(gamcdf(duration, params(1), params(1)/params(2)) - 0.99,2));
                

                % Initial guess for alpha and beta
                initial_guess = [3, 1]; % This needs to be adjusted based on your data
               
                %options = optimset('MaxFunEvals',1000000,'MaxIter',100000,'TolFun',1e-16,'Display', 'iter');
                options = optimset('MaxFunEvals',1000000,'MaxIter',100000,'TolFun',1e-16);
                % Find parameters that minimize the objective function
                estimated_params = fminsearch(objectiveFunction, initial_guess,options);
                estimated_params(2) = estimated_params(1)/estimated_params(2);
            else
                estimated_params = gamma_params;
            end
            
            % Estimated parameters
            alpha = estimated_params(1);
            beta = estimated_params(2);    
            mean = alpha*beta;

            % Shift mean point to 45 percentile
            alpha = (time_percentiles(3)-time_percentiles(2))/beta;
            beta = mean/alpha;

            %para_2 = (2*alpha)-1;
            %para_3 = 2*beta;
            %para_1 = sqrt(energy_factor.*AriasIntensity*...
            %    power(para_3,para_2)/gamma(para_2));
            %
            time = obj.Time(obj.Time<= obj.Time(end)-time_percentiles(1));
            
            %q = para_1.*power(time ,alpha-1).*...
            %    exp(-beta*time);
                      
            
            total_area = trapz(gampdf(time,alpha,beta));
            q = sqrt(((2*9.81)/pi)*AriasIntensity*gampdf(time,alpha,beta)/total_area);
            

            q = [zeros(length(obj.Time)-length(time),1);q];
            %gam = [zeros(length(obj.Time)-length(time),1);gam];
            if nargin < 5
                plot_ = false;  
            end
            
            fitted_cdfgam = cumtrapz(pi*power(q,2)/(2*9.81));
            
            %norm_fitted_cdfgam = fitted_cdfgam/max(abs(fitted_cdfgam));
            %scale_fitted_cdfgam = norm_fitted_cdfgam*AriasIntensity;
            scale_fitted_cdfgam = fitted_cdfgam;
            plot_time = obj.Time(obj.Time<=time_percentiles(4)+6);

            if plot_
                figure;
                plot(plot_time,pi*cumtrapz(...
                    power(obj.Ampl_t(obj.Time<=time_percentiles(4)+6),2))/(2*9.81));
                hold on 
                plot(plot_time,scale_fitted_cdfgam(obj.Time<=time_percentiles(4)+6));
                title('Cumulative Arias Intensity Fitting');
                xlabel('Time (s)');
                ylabel('Arias Intensity');
                legend("Recorded Arias Intensity",...
                    " Fitted Model");

                figure;
                plot(obj.Time,obj.Ampl_t,'Color',[0.2, 0.4, 0.8, 0.2]);
                hold on         
                plot(obj.Time,q);
                %plot(obj.Time,gam);
                title('Time modulating function');
                xlabel('Time (s)');
                ylabel('Amplitude');
                legend("Time modulating function","Signal");
            end

        end

        function plotTimeData(obj)
            figure
            plot(obj.Time,obj.Ampl_t);
            title('Signal in time domain');
            xlabel('Time');
            ylabel('Amplitude');
            grid on;
        end

        function plotFreqData(obj)
            figure
            plot(obj.Freq,obj.Real_f,obj.Freq,obj.Imag_f);
            title('Data in freq domain');
            xlabel('Freq');
            ylabel('Real/Imag');
            legend("Real","Imag");
            grid on;
    
            figure
            plot(obj.Freq,obj.Ampl_f);
            title('Data in freq domain');
            xlabel('Freq');
            ylabel('Amplitude');
            grid on;
        end

        function plotPS(obj)
            figure
            plot(obj.Freq,obj.PowerSpectrum);
            title('Data in freq domain');
            xlabel('Freq');
            ylabel('Sxx');
            grid on;
        end


    end
end