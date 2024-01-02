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

            % Design a low-pass filter with a cutoff frequency
            cutoff_frequency = obj.CutOffFreq / (0.5 * obj.Fs)-0.01; % Normalize the cutoff frequency
            filter_order = 50;
            low_pass_filter = fir1(filter_order, cutoff_frequency);
            
            % Apply the filter to the white noise
            bandlimited_noise = filter(low_pass_filter, 1, noise);
            
            % Scale and offset the noise to have a maximum amplitude in [-1, 1]
            noise = bandlimited_noise/max(abs(bandlimited_noise));

            if plot_
                figure;
                plot(obj.Time, noise);
                title('Band-Limited White Gaussian Noise');
                xlabel('Time (seconds)');
                ylabel('Amplitude');
            end
        end

        function output_GM = generateStaPesudoGM(obj,GM_model,init_Guess,S_init,domain_type,show_info_, plot_)
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
            coeffs = fit_GM_model(obj,GM_model,init_Guess,show_info_,false);
            if isempty(S_init)
                filter = GMmodel(obj,GM_model,coeffs,[]);
            else
                filter = GMmodel(obj,GM_model,coeffs,S_init);
            end
            %Generate FRF
            FRF = sqrt(filter/obj.PGA);
            norm_FRF = FRF;

            %Generate white noise
            noise = obj.generateWhiteNoise;
            noise_FFT = fft(noise);
            P1 = noise_FFT(1:floor(length(noise)/2+1));
            freq = obj.Fs*(0:(length(noise)/2))/length(noise);
            
            try
                if length(FRF)~=length(P1)
                    error('Lengths of FRF and Noise_FFT are not equal.');
                end
    
                if length(P1)~=length(freq)
                    error('Lengths of Freq and Noise_FFT are not equal.');
                end

                %Generate ground motion by appling FRF on white noise.
                PesudoGM_freq = transpose(P1).*norm_FRF;

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
                    ampl = data_IFFT; 
                    output_GM = table(time,ampl);

                    if plot_
                        figure;
                        plot(freq,norm_FRF);
                        title('Fitting result');
                        xlabel('radius Freq');
                        ylabel('S(w)/S0');
                        legend("fitted filter");
                        figure;
                        plot(obj.Time,data_IFFT);
                        title('Pesudo Ground Motion in time-domain');
                        xlabel('time (s)');
                        ylabel('Amplitude');                        
                    end
                end



            catch exception
                disp(['Error: ' exception.message]);
            end
            
        end

        function output_GM = generateTimeNonStaPesudoGM(obj,GM_model,init_Guess,S_init,AriasIntensity,energy_corrected_bool,show_info_, plot_)
            if isempty(AriasIntensity)
                AriasIntensity = obj.AriasIntensity;
            end

            if nargin < 6
                energy_corrected_bool = true;
            end
            
            if nargin < 7
                show_info_ = false;
            end

            if nargin < 8
                plot_ = false;
            end
            
            if isempty(S_init)
                stationaryGM = generateStaPesudoGM(obj,GM_model,init_Guess,[],"time",show_info_, false);
            else
                stationaryGM = generateStaPesudoGM(obj,GM_model,init_Guess,S_init,"time",show_info_, false);
            end
            %Get Percentile from recorded ground motion.
            time_percentiles = getPercentileInfo(obj,false);
            %Generate Time Modulating Function.
            q = generateTimeModFunc(obj,time_percentiles,AriasIntensity,1,false);
            
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
                q = generateTimeModFunc(obj,time_percentiles,AriasIntensity,energy_factor,false);
                ampl = q.*stationaryGM.ampl;           
                output_GM = table(time,ampl,K);
            else
                ampl = ampl_;
                energy_factor = 1;
                K = energy_factor*ones(length(ampl),1);
                output_GM = table(time,ampl,K);
            end

            if show_info_
                disp('----------Time Modulating Function-----------');
                disp(['AriasIntensity = ',num2str(AriasIntensity)]);
                disp(['EnergyFactor,K = ',num2str(energy_factor)]);
                disp(['Time   0.01%,P = ',num2str(time_percentiles(1)),' sec']);
                disp(['Time      5%,P = ',num2str(time_percentiles(2)),' sec']);
                disp(['Time     45%,P = ',num2str(time_percentiles(3)),' sec']);
                disp(['Time     95%,P = ',num2str(time_percentiles(4)),' sec']);
            end

            if plot_
                figure;
                plot(obj.Time,obj.Ampl_t,'Color',[0.2, 0.4, 0.8, 0.3]);
                hold on 
                plot(obj.Time,ampl);
                title('Non-Stationary Pesudo Ground Motion in Time-domain');
                xlabel('Time (s)');
                ylabel('Amplitude');
                legend("Recored Ground Motion","Non-Stationary Ground Motion");

                plot_time = obj.Time(obj.Time<=time_percentiles(4)+6);

                figure;
                plot(plot_time,pi*cumtrapz(...
                    power(obj.Ampl_t(obj.Time<=time_percentiles(4)+6),2))/(2*9.81));
                hold on 
                plot(plot_time,pi*cumtrapz(...
                    power(ampl(obj.Time<=time_percentiles(4)+6),2))/(2*9.81));
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

            model_type_list = {'Hu','KT','CP'};
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

        function coeffs = fit_GM_model(obj,model_type,init_Guess,show_info_,plot_)
            model_type_list = {'Hu','KT','CP'};
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
        end
        
        function [CP_coeffs,CP_coeffs_interval] = fit_CPmodel(obj,init_Guess,plot_)
            rad_Freq = 2*pi*obj.Freq;
            S_init = obj.PGA;
            curve_fittype = fittype('(power(rad_Freq,4))*(power(Omega_g,4)+ power(2*Omega_g*Beta_g*rad_Freq,2))./((power(power(Omega_f,2)-power(rad_Freq,2),2)+power(2*Omega_f*Beta_f*rad_Freq,2))*(power(power(Omega_g,2)-power(rad_Freq,2),2)+power(2*Omega_g*Beta_g*rad_Freq,2)))',...
                        'independent',{'rad_Freq'},...
                        'coefficients',{'Omega_g','Beta_g','Omega_f','Beta_f'});
            options = fitoptions('Method', 'NonlinearLeastSquares', 'StartPoint', init_Guess);
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
            rad_Freq = 2*pi*obj.Freq;
            S_init = obj.PGA;
            curve_fittype = fittype('(power(rad_Freq,6))*(power(Omega_g,4) + power(2*Omega_g*Beta_g*rad_Freq,2))./((power(rad_Freq,6)+power(Omega_c,6))*(power(power(Omega_g,2)-power(rad_Freq,2),2)+power(2*Omega_g*Beta_g*rad_Freq,2)))',...
                        'independent',{'rad_Freq'},...
                        'coefficients',{'Omega_g','Beta_g','Omega_c'});
            options = fitoptions('Method', 'NonlinearLeastSquares', 'StartPoint', init_Guess);
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
            rad_Freq = 2*pi*obj.Freq;
            S_init = obj.PGA;
            curve_fittype = fittype('(power(Omega_g,4) + power(2*Omega_g*Beta_g*rad_Freq,2))./((power(power(Omega_g,2)-power(rad_Freq,2),2)+power(2*Omega_g*Beta_g*rad_Freq,2)))',...
                        'independent',{'rad_Freq'},...
                        'coefficients',{'Omega_g','Beta_g'});
            options = fitoptions('Method', 'NonlinearLeastSquares', 'StartPoint',init_Guess);
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

        function time_percentiles = getPercentileInfo(obj,plot_)  
            % Calculate Arias Intensity
            cdf = pi*cumtrapz(power(obj.Ampl_t,2))/(2*9.81);

            target_percentiles = [0.01,5,45,95];
            
            % Interpolate time values corresponding to target percentiles
            time_percentiles = interp1(cdf / max(cdf), obj.Time, target_percentiles / 100);

            if nargin < 2
                plot_ = false;  
            end
            
            if plot_
                figure
                plot(obj.Time(obj.Time<=time_percentiles(4)+5),...
                    cdf(obj.Time<=time_percentiles(4)+5));
                hold on 
                xline(time_percentiles(1),'r');
                xline(time_percentiles(2),'g');
                xline(time_percentiles(3),'c');
                xline(time_percentiles(4),'m');
                title('Cumulative Arias Intensity');
                xlabel('Time (s)');
                ylabel('Arias Intensity');
                legend("Cumlative AI","0.01 percentiles",...
                    "5 percentiles","45 percentiles","95 percentiles");
            end
        end

        function q = generateTimeModFunc(obj,time_percentiles,AriasIntensity,energy_factor,plot_)
            if isempty(AriasIntensity)
                AriasIntensity = obj.AriasIntensity;
            end
            
            if isempty(energy_factor)
                energy_factor = 1;  
            end

            
            % Given information
            t_45_percentile = time_percentiles(3)-time_percentiles(1);  % Time at which 45th percentile occurs
            total_duration = time_percentiles(4)-(time_percentiles(2)-time_percentiles(1));  % Total duration
            % Objective function for optimization
            objective_function = @(params) abs(gaminv(0.45, params(1), params(2))/gaminv(1, params(1), params(2)) - t_45_percentile/total_duration) + abs(params(1) * params(2) - total_duration);
            
            % Initial guess for parameters
            initial_guess = [2, 1];
            
            options = optimset('MaxFunEvals',1000);
            % Find parameters that minimize the objective function
            estimated_params = fminsearch(objective_function, initial_guess,options);
            
            % Estimated parameters
            alpha = estimated_params(1);
            beta = estimated_params(2);

            para_2 = 2*alpha-1;
            para_3 = 2*beta;
            para_1 = sqrt(energy_factor.*AriasIntensity*...
                power(para_3,para_2)/gamma(para_2));
            
            time = obj.Time(obj.Time<= obj.Time(end)-time_percentiles(1));
            q = para_1.*power(time ,alpha-1).*...
                exp(-beta*time);
            q = [zeros(length(obj.Time)-length(time),1);q];
            
            if nargin < 5
                plot_ = false;  
            end
            
            fitted_cdfgam = cumtrapz(q);
            norm_fitted_cdfgam = fitted_cdfgam/max(abs(fitted_cdfgam));
            scale_fitted_cdfgam = norm_fitted_cdfgam*AriasIntensity;
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
                    " Fitted Model (Scaled)");

                figure;
                plot(obj.Time,q);
                hold on 
                plot(obj.Time,obj.Ampl_t);
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