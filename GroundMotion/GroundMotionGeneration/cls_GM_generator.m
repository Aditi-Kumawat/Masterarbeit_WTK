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

        function [output_GM,coeffs] = generateNormStaPesudoGM(obj,GM_model,init_Guess,show_info_, plot_)

            if nargin < 4
                show_info_ = false;
            end

            if nargin < 5
                plot_ = false;
            end            
            
            %Fitting model
            [coeffs,Flag] = fit_GM_model(obj,GM_model,init_Guess,show_info_,plot_);
            
            if strcmp(GM_model,'Hu_S0')
                 if coeffs(3) >= 0.9
                    [coeffs,Flag] = fit_GM_model(obj,'Hu_S0_NoContraint',init_Guess,show_info_,plot_);
                 end
            end

            % if Flag ~= 1, not convegence
            if Flag ~= 1
               coeffs(1) = -1; 
            end

            filter = GMmodel(obj,GM_model,coeffs);

            %Generate white noise
            noise = obj.generateWhiteNoise;
            noise_FFT = fft(noise);
            P1 = noise_FFT(1:floor(length(noise)/2+1));
            freq = obj.Fs*(0:(length(noise)/2))/length(noise);

            %Generate Transfer function from FRF
            FRF = sqrt(filter/coeffs(4)); % here is FRF is acutually the TF
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
                PesudoGM_freq(1:0) = 0;

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

            catch exception
                disp(['Error: ' exception.message]);
            end
            
        end

        function [output_GM,frequency_coeffs,time_coeffs,GM_info] = generateTimeNonStaPesudoGMbyFit(obj,GM_model,init_Guess,AriasIntensity,show_info_, plot_)
            if isempty(AriasIntensity)
                AriasIntensity = obj.AriasIntensity;
            end
            
            if nargin < 5
                show_info_ = false;
            end

            if nargin < 6
                plot_ = false;
            end
            
            [stationaryGM,coeffs] = generateNormStaPesudoGM(obj,GM_model,init_Guess,show_info_, plot_);

            %Get Percentile from recorded ground motion.
            time_percentiles = getPercentileInfo(obj,false);
            %Generate Time Modulating Function.
            q = generateTimeModFunc(obj,time_percentiles,[],AriasIntensity,plot_);
 
            time = obj.Time;
            %Generate simulated ground motion
            ampl = q.*stationaryGM.ampl;
            output_GM = table(time,ampl);

            %Output coeffs
            if strcmp(GM_model,"Hu_S0")
                frequency_coeffs = coeffs(1:3);
            else
                frequency_coeffs = coeffs;
            end

            time_coeffs = time_percentiles;
            GM_info = [AriasIntensity,obj.Time(end),obj.Fs];

            if show_info_
                disp('----------Time Modulating Function-----------');
                disp(['AriasIntensity = ',num2str(AriasIntensity)]);
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
                    "Simulated Arias Intensity");

            end
        end

        function filter = GMmodel(obj,model_type,filter_para,plot_)
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

            model_type_list = {'Hu_S0'};
            if ~ismember(model_type, model_type_list)
                fprintf('ERROR! Wrong input type, please input one of the type: [%s]\n', strjoin(data_type_list, ', '));
            end
            
            if nargin < 4
                plot_ = false;  
            end

            if strcmp(model_type,'Hu_S0') 
                Omega_g=filter_para(1);
                Beta_g=filter_para(2);
                Omega_c=filter_para(3)*filter_para(1);
                S_init = filter_para(4);

                filter = HuKTmodel(obj,Omega_g,Beta_g,S_init,Omega_c);
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

        function [coeffs,Flag] = fit_GM_model(obj,model_type,init_Guess,show_info_,plot_)
            model_type_list = {'Hu_S0','Hu_S0_NoContraint'};
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

            if strcmp(model_type,'Hu_S0') 
                [coeffs,Flag] = fit_HuKTmodel_S0(obj,init_Guess,plot_);
                if show_info_ 
                    disp('----------------Fitting result---------------');
                    disp(['PGA     = ',num2str(obj.PGA),', PGA, exract from recorded GM directly']);
                    disp(['Omega_g = ',num2str(coeffs(1))]);
                    disp(['Beta_g  = ',num2str(coeffs(2))]);
                    disp(['Omega_c = ',num2str(coeffs(3)*coeffs(1))]);
                    disp(['S_init  = ',num2str(coeffs(4))]);
                end
            end

            if strcmp(model_type,'Hu_S0_NoContraint') 
                [coeffs,Flag] = fit_HuKTmodel_S0_NoConstraint(obj,init_Guess,plot_);
                disp('Fitting by Hu_S0_NoContraint');
                if show_info_ 
                    disp('----------------Fitting result---------------');
                    disp(['PGA     = ',num2str(obj.PGA),', PGA, exract from recorded GM directly']);
                    disp(['Omega_g = ',num2str(coeffs(1))]);
                    disp(['Beta_g  = ',num2str(coeffs(2))]);
                    disp(['Omega_c = ',num2str(coeffs(3)*coeffs(1))]);
                    disp(['S_init  = ',num2str(coeffs(4))]);
                end
            end

        end

        function [HuKT_coeffs,Flag] = fit_HuKTmodel_S0_NoConstraint(obj,init_Guess,plot_)
            %Use without parameter: S0
            
            rad_Freq = obj.Rad_Freq;
            fitting_fun = @(para) para(4).*(power(rad_Freq,6)).*(power(para(1),4) + power(2*para(1)*para(2).*rad_Freq,2))./((power(rad_Freq,6)+power(para(3)*para(1),6)).*(power(power(para(1),2)-power(rad_Freq,2),2)+power(2*para(1)*para(2).*rad_Freq,2)))- obj.PowerSpectrum;
            lb = [1,0,0,0];
            ub = [500,inf,20,1];

            %options = optimoptions('lsqnonlin','Display','iter','FunctionTolerance',1e-16,'StepTolerance',1e-16);
            options = optimoptions('lsqnonlin','FunctionTolerance',1e-16,'StepTolerance',1e-16);
            x0 = init_Guess;
            [HuKT_coeffs,~,~,Flag,~] = lsqnonlin(fitting_fun,x0,lb,ub,options);

            plot_fun =  @(para) para(4).*(power(rad_Freq,6)).*(power(para(1),4) + power(2*para(1)*para(2).*rad_Freq,2))./((power(rad_Freq,6)+power(para(3)*para(1),6)).*(power(power(para(1),2)-power(rad_Freq,2),2)+power(2*para(1)*para(2).*rad_Freq,2)));
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
            ub = [500,inf,1,1];

           
            %options = optimoptions('lsqnonlin','Display','iter','FunctionTolerance',1e-16,'StepTolerance',1e-16);
            options = optimoptions('lsqnonlin','FunctionTolerance',1e-16,'StepTolerance',1e-16);
            x0 = init_Guess;
            [HuKT_coeffs,~,~,Flag,~] = lsqnonlin(fitting_fun,x0,lb,ub,options);

            %HuKT_coeffs = HuKT_coeffs(1:3);
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

        function q = generateTimeModFunc(obj,time_percentiles,gamma_params,AriasIntensity,plot_)
            if isempty(AriasIntensity)
                AriasIntensity = obj.AriasIntensity;
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
                [estimated_params,~,~] = fminsearch(objectiveFunction, initial_guess,options);
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

            time = obj.Time(obj.Time<= obj.Time(end)-time_percentiles(1));
                      
            gamma_pdf = gampdf(time,alpha,beta);
            % Find indices of inf values
            infIndices = isinf(gamma_pdf);  
            % Replace inf values with 1
            gamma_pdf(infIndices) = 1;
            total_area = trapz(gamma_pdf);

            q = sqrt(((2*9.81)/pi)*AriasIntensity*gamma_pdf/total_area);
            q = [zeros(length(obj.Time)-length(time),1);q];

            if nargin < 5
                plot_ = false;  
            end
            
            if plot_

                fitted_cdfgam = cumtrapz(pi*power(q,2)/(2*9.81));
                scale_fitted_cdfgam = fitted_cdfgam;
                plot_time = obj.Time(obj.Time<=time_percentiles(4)+6);

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