clear;
clc;
close all;

X_path = ".\X_SBAGM_PredDOE_800_1000.mat";
load(X_path);

%[Time, Acc, ~, Disp] = fns_generateGM_Params([], [], 100 ,"Hu_S0", FRF_info(1:3), [0.5, 1 ,4, 10], GM_info(1));

tic
Yz = SDOF_simulation(X_valid_800,"Table_val", false);
toc


function Y = SDOF_simulation(X, X_type , plot_)
    
    %% Z dir
    % log(PGA) ~ Normal(-8.25,1.4)
    % Wg ~ LogNormal(2.5,0.3)
    % Beta = 0.3
    % W_c_ratio ~ LogNormal(-1.2,0.4)
    % Wn ~ Normal(12,1)
    
    if strcmp(X_type, "Random")
        PGA = exp(-8.25 + (1.4*X(:,1)));
        W_g = 2*pi*exp(2.5+ 0.3*X(:,2));
        beta = 0.3;
        W_c_ratio = exp(-1.2+ 0.4*X(:,3));
        
        Wn = 2*pi*(12+1*X(:,4));
        DR = 0.05;
    
        Y = zeros(length(X),1);
        for i = 1:length(X)
            disp(['Case ',num2str(i)])
            disp(['     Parameter: W_g = ',num2str(W_g(i)), ',W_c = ',num2str(W_c_ratio(i))])
            disp(['     Parameter: PGA = ',num2str(PGA(i)), ',W_n = ',num2str(Wn(i))])
    
            GM_params = [W_g(i), beta ,W_c_ratio(i)];
            [Time, ~, Vel, Disp] = fns_generateGM_Params([], [], 100 ,"Hu_S0", GM_params, [0.5, 1 ,4, 10], PGA(i));
            
            % FFT half-positive
            L = length(Vel);
            Fs = round(1/(Time(3)-Time(2)),200);
            data_FFT = fft(Vel);
            P2 = data_FFT;
            P1 = P2(1:floor(L/2+1));
            f = Fs*(0:(L/2))/L;
        
            % SDOF FRF, excited by Displacement
            Omega = 2*pi*f;
            FRF = (power(Omega,2))./(-power(Omega,2)+ 2*1i*Omega*DR*Wn(i)+ power(Wn(i),2));
        
            % Compute the Velocity
            Resp_f = P1.*transpose(FRF);
        
            % IFFT from half-positive
            Resp_f_pad = [Resp_f; conj(flipud(Resp_f(2:end-1,:)))];
            Resp_ifft = ifft(Resp_f_pad*Fs, L, 1, 'symmetric');
            Resp_ifft = Resp_ifft(1:L,:)/Fs;
        
            Y(i) = log(max(abs(Resp_ifft)));
    
        end
        file_name = sprintf('SDOF_AGM_Y_%d.mat', length(X));
        save(file_name,'Y','-mat');

    elseif strcmp(X_type, "Validation")
        
        valid_point = 200;
        X(valid_point,1)
        X(valid_point,2)
        X(valid_point,3)
        X(valid_point,4)
        PGA = exp(-8.25 + (1.4*X(:,1)));
        W_g = 2*pi*exp(2.5+ 0.3*X(:,2));
        beta = 0.3;
        W_c_ratio = exp(-1.2+ 0.4*X(:,3));
        
        Wn = 2*pi*(12+1*X(:,4));
        DR = 0.05;
    
        Y = zeros(length(X),1);
        for i = 1:length(X)
            disp(['Case ',num2str(i)])
            disp(['     Parameter: W_g = ',num2str(W_g(valid_point)), ',W_c = ',num2str(W_c_ratio(valid_point))])
            disp(['     Parameter: PGA = ',num2str(PGA(valid_point)), ',W_n = ',num2str(Wn(valid_point))])
    
            GM_params = [W_g(valid_point), beta ,W_c_ratio(valid_point)];
            [Time, ~, Vel, Disp] = fns_generateGM_Params([], [], 100 ,"Hu_S0", GM_params, [0.5, 1 ,4, 10], PGA(valid_point));
            
            % FFT half-positive
            L = length(Vel);
            Fs = round(1/(Time(3)-Time(2)),200);
            data_FFT = fft(Vel);
            P2 = data_FFT;
            P1 = P2(1:floor(L/2+1));
            f = Fs*(0:(L/2))/L;
        
            % SDOF FRF, excited by Displacement
            Omega = 2*pi*f;
            FRF = (power(Omega,2))./(-power(Omega,2)+ 2*1i*Omega*DR*Wn(valid_point)+ power(Wn(valid_point),2));
        
            % Compute the Velocity
            Resp_f = P1.*transpose(FRF);
        
            % IFFT from half-positive
            Resp_f_pad = [Resp_f; conj(flipud(Resp_f(2:end-1,:)))];
            Resp_ifft = ifft(Resp_f_pad*Fs, L, 1, 'symmetric');
            Resp_ifft = Resp_ifft(1:L,:)/Fs;
        
            Y(i) = max(abs(Resp_ifft));
    
        end
        file_name = sprintf('SDOF_AGM_Y_Valid_%d_Num_%d.mat',valid_point,length(X));
        save(file_name,'Y','-mat');

    elseif strcmp(X_type, "Table_rand")
        PGA = exp(X.lnPGA);
        W_g = 2*pi*X.Wg;
        beta = X.DRg;
        W_c_ratio = X.Wc;
        
        Wn = 2*pi*X.Wb;
        DR = 0.05;
    
        Y = zeros(length(X.lnPGA),1);
        for i = 1:length(X.lnPGA)
            disp(['Case ',num2str(i)])
            disp(['     Parameter: W_g = ',num2str(W_g(i)), ',W_c = ',num2str(W_c_ratio(i))])
            disp(['     Parameter: PGA = ',num2str(PGA(i)), ',W_n = ',num2str(Wn(i))])
    
            GM_params = [W_g(i), beta(i) ,W_c_ratio(i)];
            [Time, ~, Vel, Disp] = fns_generateGM_Params([], [], 100 ,"Hu_S0", GM_params, [0.5, 1 ,4, 10], PGA(i));
            
            % FFT half-positive
            L = length(Vel);
            Fs = round(1/(Time(3)-Time(2)),200);
            data_FFT = fft(Vel);
            P2 = data_FFT;
            P1 = P2(1:floor(L/2+1));
            f = Fs*(0:(L/2))/L;
        
            % SDOF FRF, excited by Displacement
            Omega = 2*pi*f;
            FRF = (power(Omega,2))./(-power(Omega,2)+ 2*1i*Omega*DR*Wn(i)+ power(Wn(i),2));
        
            % Compute the Velocity
            Resp_f = P1.*transpose(FRF);
        
            % IFFT from half-positive
            Resp_f_pad = [Resp_f; conj(flipud(Resp_f(2:end-1,:)))];
            Resp_ifft = ifft(Resp_f_pad*Fs, L, 1, 'symmetric');
            Resp_ifft = Resp_ifft(1:L,:)/Fs;
        
            Y(i) = log(max(abs(Resp_ifft)));
    
        end
        file_name = sprintf('SDOF_SBAGM_Y_Pred_%d.mat', length(X.lnPGA));
        save(file_name,'Y','-mat');


    elseif strcmp(X_type, "Table_val")
        valid_point = 800;
        PGA = exp(X.lnPGA);
        W_g = 2*pi*X.Wg;
        beta = X.DRg;
        W_c_ratio = X.Wc;
        
        Wn = 2*pi*X.Wb;
        DR = 0.05;
    
        Y = zeros(length(X.lnPGA),1);
        for i = 1:length(X.lnPGA)
            disp(['Case ',num2str(i)])
            disp(['     Parameter: W_g = ',num2str(W_g(i)), ',W_c = ',num2str(W_c_ratio(i))])
            disp(['     Parameter: PGA = ',num2str(PGA(i)), ',W_n = ',num2str(Wn(i))])
    
            GM_params = [W_g(i), beta(i) ,W_c_ratio(i)];
            [Time, ~, Vel, Disp] = fns_generateGM_Params([], [], 100 ,"Hu_S0", GM_params, [0.5, 1 ,4, 10], PGA(i));
            
            % FFT half-positive
            L = length(Vel);
            Fs = round(1/(Time(3)-Time(2)),200);
            data_FFT = fft(Vel);
            P2 = data_FFT;
            P1 = P2(1:floor(L/2+1));
            f = Fs*(0:(L/2))/L;
        
            % SDOF FRF, excited by Displacement
            Omega = 2*pi*f;
            FRF = (power(Omega,2))./(-power(Omega,2)+ 2*1i*Omega*DR*Wn(i)+ power(Wn(i),2));
        
            % Compute the Velocity
            Resp_f = P1.*transpose(FRF);
        
            % IFFT from half-positive
            Resp_f_pad = [Resp_f; conj(flipud(Resp_f(2:end-1,:)))];
            Resp_ifft = ifft(Resp_f_pad*Fs, L, 1, 'symmetric');
            Resp_ifft = Resp_ifft(1:L,:)/Fs;
        
            Y(i) = log(max(abs(Resp_ifft)));
    
        end
        file_name = sprintf('SDOF_SBAGM_Y_Pred_%d_num_%d.mat',valid_point,length(X.lnPGA));
        save(file_name,'Y','-mat');
    end





    if plot_ == true

        % FFT half-positive
        L = length(Vel);
        Fs = round(1/(Time(3)-Time(2)),200);
        data_FFT = fft(Vel);
        P2 = data_FFT;
        P1 = P2(1:floor(L/2+1));
        f = Fs*(0:(L/2))/L;

        % IFFT from half-positive
        P1_pad = [P1; conj(flipud(P1(2:end-1,:)))];
        P1_ifft = ifft(P1_pad*Fs, L, 1, 'symmetric');
        data_IFFT = P1_ifft(1:L,:)/Fs;

        % Plot the Velocity in time domain and check IFFT
        figure;
        subplot(2,1,1)
        plot(Time,Vel)
        hold on 
        plot(Time,data_IFFT,'--')
        grid on
        xlabel('Time','FontSize',13);
        ylabel('Velocity','FontSize',13);
        legend("Ground Motion","Ground Motion IFFT")
        title('Pseudo Acceleration Plot','FontSize',13)

        % Plot the Response in time domain
        subplot(2,1,2)
        plot(Time,Vel)
        hold on 
        plot(Time,Resp_ifft)
        grid on
        xlabel('Time','FontSize',13);
        ylabel('Velocity','FontSize',13);
        legend("Ground Motion","Response")
        title('Pseudo Displacement Plot','FontSize',13)

        % Plot the Velocity in Fz domain
        figure;
        subplot(2,1,1)
        plot(Omega,abs(P1))
        hold on 
        plot(Omega,abs(Resp_f))
        grid on
        xlabel('Frequency (Hz)','FontSize',13);
        ylabel('Velocity','FontSize',13);
        legend("Ground Motion","Response")
        title('Pseudo Acceleration Spectrum','FontSize',13)


        subplot(2,1,2)
        plot(Omega,abs(P1./transpose(-(1i*2*pi*f))))
        hold on 
        
        data_FFT = fft(Disp);
        P2 = data_FFT;
        P1 = P2(1:floor(L/2+1));
        plot(Omega,abs(P1),'--')

        grid on
        xlabel('Time','FontSize',13);
        ylabel('Displacement','FontSize',13);
        legend("Disp by int(Vel)","Disp")
        title('Pseudo Displacement Spectrum','FontSize',13)

        % Plot the FRF in Fz domain
        figure;
        subplot(2,1,1)
        plot(Omega,abs(FRF))
        grid on
        xlabel('Frequency (Hz)','FontSize',13);
        ylabel('Acc','FontSize',13);
        title('Pseudo Acceleration Spectrum','FontSize',13)

        subplot(2,1,2)
        plot(Omega,real(FRF))
        hold on 
        plot(Omega,imag(FRF))
        grid on
        xlabel('Time','FontSize',13);
        ylabel('Displacement','FontSize',13);
        title('Pseudo Displacement Spectrum','FontSize',13)
    end


end