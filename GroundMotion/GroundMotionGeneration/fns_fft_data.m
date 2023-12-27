function data = fns_fft_data(time_data_table,cut_off_freq,check_IFFT,plot_)
    time = time_data_table.time;
    ampl = time_data_table.ampl;
    L = length(ampl);
    Fs = round(1/(time(3)-time(2)),200);
    data_FFT = fft(ampl);
    P2 = data_FFT;
    
    if round(Fs/2) == cut_off_freq
        P1 = P2(1:L/2+1);
        f = Fs*(0:(L/2))/L;
    else
        disp(['WARNING, The detacted Fs of import data is ',int2str(Fs),' Hz']);
        disp(['     Positive half of the frequency range (CutOffFreq) should be ',num2str(Fs/2)]);
        disp(['     Using current CutOffFreq ',num2str(cut_off_freq), ' might lead to data distortion.']);
        f = 0:Fs/L:cut_off_freq;
        P1 = P2(1:L/2+1);
        P1 = P1(1:length(f));
    end
    Freq = transpose(f);
    Cmlx = P1;
    Real = real(P1);
    Imag = imag(P1);
    Ampl = abs(P1);

    data = table(Freq,Real,Imag,Ampl,Cmlx);

    
    if nargin < 3
        check_IFFT = false;  
    end

    if check_IFFT
        P1_pad = [P1; conj(flipud(P1(2:end-1,:)))];
        P1_ifft = ifft(P1_pad*Fs, L, 1, 'symmetric');
        data_IFFT = P1_ifft(1:L,:)/Fs;
        
        figure
        plot(time,ampl);
        hold on 
        plot(time,data_IFFT,'--');
        title('Checking FFT/IFFT');
        xlabel('time');
        ylabel('amplitude');
        legend("Original","IFFT")
        grid on;

    end

    if nargin < 4
        plot_ = false;  
    end

    if plot_ 
        figure
        plot(f,real(P1),f,imag(P1));
        title('Data after FFT');
        xlabel('freq');
        ylabel('Real/Imag');
        legend("Real","Imag")
        grid on;

        figure
        plot(f,abs(P1));
        title('Data after FFT');
        xlabel('freq');
        ylabel('Amplitude');
        grid on;
    end

end