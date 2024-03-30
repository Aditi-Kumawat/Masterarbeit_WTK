classdef fns_data_process
    methods (Static)
        function [fn_fft_ss,freq,fn_ifft,t]= fun_fftandifft(t_in,Fs,fn_in)
            nfft = 2^nextpow2(length(t_in));
            freq = Fs / 2 * linspace(0, 1, nfft/2+1);
            fn_fft = fft(fn_in, nfft) * (1/Fs);
            %             fn_fft_ss = 2 * fn_fft(1:nfft/2+1,:); %% orignal
            fn_fft_ss =fn_fft(1:nfft/2+1,:);
            %     u_fft_ss=v_fft_ss./1i./(2*pi*f_matrix);

            fn_fft_pad = [fn_fft_ss; conj(flipud(fn_fft_ss(2:end-1,:)))];
            %             fn_ifft = (0.5)*ifft(fn_fft_pad*Fs, nfft, 1, 'symmetric'); %
            fn_ifft = ifft(fn_fft_pad*Fs, nfft, 1, 'symmetric');
            fn_ifft = fn_ifft(1:length(t_in),:); % Truncate to same length as v
            t = (0:length(fn_ifft)-1) / Fs;
        end
        %%
        function save_data_time(qnt, stn, date, time_evnt,ff_fldrnew,t,fn_time)
            for i_v=1:3
                cd ..
                file_name_2 = strcat(sprintf('%s_%d_%s_%s_%s',qnt, i_v, stn, date, time_evnt), '.txt');
                full_path = fullfile(ff_fldrnew, file_name_2);
                disp(['Trying to open: ', full_path]);
                fileID = fopen(full_path, 'w');
                fprintf(fileID, 'Transient signals (denoised)  \n');
                fprintf(fileID, 'Units as in the International System of Units (Hz, m, m/s, m/s^2) \n');
                fprintf(fileID, 'Station, UTC, component values in the filename \n');
                fprintf(fileID, '-------------------- \n');

                fprintf(fileID, 't(s)\tRe\n');

                for i_t = 1:length(t)
                    fprintf(fileID, ' %14.7e %14.7e\n', t(i_t), real(fn_time(i_t,i_v)));
                end

                fclose(fileID);
                cd Matlab_codes
            end
        end
        function save_data_freq(qnt, stn, date, time_evnt,ff_fldrnew,freq,fn_fft_ss)
            for i_v=1:3
                cd ..
                file_name_2 = strcat(sprintf('fft%s_%d_%s_%s_%s',qnt, i_v, stn, date, time_evnt), '.txt');
                full_path = fullfile(ff_fldrnew, file_name_2);
                disp(['Trying to open: ', full_path]);
                fileID = fopen(full_path, 'w');
                fprintf(fileID, 'frequency Spectra (only positive half of the frequency range, not doubled in amplitude) \n');
                fprintf(fileID, 'Units as in the International System of Units (Hz, m, m/s, m/s^2) \n');
                fprintf(fileID, 'Station, UTC, component values in the filename \n');
                fprintf(fileID, '-------------------- \n');

                fprintf(fileID, 'f(Hz)\tRe\tIm\tAbs\n');

                for i_f = 1:length(freq)
                    fprintf(fileID, ' %14.7e %14.7e %14.7e %14.7e\n', freq(i_f), real(fn_fft_ss(i_f,i_v)), imag(fn_fft_ss(i_f,i_v)), abs(fn_fft_ss(i_f,i_v)));
                end

                fclose(fileID);
                cd Matlab_codes
            end
        end
    end
end