function save_data(qnt, stn, date, time_evnt,ff_fldrnew)
for i_v=1:3
    cd ..
    file_name_2 = strcat(sprintf('%s_%d_%s_%s_%s',qnt, i_v, stn, date, time_evnt), '.txt');
    full_path = fullfile(ff_fldrnew, file_name_2);
    disp(['Trying to open: ', full_path]);
    fileID = fopen(full_path, 'w');
    fprintf(fileID, 'Transient signals (denoised)  \n');
    fprintf(fileID, 'Units as in the International System of Units (Hz, m, m/s, m/s^2) \n');
    fprintf(fileID, 'Station = %f - time= %f - date= %f  - Channel = %d \n', stn, time_evnt, date, i_v);
    fprintf(fileID, '-------------------- \n');

    fprintf(fileID, 't(s)\tRe\n');

    for i_t = 1:length(freq)
        fprintf(fileID, ' %14.7e %14.7e\n', t(i_t), real(fn_time(i_t,i_v)));
    end

    fclose(fileID);
    cd Matlab_codes
end

for i_v=1:3
    cd ..
    file_name_2 = strcat(sprintf('fft%s_%d_%s_%s_%s',qnt, i_v, stn, date, time_evnt), '.txt');
    full_path = fullfile(ff_fldrnew, file_name_2);
    disp(['Trying to open: ', full_path]);
    fileID = fopen(full_path, 'w');
    fprintf(fileID, 'frequency Spectra (only positive half of the frequency range, doubled in amplitude) \n');
    fprintf(fileID, 'Units as in the International System of Units (Hz, m, m/s, m/s^2) \n');
    fprintf(fileID, 'Station = %f - time= %f - date= %f  - Channel = %d \n', stn, time_evnt, date, i_v);
    fprintf(fileID, '-------------------- \n');

    fprintf(fileID, 'f(Hz)\tRe\tIm\tAbs\n');

    for i_f = 1:length(freq)
        fprintf(fileID, ' %14.7e %14.7e %14.7e %14.7e\n', freq(i_f), real(fn_fft_ss(i_f,i_v)), imag(fn_fft_ss(i_f,i_v)), abs(fn_fft_ss(i_f,i_v)));
    end

    fclose(fileID);
    cd Matlab_codes
end
end