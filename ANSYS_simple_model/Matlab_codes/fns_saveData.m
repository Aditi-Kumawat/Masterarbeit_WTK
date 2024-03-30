classdef fns_saveData
    methods (Static)
        function fn_saveVifft(v_ifft_mat,t,r_fldr,fil_nm)
            cd SAVE_DATA
            if ~exist(r_fldr, 'dir')
                mkdir(r_fldr);
            end
            fileID = fopen(fullfile(r_fldr,fil_nm), 'w');
            fprintf(fileID, 't(s)\tv_x(t)\tv_y(t)\tv_z(t)\n');
            for i_t = 1:length(t)
                fprintf(fileID, ' %14.7e %14.7e %14.7e %14.7e\n',...
                    t(i_t), v_ifft_mat(i_t,1),...
                    v_ifft_mat(i_t,2), v_ifft_mat(i_t,3));
            end
            fclose(fileID);
            cd ..
            cd ..
            cd Matlab_codes
        end
        %%
        function fn_saveUiift(u_ifft_mat,t,r_fldr,fil_nm)
            cd SAVE_DATA
            if ~exist(r_fldr, 'dir')
                mkdir(r_fldr);
            end
            fileID = fopen(fullfile(r_fldr,fil_nm), 'w');
            fprintf(fileID, 't(s)\tv_x(t)\tv_y(t)\tv_z(t)\n');
            for i_t = 1:length(t)
                fprintf(fileID, ' %14.7e %14.7e %14.7e %14.7e\n',...
                    t(i_t), u_ifft_mat(i_t,1),...
                    u_ifft_mat(i_t,2), u_ifft_mat(i_t,3));
            end
            fclose(fileID);
            cd ..
            cd ..
            cd Matlab_codes
        end
        %%
        function fn_saveUF(data_f,f_vect,i_c,i_str,r_fldr)

            cd SAVE_DATA
            % Create the directory (if it doesn't already exist)
            if ~exist(r_fldr, 'dir')
                mkdir(r_fldr);
            end
            fil_nm = strcat(sprintf('fftd_%d_str_%d', i_c,i_str), '.txt');
            fileID = fopen(fullfile(r_fldr,fil_nm), 'w');

            fprintf(fileID, 'Spectra of the disp double single-sided \n');
            fprintf(fileID, 'IS Units (Hz, m, m/s, m/s^2) \n');
            fprintf(fileID, 'Receiver = %f - Channel = %d \n', i_str, i_c);
            fprintf(fileID, '-------------------- \n');

            % Write the column headers to the file
            fprintf(fileID, 'f(Hz)\tRe\tIm\tAbs\n');

            % Write the data to the file
            f=f_vect{i_c};
            for i_f = 1:length(f)
                datR=real(data_f{i_c}(i_f,:));
                datIm=imag(data_f{i_c}(i_f,:));
                datAb=abs(data_f{i_c}(i_f,:));
                fprintf(fileID, ' %14.7e %14.7e %14.7e %14.7e\n',...
                    f(i_f),datR,datIm,datAb);
            end

            % Close the file
            fclose(fileID);
            cd ..
            cd ..
            cd Matlab_codes
        end
        %%
        function fn_saveVF(data_f,f_vect,i_c,i_str,r_fldr)

            cd SAVE_DATA
            % Create the directory (if it doesn't already exist)
            if ~exist(r_fldr, 'dir')
                mkdir(r_fldr);
            end
            fil_nm = strcat(sprintf('fftv_%d_str_%d', i_c,i_str), '.txt');
            fileID = fopen(fullfile(r_fldr,fil_nm), 'w');

            fprintf(fileID, 'Spectra of the vel double single-sided \n');
            fprintf(fileID, 'IS Units (Hz, m, m/s, m/s^2) \n');
            fprintf(fileID, 'Receiver = %f - Channel = %d \n', i_str, i_c);
            fprintf(fileID, '-------------------- \n');

            % Write the column headers to the file
            fprintf(fileID, 'f(Hz)\tRe\tIm\tAbs\n');

            % Write the data to the file
            f=f_vect{i_c};
            for i_f = 1:length(f)
                datR=real(data_f{i_c}(i_f,:));
                datIm=imag(data_f{i_c}(i_f,:));
                datAb=abs(data_f{i_c}(i_f,:));
                fprintf(fileID, ' %14.7e %14.7e %14.7e %14.7e\n',...
                    f(i_f),datR,datIm,datAb);
            end

            % Close the file
            fclose(fileID);
            cd ..
            cd ..
            cd Matlab_codes
        end
    end
end
