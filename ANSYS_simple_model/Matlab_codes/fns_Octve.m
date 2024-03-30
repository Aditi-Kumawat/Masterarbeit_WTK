% This MATLAB code defines a class funs_Octave_analysis with three static
% methods. Here are the comments for each method:
% 1. get_octave_band: This method calculates the RMS values of a signal
% in frequency bands of 1/3 octave. It takes as input the signal vector,
% a frequency vector, and the frequency resolution.
% The method first defines a series of center frequencies in the 1/3
% octave bands, and then calculates the upper and lower frequencies for
% each band. The method then calculates the index of the frequency vector
% that corresponds to each band. Finally, the method calculates the RMS
% value of the signal in each band and returns these values along with
% the center frequencies of the bands.
%
% 2. get_vel_on_octave_scale: This method calculates the 1/3 octave band
% RMS values of the transfer function between two signals. The method
% takes as input a cell array containing the absolute values of the
% transfer function for different storeys, a reference velocity amplitude
% matrix, the number of storeys, a frequency range, a length vector, and
% the frequency resolution. The method first calculates the transfer
% function in decibels for each storey relative to the reference velocity.
% It then loops through each storey and each length in the length vector
% to calculate the RMS values of the transfer function in each 1/3 octave 
% band for each storey and length.
%
% 3. get_freq_range_iso: This method calculates the frequency range of the
% ISO 2631-1 frequency weighting standard for evaluating human exposure
% to vibration. The method takes as input a vector of center frequencies.
% The method first defines the ISO 2631-1 frequency bands, finds the
% indices of the closest values to the band limits, and constructs the
% frequency range in ISO 2631-1 bands. The method then calculates the
% number of 1/3-octave bands required to cover the frequency range.
%%
classdef fns_Octve
    methods (Static)
        %%
        function [Fn_rmsVect, f_cenVect]=get_octBand(Fun_vect,f_vect,df)

            f_19 = 1000;
            f_cenVect_1 = zeros(1, 31);
            f_cen = f_19;
            for i_f = 1:31
                f_cen = f_cen / (2^(1/3));
                f_cenVect_1(i_f) = f_cen;
            end
            f_cenVect_1 = [f_19 f_cenVect_1];
            f_cenVect_1 = fliplr(f_cenVect_1);
            f_lowVect_1 = f_cenVect_1 / (2^(1/6));
            f_upVect_1 = f_cenVect_1 * (2^(1/6));

            f_bandVect = [f_lowVect_1(1,1) f_upVect_1];
            f_bandVect = f_bandVect(f_bandVect < f_vect(end));
            f_cenVect = f_bandVect * 2^(1/6);
            f_cenVect = f_cenVect(1:end-1);

            idx_vect_1 = zeros(1, length(f_bandVect));
            for i_up = 1:length(f_bandVect)
                f_up = f_bandVect(i_up);
                idx_vect_1(i_up) = min(find(abs(f_vect - f_up) < df));
            end

            Fn_rmsVect = zeros(1, length(idx_vect_1)-1);
            for i_1 = 1:length(idx_vect_1)-1
                idx_F_1 = idx_vect_1(i_1);
                idx_F_2 = idx_vect_1(i_1+1);
                F_0 = mean(Fun_vect(idx_F_1:idx_F_2));
                Fn_rmsVect(i_1) = F_0;
            end

        end
        %%
        function [Vratio_dbCell,f_linVect,V_rms_cell,f_cenVect]=...
                get_V_octave(Vabs_zCell,ff_VzMat,n_str,f_in,l_vect, df)
            % TF_v_ff_to_v_foundation=n_storeys
            f_linVect=f_in(4:end);
            for i_str = 0:n_str
                Vabs_z_str=Vabs_zCell{i_str+1};
                Vabs_z_fndtn=Vabs_zCell{1};
                if i_str==0
                    Vratio_dbCell{i_str+1}=...
                        20*log10(Vabs_z_fndtn(4:end,:)./...
                        ff_VzMat(4:end,:));
                else
                    Vratio_dbCell{i_str+1}=...
                        20*log10(Vabs_z_str(4:end,:)./...
                        Vabs_z_fndtn(4:end,:));
                end
            end
            % n_cell is the number of storeys from 0 to n_storeys
            for i_str = 0:n_str
                for i_lb=1:length(l_vect)
                    Vratio_db_vect=Vratio_dbCell{i_str+1}(:,i_lb);
                    [Fun_rms_vect, f_cenVect] = fns_Octve.get_octBand(...
                        Vratio_db_vect, f_linVect, df);
                    V_rms_cell{i_str+1}(:, i_lb)=Fun_rms_vect;
                end
            end
        end
        %%
        % The frequency bands, 4, 8, 16, 31.5, 63, and 125 Hz, 
        % are the ISO 2631-1 frequency weighting standard for 
        % evaluating human exposure to vibration.
        % This standard specifies that the vibration levels 
        % should be weighted according to the frequency content of
        % the vibration signal, with greater weighting applied to 
        % frequencies in the range of 4 to 125 Hz.
        %
        % The choice of these particular frequency bands is based on 
        % the sensitivity of the human body to vibration in this 
        % frequency range. These frequencies are most likely to cause 
        % discomfort, pain, or injury to the human body, and are therefore
        % of most concern in vibration exposure assessments.

        function [f_iso,n_octBands]=get_fvect_iso(f_center_vect)
            % Define the ISO 2631-1 frequency bands
            f_bnd = [4 8 16 31.5 63 125];

            % Find the indices of the closest values to the band limits
            [~,idx_min] = min(abs(f_center_vect - f_bnd(1)));
            [~,idx_max] = min(abs(f_center_vect - f_bnd(end)));

            % Construct the frequency range in ISO 2631-1 bands
            f_iso = [];
            for i = 1:length(f_bnd)-1
                f_iso = [f_iso, logspace(log10(f_bnd(i)),...
                    log10(f_bnd(i+1)),4)];
            end
            f_iso = unique(f_iso);
            f_iso = f_iso(f_iso >= f_center_vect(idx_min) &...
                f_iso <= f_center_vect(idx_max));

            % Calculate the number of 1/3-octave bands required to 
            % cover the frequency range
            % An octave is a frequency ratio of 2:1
            % (i.e., the upper frequency is twice the lower frequency),
            % and a 1/3-octave is a frequency ratio of the 2^(1/3):1 
            % (approximately 1.26:1).
            % Therefore, to move up or down by a 1/3-octave,
            % we need to multiply or divide the frequency by 2^(1/3) 
            % (approximately 1.26).
            log_min_f = log10(min(f_iso));
            log_max_f = log10(max(f_iso));
            octBandwidth = log10(1.26);
            n_octBands = round((log_max_f - log_min_f) / octBandwidth);
        end
    end
end