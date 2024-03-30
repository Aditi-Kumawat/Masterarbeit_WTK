classdef fns_mexihat
    methods (Static)
        %%
        function wavelet = def_mexihat(t, center_freq)
            t = t - 0.5;
            % Parameters for the Mexican hat wavelet (Ricker wavelet)
            sigma = 1 / (2 * pi * center_freq); % Standard deviation to control the width of the wavelet

            % Calculate the wavelet
            wavelet = (2 / (sqrt(3 * sigma) * (pi^(1/4)))) * (1 - (t / sigma).^2) .* exp(-(t.^2) / (2 * sigma^2));
        end
        %%
        function plt_ff_mexihat(f, fnabs_mat,bf_nm,s_dir,...
                stn,date, time,y_lbl,typ,freq_mexht_ss,mexht_fft_ss)
            ha_cl = @colors;
            lStyl = {'-', '--', ':', '-.'};
            lcol = {ha_cl('boston university red'),ha_cl('cadmium green'),...
                ha_cl('denim')};

            figure
            subplot(3,1,1);
            plot(f{1}, 2*fnabs_mat{1}, 'LineStyle', lStyl{1},...
                'Color', lcol{1},'DisplayName','free-field (X-dir)','LineWidth', 1.2)
            hold on
            plot(freq_mexht_ss, abs(mexht_fft_ss{1}), 'LineStyle', lStyl{2},...
                'Color', lcol{1},'DisplayName','mexican hat (X-dir)','LineWidth', 1.2)
            xlim([0,80])
            legend('show', 'Box', 'off', 'Interpreter','latex',...
                'FontSize', 11)
            ylabel(y_lbl, 'FontSize', 12, 'Interpreter', 'latex')
            xlabel({'Frequency (Hz)'}, 'FontSize', 12,...
                'Interpreter', 'latex')
            subplot(3,1,2);
            plot(f{2}, 2*fnabs_mat{2}, 'LineStyle', lStyl{1},...
                'Color', lcol{2},'DisplayName','Y-dir','LineWidth',1.2)
            hold on
            plot(freq_mexht_ss, abs(mexht_fft_ss{2}), 'LineStyle', lStyl{2},...
                'Color', lcol{1},'DisplayName','mexican hat (Y-dir)','LineWidth', 1.2)
            xlim([0,80])
            ylabel(y_lbl, 'FontSize', 12, 'Interpreter', 'latex')
            xlabel({'Frequency (Hz)'}, 'FontSize', 12,...
                'Interpreter', 'latex')
            legend('show', 'Box', 'off', 'Interpreter', 'latex',...
                'FontSize', 11)
            subplot(3,1,3);
            plot(f{3}, 2*fnabs_mat{3}, 'LineStyle', lStyl{1},...
                'Color', lcol{3},'DisplayName', 'Z-dir','LineWidth',1.2)
            hold on
            plot(freq_mexht_ss, abs(mexht_fft_ss{3}), 'LineStyle', lStyl{2},...
                'Color', lcol{1},'DisplayName','mexican hat (Z-dir)','LineWidth', 1.2)
            xlim([0,80])
            legend('show', 'Box', 'off', 'Interpreter', 'latex',...
                'FontSize', 11)
            xlabel({'Frequency (Hz)'}, 'FontSize', 12,...
                'Interpreter', 'latex')
            ylabel(y_lbl, 'FontSize', 12, 'Interpreter', 'latex')

            set(gca, 'XTickLabelMode', 'auto');
            set(gca, 'YTickLabelMode', 'auto');

            set(gcf, 'Units', 'inches', 'Position', [18 3 4.5 6],...
                'PaperUnits', 'Inches','PaperSize', [7.25, 9.125]);

            filename = [sprintf(bf_nm, s_dir, stn, date, time),...
                '_',typ,'.emf'];

            cd SAVE_FIGS
            cd FF_Data
            saveas(gcf, filename);
            cd ..
            cd ..
        end
        %%
        function plt_ff_svrlstns_mhat(f, fnabs_mat,stn,y_lbl,freq_mexht_ss,mexht_fft_ss)
            ha_cl = @colors;
            lStyl = {'-', '--', ':', '-.'};
            lcol = {ha_cl('boston university red'),ha_cl('cadmium green'), ha_cl('denim')};
            txt_l=sprintf(stn);

            for idx = 1:3
                subplot(3,1,idx);
                hold on;  % ensure new plots are added, not replacing old ones
                plot(f{idx}, fnabs_mat{idx}, 'LineStyle', lStyl{1}, 'DisplayName',txt_l,'LineWidth', 1.2)
                hold on
                plot(freq_mexht_ss, abs(mexht_fft_ss{idx}), 'LineStyle', lStyl{2},...
                    'Color', lcol{1},'DisplayName','mexican hat','LineWidth', 1.2)
                legend('show', 'Box', 'off', 'Interpreter','latex', 'FontSize', 11)
                ylabel(y_lbl, 'FontSize', 12, 'Interpreter', 'latex')
                xlabel({'Frequency (Hz)'}, 'FontSize', 12, 'Interpreter', 'latex')
            end

            set(gca, 'XTickLabelMode', 'auto');
            set(gca, 'YTickLabelMode', 'auto');

            set(gcf, 'Units', 'inches', 'Position', [18 3 4.5 6], 'PaperUnits', 'Inches','PaperSize', [7.25, 9.125]);
            %             filename = [sprintf(bf_nm, s_dir, stn, date, time),...
            %                 '_',typ,'.emf'];
            %
            %             cd SAVE_FIGS
            %             cd FF_Data
            %             saveas(gcf, filename);
            %             cd ..
            %             cd ..
        end
        %%
        function [sig_t_xyz]=get_tdmain_rslt(sigR_mat,...
                sigIm_mat,n_s,Fs,nfft,time,y_lbl,r_fldr)
            ltxt = {'X-dir','Y-dir','Z-dir'};

            figure
            for i_s = 1:n_s
                for i_col = 1:size(sigR_mat{i_s}, 2)
                    sigR = sigR_mat{i_s}(:, i_col);
                    sigIm = sigIm_mat{i_s}(:, i_col);
                    sigcmplx = sigR + 1i * sigIm;
                    % Shift the zero frequency to the center of the signal
                    sigcmplx_pad = [sigcmplx;...
                        conj(flipud(sigcmplx(2:end-1,:)))];
                    sig_ifft = (0.5)*ifft(sigcmplx_pad*Fs,...
                        nfft, 1, 'symmetric');
                    sig_ifft = sig_ifft(1:length(time),:);
                    sig_t_xyz{i_s,:}=sig_ifft;
                    t = (0:length(sig_ifft)-1) / Fs;
                    % Plot the time-domain signal
                    subplot(n_s, size(sigR_mat{i_s}, 2),...
                        (i_s-1)*size(sigR_mat{i_s}, 2) + i_col);
                    plot(t, real(sig_ifft),'DisplayName',ltxt{i_s},...
                        'LineWidth', 1.2);
                    xlabel('Time (s)','Interpreter','latex','FontSize',10);
                    ylabel(y_lbl,'Interpreter','latex','FontSize',10);
                    legend show
                    legend('Box','off','Interpreter','latex','FontSize',11)
                    set(gcf,'Units','inches', 'Position', [18 3 10 8],...
                        'PaperUnits', 'Inches',...
                        'PaperSize', [7.25, 9.125]);
                end
            end
            filename = ['Vel_t', '.png'];
            cd SAVE_FIGS
            if ~exist(r_fldr, 'dir')
                mkdir(r_fldr);
            end
            saveas(gcf, fullfile(r_fldr, filename));
            cd ..
            cd ..
            cd Matlab_codes

        end
    end
end