classdef fns_tdomn_eval
    methods (Static)
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
                    sig_ifft = ifft(sigcmplx_pad*Fs,...
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
        %%
        function [sig_ifft_mat,t]=get_tdmain_rslt_MAT(sigcmplx_mat,...
                bldcmplx_mat,n_s,Fs,nfft,time,l_vect,b_vect,i_str,...
                ftyp,V_s,L_f,B_f,y_lbl,r_fldr)
            figure
            for i_s = 1:n_s
                for i_col = 1:size(sigcmplx_mat{i_s}, 2)
                    sigcmplx = sigcmplx_mat{i_s}(:, i_col);
                    sigcmplx_pad = [sigcmplx;...
                        conj(flipud(sigcmplx(2:end-1,:)))];
                    sig_ifft = (0.5)*ifft(sigcmplx_pad*Fs,...
                        nfft, 1, 'symmetric');
                    sig_ifft = sig_ifft(1:length(time),:);

                    t = (0:length(sig_ifft)-1) / Fs;
                    % Plot the time-domain signal
                    l=l_vect(i_col);
                    b=b_vect(i_col);
                    txt = ['lxb:',num2str(l),'m~x',num2str(b),'m'];
                    txt_2 = ['Floor:',num2str(i_str),',~Component:',...
                        num2str(i_s)];
                    subplot(n_s, size(sigcmplx_mat{i_s}, 2),...
                        (i_s-1)*size(sigcmplx_mat{i_s}, 2) + i_col);
                    plot(t, real(sig_ifft), 'LineWidth', 1.2);
                    xlabel('Time (s)','Interpreter','latex','FontSize',10);
                    ylabel(y_lbl,'Interpreter','latex','FontSize',10);
                    title(txt_2,'Interpreter','latex','FontSize',12);
                    legend(txt);
                    legend('Box','off','Interpreter','latex','FontSize',11)
                    hold on
                    set(gcf,'Units','inches', 'Position', [18 3 10 8],...
                        'PaperUnits', 'Inches', 'PaperSize', [7.25, 9.125]);
                    sig_ifft_mat(:,i_s)=sig_ifft;
                end
            end
            filename = ['Vel_t_','_Floor_', num2str(i_str),...
                '_ftyp_', ftyp,'_Vs_', num2str(V_s),...
                '_Lf_', num2str(L_f), '_Bf_', num2str(B_f), '.png'];
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