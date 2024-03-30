classdef fns_plot
    methods (Static)
        %%
        function fldr_nm = get_fldrnm(n_str, n_rx, n_ry, l, b, ftyp,...
                V_s, L_f, B_f)
            fldr_nm = ['n_storeys_',num2str(n_str),'_n_rooms_X_',...
                num2str(n_rx),'_n_rooms_Y_',num2str(n_ry),...
                '_l',num2str(l),'_by_b',num2str(b),'_ftyp_',ftyp,...
                '_Vs_',num2str(V_s),...
                '_Lf_',num2str(L_f),'_Bf_',num2str(B_f)];

        end
        %%
        function fldr_nm = get_fldrnm_rec(n_str, n_rx, n_ry, l, b,...
                ftyp,V_s, L_f, B_f,st_dtls)
            fldr_nm = ['n_storeys_',num2str(n_str),'_n_rooms_X_',...
                num2str(n_rx),'_n_rooms_Y_',num2str(n_ry),...
                '_l',num2str(l),'_by_b',num2str(b),'_ftyp_',ftyp,...
                '_Vs_',num2str(V_s),...
                '_Lf_',num2str(L_f),'_Bf_',num2str(B_f),...
                '_st_dtls_',st_dtls];
        end

        function fldr_nm = get_fldrnm_wall_config(n_str, n_rx, n_ry, l, b,...
                ftyp,V_s, L_f, B_f,WallConfig)
            fldr_nm = ['n_storeys_',num2str(n_str),'_n_rooms_X_',...
                num2str(n_rx),'_n_rooms_Y_',num2str(n_ry),...
                '_l',num2str(l),'_by_b',num2str(b),'_ftyp_',ftyp,...
                '_Vs_',num2str(V_s),...
                '_Lf_',num2str(L_f),'_Bf_',num2str(B_f),...
                '_CONFIG_',WallConfig];
        end

        function fldr_nm = get_fldrnm_vary_DR(n_str, n_rx, n_ry, l, b,...
                ftyp,V_s, L_f, B_f,DampingRatio)
            fldr_nm = ['n_storeys_',num2str(n_str),'_n_rooms_X_',...
                num2str(n_rx),'_n_rooms_Y_',num2str(n_ry),...
                '_l',num2str(l),'_by_b',num2str(b),'_ftyp_',ftyp,...
                '_Vs_',num2str(V_s),...
                '_Lf_',num2str(L_f),'_Bf_',num2str(B_f),...
                '_DR_',DampingRatio];
        end
        %%
        function setPltProps(y_vect,i_c)
            set(gca,'XTickLabelMode','auto');
            set(gca,'YTickLabelMode','auto');
            ylabel(y_vect{i_c},'FontSize',11,'Interpreter','latex')
            set(gca,'FontSize',10, 'Box', 'on','LineWidth',1,...
                'TickLabelInterpreter','latex',...
                'TickLength',[0.01, 0.01]);
            %             xlim([0,60])
            % ylim([0,30])
            xlabel({'Frequency~(Hz)'},'FontSize',11,'Interpreter','latex')
            set(gca,'FontSize',10, 'Box', 'on','LineWidth',1,...
                'TickLabelInterpreter','latex',...
                'TickLength',[0.01, 0.01]);
            legend show
            legend('Box','off','Interpreter','latex','FontSize',10)
            hold on
            set(gcf,'Units','inches', 'Position', [18 3 4 2.5],...
                'PaperUnits', 'Inches', 'PaperSize', [4 2.5]);
        end
        %%
        function plt_ff(f, fnabs_mat,bf_nm,s_dir,...
                stn,date, time,y_lbl,typ)
            ha_cl = @colors;
            lStyl = {'-', '--', ':', '-.'};
            lcol = {ha_cl('boston university red'),ha_cl('cadmium green'),...
                ha_cl('denim')};

            figure
            subplot(3,1,1);
            plot(f{1}, fnabs_mat{1}, 'LineStyle', lStyl{1},...
                'Color', lcol{1},'DisplayName','X-dir','LineWidth', 1.2)
            %             xlim([0,80])
            legend('show', 'Box', 'off', 'Interpreter','latex',...
                'FontSize', 11)
            ylabel(y_lbl, 'FontSize', 12, 'Interpreter', 'latex')
            xlabel({'Frequency (Hz)'}, 'FontSize', 12,...
                'Interpreter', 'latex')
            subplot(3,1,2);
            plot(f{2}, fnabs_mat{2}, 'LineStyle', lStyl{1},...
                'Color', lcol{2},'DisplayName','Y-dir','LineWidth',1.2)
            %             xlim([0,80])
            ylabel(y_lbl, 'FontSize', 12, 'Interpreter', 'latex')
            xlabel({'Frequency (Hz)'}, 'FontSize', 12,...
                'Interpreter', 'latex')
            legend('show', 'Box', 'off', 'Interpreter', 'latex',...
                'FontSize', 11)
            subplot(3,1,3);
            plot(f{3}, fnabs_mat{3}, 'LineStyle', lStyl{1},...
                'Color', lcol{3},'DisplayName', 'Z-dir','LineWidth',1.2)

            %             xlim([0,80])
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
        function plt_ff_svrlstns(f, fnabs_mat,bf_nm,s_dir, stn,date,...
                time,y_lbl,x_lbl,typ,evnt)
            ha_cl = @colors;
            lStyl = {'-', '--', ':', '-.'};
            txt_l=sprintf(stn);

            for idx = 1:3
                subplot(3,1,idx);
                hold on;
                plot(f{idx}, 2*fnabs_mat{idx},...
                    'DisplayName',txt_l,'LineWidth', 0.8)
                legend('show', 'Box', 'off', 'Interpreter','latex',...
                    'FontSize', 8)
%                 ylim([0 1.5e-4])
%                 xlim([0 80])
%                 ylim([-1.5e-3 2e-3])
%                 xlim([0 15])
                ylim([-4e-4 6e-4])
                xlim([0 4])
                ylabel(y_lbl{idx}, 'FontSize', 10, 'Interpreter', 'latex')
                xlabel(x_lbl, 'FontSize', 10,...
                    'Interpreter', 'latex')
                set(gca, 'XTickLabelMode', 'auto');
                set(gca, 'YTickLabelMode', 'auto');
                %                 hold off
                pause(0.01)
                set(gca,'FontSize',8, 'Box', 'on','LineWidth',1,...
                    'TickLabelInterpreter', 'latex',...
                    'TickLength',[0.01, 0.01]);
            end
            %             hold off
            set(gcf, 'Renderer', 'painters');
            set(gcf, 'Units', 'inches', 'Position', [18 3 3.5 4.5],...
                'PaperUnits', 'Inches','PaperSize', [3.5 4.5]);

            filename = [sprintf(bf_nm, s_dir, evnt, date, time),...
                '_',typ, '.pdf'];
            filename1 = [sprintf(bf_nm, s_dir, evnt, date, time),...
                '_',typ, '.emf'];

            cd SAVE_FIGS
            cd FF_Data
            print(gcf, '-dpdf', '-r300', filename);
            saveas(gcf, fullfile(filename1));
            cd ..
            cd ..
        end
        %% function for plotting that uses for loop to plot the
        % amplitude of the displacement for each floor size.
        function plt_TF(f, U_mat,i_str, n_rx, n_ry, l_vect, b_vect, ...
                ftyp,V_s, L_f, B_f,i_c,cmpt,r_fldr,xl1)
            ha_cl = @colors;
            lStyl = {'-', '--', ':', '-.'};
            lcol = {ha_cl('persian orange'), ha_cl('ball blue'),...
                ha_cl('black'),ha_cl('cadmium green'),...
                ha_cl('dark goldenrod')};
            ustr_vect={'$u_x$~(m)','$u_y$~(m)','$u_z$~(m)'};
            figure
            for i_lb=1:length(l_vect)
                l=l_vect(i_lb);
                b=b_vect(i_lb);
                txt = ['Floor~size:~',num2str(l),'m~x',num2str(b),'m'];
                hold on
                plot(f,U_mat(:,i_lb),...
                    'linestyle',lStyl{mod(i_lb-1,numel(lStyl))+1},...
                    'DisplayName',txt,'LineWidth',1.5,...
                    'Color',lcol{mod(i_lb-1,numel(lcol))+1})
            end
            %             xlim([0,xl1])
            txt_TF=['Floor=~',num2str(i_str),',~Transfer~function'];
            title(txt_TF,'Interpreter','latex','FontSize',8)

            fns_plot.setPltProps(ustr_vect,i_c);

            filnm = ['TF_U_center',...
                cmpt{i_c}, num2str(i_str),'_n_rooms_X_', num2str(n_rx),...
                '_n_rooms_Y_', num2str(n_ry),'_ftyp_', ftyp,...
                '_Vs_', num2str(V_s),...
                '_Lf_', num2str(L_f), '_Bf_', num2str(B_f), '.png'];

            cd SAVE_FIGS
            if ~exist(r_fldr, 'dir')
                mkdir(r_fldr);
            end
            saveas(gcf, fullfile(r_fldr, filnm));
            cd ..
            cd ..
            cd Matlab_codes
        end
        %%
        function plt_VampZ_Oct(f_cenVect, Fun_rms_cell,i_str,...
                n_rx, n_ry, l_vect, b_vect, ftyp, V_s, L_f, B_f,...
                i_c,cmpt,f_iso,ustr_vect)
            ha_cl = @colors;
            lStyl = {'-', '--', ':', '-.'};
            lcol = {ha_cl('persian orange'), ha_cl('ball blue'),...
                ha_cl('black'),...
                ha_cl('cadmium green'), ha_cl('dark goldenrod')};

            figure
            for i_lb=1:length(l_vect)
                l=l_vect(i_lb);
                b=b_vect(i_lb);
                txt = ['Floor~size:~',num2str(l),'m~x',num2str(b),'m'];
                hold on
                plot(log10(f_cenVect),Fun_rms_cell(:,i_lb),...
                    'linestyle',lStyl{mod(i_lb-1,numel(lStyl))+1},...
                    'DisplayName',txt,'LineWidth',1.5,...
                    'Color',lcol{mod(i_lb-1,numel(lcol))+1})
            end

            txt_vlvl=['Floor=~',num2str(i_str),',~Velocity~levels'];
            title(txt_vlvl,'Interpreter','latex','FontSize',8)

            set(gca,'XTick',log10(f_iso),'XTickLabel',round(f_iso,1));
            set(gca,'YTickLabelMode','auto');
            ylabel(ustr_vect{i_str+1},'FontSize',11,'Interpreter','latex')
            set(gca,'FontSize',12, 'Box', 'on','LineWidth',1,...
                'TickLabelInterpreter','latex',...
                'TickLength',[0.01,0.01]);
            xlim([log10(f_iso(1)),log10(f_iso(end))])
            if i_str==0
                ylim([-2,4])
            else
                ylim([-5,25])
            end
            xlabel({'Frequency~(1/3-octave)'},'FontSize',11,...
                'Interpreter','latex')
            set(gca,'FontSize',12, 'Box', 'on','LineWidth',1,...
                'TickLabelInterpreter','latex',...
                'TickLength',[0.01,0.01]);
            legend show
            legend('Box','off','Interpreter','latex','FontSize',11)

            hold on
            set(gcf,'Units','inches', 'Position', [18 3 6 3],...
                'PaperUnits', 'Inches', 'PaperSize', [7.25, 9.125]);

            filename = ['Oct_Cmpr_Vamp_cntr_',...
                cmpt{i_c}, num2str(i_str),'_n_rooms_X_',...
                num2str(n_rx), '_n_rooms_Y_',num2str(n_ry),...
                '_ftyp_', ftyp,'_Vs_', num2str(V_s),...
                '_Lf_', num2str(L_f), '_Bf_', num2str(B_f), '.emf'];

            cd SAVE_FIGS
            cd Velocity_results
            saveas(gcf, filename);
            cd ..
            cd ..
        end
        
        %%
        function plt_VampZ_Oct1(f_cenVect, Fun_rms_cell,i_str,...
                n_rx, n_ry, l_vect, b_vect, ftyp, V_s, L_f, B_f,...
                i_c,cmpt,f_iso,ustr_vect)
            ha_cl = @colors;
            lStyl = {'-', '--', ':', '-.'};
            lcol = {ha_cl('persian orange'), ha_cl('ball blue'),...
                ha_cl('black'),...
                ha_cl('cadmium green'), ha_cl('dark goldenrod')};

            figure
            for i_lb=1:l_vect
                %                 l=l_vect(i_lb);
                %                 b=b_vect(i_lb);
                %                 txt = ['Floor~size:~',num2str(l),'m~x',num2str(b),'m'];
                hold on
                plot(log10(f_cenVect),Fun_rms_cell(:,i_lb),...
                    'linestyle',lStyl{mod(i_lb-1,numel(lStyl))+1},...
                    'LineWidth',1.5,...
                    'Color',lcol{mod(i_lb-1,numel(lcol))+1})
            end

            txt_vlvl=['Floor=~',num2str(i_str),',~Velocity~levels'];
            title(txt_vlvl,'Interpreter','latex','FontSize',8)

            set(gca,'XTick',log10(f_iso),'XTickLabel',round(f_iso,1));
            set(gca,'YTickLabelMode','auto');
            ylabel(ustr_vect{i_str+1},'FontSize',11,'Interpreter','latex')
            set(gca,'FontSize',12, 'Box', 'on','LineWidth',1,...
                'TickLabelInterpreter','latex',...
                'TickLength',[0.01,0.01]);
            xlim([log10(f_iso(1)),log10(f_iso(end))])
            if i_str==0
                ylim([-2,4])
            else
                ylim([-5,25])
            end
            xlabel({'Frequency~(1/3-octave)'},'FontSize',11,...
                'Interpreter','latex')
            set(gca,'FontSize',12, 'Box', 'on','LineWidth',1,...
                'TickLabelInterpreter','latex',...
                'TickLength',[0.01,0.01]);
            %             legend show
            %             legend('Box','off','Interpreter','latex','FontSize',11)

            hold on
            set(gcf,'Units','inches', 'Position', [18 3 6 3],...
                'PaperUnits', 'Inches', 'PaperSize', [6 3]);

            filename = ['Oct_Cmpr_Vamp_cntr_',...
                cmpt{i_c}, num2str(i_str),'_n_rooms_X_',...
                num2str(n_rx), '_n_rooms_Y_',num2str(n_ry),...
                '_ftyp_', ftyp,'_Vs_', num2str(V_s),...
                '_Lf_', num2str(L_f), '_Bf_', num2str(B_f), '.pdf'];

            cd SAVE_FIGS
            cd Velocity_results
            saveas(gcf, filename);
            cd ..
            cd ..
        end

        %%
        function plt_VampZ_db(f_linVect, vel_db_mat,i_str,...
                n_rx, n_ry, l_vect, b_vect, ftyp, V_s, L_f, B_f,...
                i_c,cmpt)
            ha_cl = @colors;
            lStyl = {'-', '--', ':', '-.'};
            lcol = {ha_cl('persian orange'), ha_cl('ball blue'),...
                ha_cl('black'),...
                ha_cl('cadmium green'), ha_cl('dark goldenrod')};
            ustr_vect={'$v_x$~(dB)','$v_y$~(dB)','$v_z$~(dB)'};
            figure
            for i_lb=1:length(l_vect)
                l=l_vect(i_lb);
                b=b_vect(i_lb);
                txt = ['Floor~size:~',num2str(l),'m~x',num2str(b),'m'];
                hold on
                plot(log10(f_linVect),vel_db_mat(:,i_lb),...
                    'linestyle',lStyl{mod(i_lb-1,numel(lStyl))+1},...
                    'DisplayName',txt,'LineWidth',1.5,...
                    'Color',lcol{mod(i_lb-1,numel(lcol))+1})
            end

            txt_vlvl=['Floor=~',num2str(i_str),',~Velocity~levels'];
            title(txt_vlvl,'Interpreter','latex','FontSize',8)

            fns_plot.setPltProps(ustr_vect,i_c);

            filename = ['Lin_Cmpr_Vamp_cntr_',...
                cmpt{i_c}, num2str(i_str),'_n_rooms_X_',...
                num2str(n_rx), '_n_rooms_Y_',num2str(n_ry),...
                '_ftyp_', ftyp,'_Vs_', num2str(V_s),...
                '_Lf_', num2str(L_f), '_Bf_', num2str(B_f), '.png'];

            cd SAVE_FIGS
            cd Velocity_results
            saveas(gcf, filename);
            cd ..
            cd ..
        end

    end
end