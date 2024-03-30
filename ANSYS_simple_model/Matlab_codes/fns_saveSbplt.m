classdef fns_saveSbplt
    methods (Static)
        function plt_Sbplt(f,reslt_mat,i_c,i_s,y_lbl)
            subplot(3, 1, i_c);
            plot(f{i_c}, abs(reslt_mat{i_c}), 'LineWidth', 1.2);
            xlabel('Time');
            ylabel(y_lbl);
            set(gcf,'Units','inches', 'Position', [18 3 6.5 7],...
                'PaperUnits', 'Inches', 'PaperSize', [7.25, 9.125]);
            txt_2 = ['Floor:',num2str(i_s),',~Component:',num2str(i_c)];
            title(txt_2,'Interpreter','latex','FontSize',12);
        end

        function plt_Sbplt_SS(f,reslt_mat,i_c,i_s,y_lbl,bld_mat)
            bld=bld_mat{i_s+1};
            bld=bld(:,i_c);
            subplot(3, 1, i_c);

            plot(f{i_c}, bld,'DisplayName','SeisSol', 'LineWidth', 1.2);
            hold on
            plot(f{i_c}, abs(reslt_mat{i_c}),'k','DisplayName',...
                'LPM-FEM', 'LineWidth', 1.2);
            xlabel('Frequency~(Hz)','Interpreter','latex','FontSize',12);
            ylabel(y_lbl,'Interpreter','latex','FontSize',12);
            xlim([0 50])
            legend show
            legend('Box','off','Interpreter','latex','FontSize',11)
            set(gcf,'Units','inches', 'Position', [18 3 4.5 6],...
                'PaperUnits', 'Inches', 'PaperSize', [7.25, 9.125]);
            txt_2 = ['Floor:',num2str(i_s),',~Component:',num2str(i_c)];
            title(txt_2,'Interpreter','latex','FontSize',11);
        end

        function plt_Sbplt_SS_lb(f,reslt_mat,i_c,i_s,y_lbl,bld_mat,...
                l_vect,b_vect)
              ha_cl = @colors;
            lStyl = {'-', '--', ':', '-.'};
            lcol = {ha_cl('persian orange'), ha_cl('ball blue'),...
                ha_cl('black'),...
                ha_cl('cadmium green'), ha_cl('dark goldenrod')};
            bld=bld_mat{i_s+1};
            bld=bld(:,i_c);
            subplot(3, 1, i_c);

%             plot(f{i_c}, bld,'DisplayName','SeisSol', 'LineWidth', 1.2);
            hold on
            for i_lb=1:length(l_vect)
                l=l_vect(i_lb);
                b=b_vect(i_lb);
                txt = ['Floor~size:~',num2str(l),'m~x',num2str(b),'m'];
            plot(f{i_c}, abs(reslt_mat{i_c}(:,i_lb)),...
                'linestyle',lStyl{mod(i_lb-1,numel(lStyl))+1},...
                'DisplayName',txt, 'LineWidth', 1.5,...
                    'Color',lcol{mod(i_lb-1,numel(lcol))+1})
            hold on
            end
            xlabel('Frequency~(Hz)','Interpreter','latex','FontSize',10);
            ylabel(y_lbl,'Interpreter','latex','FontSize',10);
            xlim([0 50])
            legend show
            legend('Box','off','Interpreter','latex','FontSize',11)
            set(gcf,'Units','inches', 'Position', [18 3 4.5 6],...
                'PaperUnits', 'Inches', 'PaperSize', [7.25, 9.125]);
            txt_2 = ['Floor:',num2str(i_s),',~Component:',num2str(i_c)];
            title(txt_2,'Interpreter','latex','FontSize',11);
        end

        function plt_Sbplt_lb(f,reslt_mat,i_c,i_s,y_lbl,...
                l_vect,b_vect)
              ha_cl = @colors;
            lStyl = {'-', '--', ':', '-.'};
            lcol = {ha_cl('persian orange'), ha_cl('ball blue'),...
                ha_cl('black'),...
                ha_cl('cadmium green'), ha_cl('dark goldenrod')};
            subplot(3, 1, i_c);
            for i_lb=1:length(l_vect)
                l=l_vect(i_lb);
                b=b_vect(i_lb);
                txt = ['Floor~size:~',num2str(l),'m~x',num2str(b),'m'];
            plot(f{i_c}, abs(reslt_mat{i_c}(:,i_lb)),...
                'linestyle',lStyl{mod(i_lb-1,numel(lStyl))+1},...
                'DisplayName',txt, 'LineWidth', 1.5,...
                    'Color',lcol{mod(i_lb-1,numel(lcol))+1})
            hold on
            end
            xlabel('Frequency~(Hz)','Interpreter','latex','FontSize',10);
            ylabel(y_lbl,'Interpreter','latex','FontSize',10);
            xlim([0 50])
            legend show
            legend('Box','off','Interpreter','latex','FontSize',11)
            set(gcf,'Units','inches', 'Position', [18 3 4.5 6],...
                'PaperUnits', 'Inches', 'PaperSize', [7.25, 9.125]);
            txt_2 = ['Floor:',num2str(i_s),',~Component:',num2str(i_c)];
            title(txt_2,'Interpreter','latex','FontSize',11);
        end

        function saveFig(r_fldr,fil_nm)
            cd SAVE_FIGS
            if ~exist(r_fldr, 'dir')
                mkdir(r_fldr);
            end
            saveas(gcf, fullfile(r_fldr, fil_nm));
            cd ..
            cd ..
            cd Matlab_codes
        end
    end
end