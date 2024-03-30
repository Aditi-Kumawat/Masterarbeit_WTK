classdef fns_unitgeomdb
    methods (Static)

        %%
        function plt_Vdb_MeanSD(x,y_mean,y_low,y_up,xlim_1,xlim_2,...
                ylim_1,ylim_2,i_fl,V_s,r_fldr,cmpt,y_lbl)
            % Create the plot
            figure
            hold on
            plot(x, y_mean, 'k', 'LineWidth', 1.5)
            fill([x; flipud(x)], [y_low; flipud(y_mean)], 'b',...
                'EdgeColor', 'none', 'FaceAlpha', 0.3)
            fill([x; flipud(x)], [y_mean; flipud(y_up)], 'r',...
                'EdgeColor', 'none', 'FaceAlpha', 0.3)
            xlim([xlim_1, xlim_2])
            ylim([ylim_1, ylim_2])
            xlabel('f (Hz)','FontSize',10,'Interpreter','latex')
            ylabel(y_lbl,'FontSize',10,'Interpreter','latex')
            legend('Mean', 'Standard~Deviation~Below~Mean',...
                'Standard~Deviation~Above~Mean')
            set(gca,'XTickLabelMode','auto');
            set(gca,'YTickLabelMode','auto');
            legend show
            legend('Box','off','Interpreter','latex','FontSize',10)
            hold on
            set(gca,'FontSize',10, 'Box', 'on','LineWidth',1,...
                'TickLabelInterpreter', 'latex','TickLength',[0.01, 0.01]);
            set(gcf,'Units','inches', 'Position', [18 3 4 3],...
                'PaperUnits', 'Inches', 'PaperSize', [4 3]);
            filnm = ['Vdb_Mean_Std_plot_',cmpt,'_',num2str(i_fl),...
                '_Vs_', num2str(V_s), '.pdf'];
            %             filnm1 = ['TF_Mean_Std_plot_',cmpt,'_',num2str(i_fl),...
            %                 '_Vs_', num2str(V_s), '.emf'];
            filnm2 = ['Vdb_Mean_Std_plot_',cmpt,'_',num2str(i_fl),...
                '_Vs_', num2str(V_s), '.png'];
            cd SAVE_FIGS
            if ~exist(r_fldr, 'dir')
                mkdir(r_fldr);
            end
            saveas(gcf, fullfile(r_fldr, filnm));
            %             saveas(gcf, fullfile(r_fldr, filnm1));
            saveas(gcf, fullfile(r_fldr, filnm2));
            cd ..
            cd ..
            cd Matlab_codes
        end
        %%
        function plt_Vrms(f_cenVect,V_rms_Zmat,f_iso,i_flur,V_s,ylbl)
            figure;
            hold on;

            % Define frequency band limits
            f_low = log10(f_cenVect / 2^(1/6));
            f_up = log10(f_cenVect * 2^(1/6));

            for j=1:size(V_rms_Zmat,2)
                for i=1:length(f_cenVect)
                    line([f_low(i) f_up(i)], [V_rms_Zmat(i,j) V_rms_Zmat(i,j)]);
                    if i < length(f_cenVect)
                        line([f_up(i) f_up(i)], [V_rms_Zmat(i,j) V_rms_Zmat(i+1,j)]);
                    end
                end
            end

            xlabel({'Frequency~(1/3-octave)'},'FontSize',11,...
                'Interpreter','latex')
            ylabel(ylbl,'FontSize',11,'Interpreter','latex')
            grid on;
            set(gca,'XTick',log10(f_iso),'XTickLabel',round(f_iso,1));
            xlim([log10(f_iso(1)),log10(f_iso(end))])
            hold off;
            set(gca,'FontSize',12, 'Box', 'on','LineWidth',1,...
                'TickLabelInterpreter','latex',...
                'TickLength',[0.01,0.01]);
            hold on
            set(gcf,'Units','inches', 'Position', [18 3 6 3],...
                'PaperUnits', 'Inches', 'PaperSize', [6 3]);

            filename = ['Vrms_db_geomvary_',num2str(i_flur),...
                '_Vs_', num2str(V_s), '.pdf'];

            cd SAVE_FIGS
            cd UnitGeom
            saveas(gcf, filename);
            cd ..
            cd ..
        end
        %%
        function V_rms_mean=plt_Vrms_stats(f_cenVect, V_rms_Zmat, i_flur, V_s, ylbl,cmp,stn,n_str)
            ha_cl = @colors;
            figure;
            hold on;
            f_iso = [0.8, 1, 1.25, 1.6, 2, 2.5, 3.15, 4, 5, 6.3,...
                8, 10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80];

            % Define frequency band limits
            f_low = log10(f_cenVect / 2^(1/6));
            f_up = log10(f_cenVect * 2^(1/6));

            % Calculate mean and standard deviation
            V_rms_mean = mean(V_rms_Zmat, 2);
            V_rms_std = std(V_rms_Zmat, 0, 2);

            % Plot mean and standard deviation with shading
            for i=1:length(f_cenVect)
                fill([f_low(i) f_up(i) f_up(i) f_low(i)],...
                    [V_rms_mean(i)-V_rms_std(i) V_rms_mean(i)-V_rms_std(i)...
                    V_rms_mean(i)+V_rms_std(i) V_rms_mean(i)+V_rms_std(i)],...
                    'b', 'FaceAlpha', 0.1);
                line([f_low(i) f_up(i)], [V_rms_mean(i) V_rms_mean(i)],...
                    'Color', ha_cl('red'),'LineWidth', 1.5);

                % Add vertical lines connecting ends of each band
                if i < length(f_cenVect)
                    line([f_up(i) f_low(i+1)], [V_rms_mean(i)+V_rms_std(i),...
                        V_rms_mean(i+1)-V_rms_std(i+1)], 'Color', 'k');
                end
            end
            xlabel({'Frequency~(1/3-octave),~Hz'},'FontSize',10, 'Interpreter','latex')
            ylabel(ylbl,'FontSize',10,'Interpreter','latex')
            grid off;

            label_frequencies = [1, 2, 4, 8, 16, 31.5, 63];
            set(gca, 'XTick', log10(f_iso));

            labels = repmat({' '}, 1, length(f_iso));

            for freq = label_frequencies
                idx = find(f_iso == freq);
                labels{idx} = num2str(freq);
            end

            set(gca, 'XTickLabel', labels);
            xtickangle(0)

            xlim([log10(f_iso(1)),log10(f_iso(end))])
            ylim([15 125])
            set(gca,'FontSize',8, 'Box', 'on','LineWidth',1,...
                'TickLabelInterpreter','latex','TickLength',[0.01,0.01]);
            hold on
            %             plot(log10(flin),vdblim,'r','LineWidth',1.5);
            %             legend('Stats plot', 'DIN4150_3 limits');
            %
            % mid_f = log10(2);
            % height_position = min(vdblim) + 5;
            % text(mid_f, height_position, 'DIN 4150-3', 'Color', 'r', 'HorizontalAlignment', 'center', 'FontSize', 8, 'Interpreter', 'none');

            set(gcf,'Units','inches', 'Position', [18 3 3.5 2],...
                'PaperUnits', 'Inches', 'PaperSize', [3.5 2]);

            filename = ['Vrms_db_geomvary_',cmp,'_',stn,'_nstr','_',num2str(n_str),...
                '_flur','_',num2str(i_flur),'_Vs_', num2str(V_s), '_mean_std.emf'];
            filename1 = sprintf('PPV_data_flur%d_Vs%d_stn%s_cmp%s', i_flur, V_s, stn, cmp);
            filename_csv = [filename1 '.csv'];
            data_to_save = [V_rms_mean, V_rms_std];
            max_v_mean=max(V_rms_mean)
            [maxrow_indx, ~] = find(bsxfun(@eq, V_rms_mean, max_v_mean));
            max_f=f_cenVect(maxrow_indx)
            cd SAVE_FIGS
            cd UnitGeom
            saveas(gcf, filename);
            writematrix(data_to_save, filename_csv);
            cd ..
            cd ..
        end
        %%
        function [f,v,v_db]=DIN4150_3_lims(v_ref,iflur)
            % Define frequency intervals with delta f of 0.2
            if iflur==1
                f1 = 0:0.2:10;
                f2 = 10.2:0.2:50;
                f3 = 50.2:0.2:100;
                lim1=5e-3;
                lim2=15e-3;
                lim3=20e-3;
                lim1db=20*log10(lim1/v_ref);
                lim2db=20*log10(lim2/v_ref);
                lim3db=20*log10(lim3/v_ref);
                % Define corresponding velocity values
                v1 = lim1 * ones(1, length(f1));
                v2 = linspace(lim1, lim2, length(f2));
                v3 = linspace(lim2, lim3, length(f3));
                % Define corresponding velocity values
                v1db = lim1db * ones(1, length(f1));
                v2db = linspace(lim1db, lim2db, length(f2));
                v3db = linspace(lim2db, lim3db, length(f3));
                % Combine frequencies and velocities for plotting
                f = [f1, f2, f3];
                v = [v1, v2, v3];
                v_db=[v1db, v2db, v3db];
            else
                f = 0:0.2:100;
                lim=15e-3;
                v= lim * ones(1, length(f));
                lim_db=20*log10(lim/v_ref);
                v_db = lim_db * ones(1, length(f));
            end
        end

        %%
        function plot_DIN4150_3(v_ref,n_flur,f_max3D,max_Vxyz3D,V_s,n_stns,stn_vect,name_evnt)
            ha_cl = @colors;
            lcol = {ha_cl('ball blue'),ha_cl('crimson'),...
                ha_cl('gray')};
            scatterHandles = zeros(1, n_stns);
            sz=80;
            for i_flur=1:n_flur
                [f,v,v_db]=fns_unitgeomdb.DIN4150_3_lims(v_ref,i_flur);
                figure
                for i_stn = 1:n_stns
                    f_max = f_max3D(i_stn,:,i_flur);
                    max_Vxyz = max_Vxyz3D(i_stn,:,i_flur);
                    plot(f,v)
                    hold on
                    scatterHandles(i_stn) = scatter(f_max,max_Vxyz,sz,'MarkerEdgeColor','k',...
                        'MarkerFaceColor',lcol{i_stn}, 'MarkerFaceAlpha', 0.6, 'MarkerEdgeAlpha', 0.6);
                end
                legend(scatterHandles, stn_vect,'Box', 'off','FontSize',8,'Interpreter','latex');
                ylim([0 2e-3])
                xlim([0 60])
                xlabel({'Frequency,~Hz'},'FontSize',10,...
                    'Interpreter','latex')
                ylabel({'$v_{max}$,~m/s'},'FontSize',10,'Interpreter','latex')
                grid off;
                hold off;
                set(gca,'FontSize',8, 'Box', 'on','LineWidth',1,...
                    'TickLabelInterpreter','latex',...
                    'TickLength',[0.01,0.01]);
                hold on
                set(gcf,'Units','inches', 'Position', [18 3 3.5 2.5],...
                    'PaperUnits', 'Inches', 'PaperSize', [3.5 2.5]);

                filename = [name_evnt,'_Vmax_DIN4150_3_cmpr_',num2str(i_flur),...
                    '_Vs_', num2str(V_s), '.pdf'];

                cd SAVE_FIGS
                cd UnitGeom
                saveas(gcf, filename);
                cd ..
                cd ..
            end
        end
        %%
        function plot_DIN4150_3_XYZ(v_ref,n_flur,f_max3D,max_Vxyz3D,V_s,n_stns,stn_vect,name_evnt,ylbl,cmp)
            ha_cl = @colors;
            lcol = {ha_cl('ball blue'),ha_cl('crimson'),...
                ha_cl('gray')};
            scatterHandles = zeros(1, n_stns);
            sz=80;
            for i_flur=n_flur
                                [f,v,v_db]=fns_unitgeomdb.DIN4150_3_lims(v_ref,i_flur);
                figure
                for i_stn = 1:n_stns
                    f_max = f_max3D(i_stn,:,i_flur);
                    max_Vxyz = max_Vxyz3D(i_stn,:,i_flur);
                                        plot(f,v)
                                        hold on
                    scatterHandles(i_stn) = scatter(f_max,max_Vxyz,sz,'MarkerEdgeColor','k',...
                        'MarkerFaceColor',lcol{i_stn}, 'MarkerFaceAlpha', 0.6, 'MarkerEdgeAlpha', 0.6);
      
                end
                legend(scatterHandles, stn_vect,'Box', 'off','FontSize',8,'Interpreter','latex');
                x_upper_lim_for_plot = max(max(f_max3D(:,:,i_flur)));
                y_upper_lim_for_plot = max(max(max_Vxyz3D(:,:,i_flur)));

                ylim([0 y_upper_lim_for_plot*1.2])
                xlim([0 floor(x_upper_lim_for_plot)+2])
                grid off;
                hold off;
                set(gca,'FontSize',10, 'Box', 'on','LineWidth',1,...
                    'TickLabelInterpreter','latex',...
                    'TickLength',[0.01,0.01]);
                xlabel({'Frequency,~Hz'},'FontSize',11,...
                    'Interpreter','latex')
                ylabel(ylbl,'FontSize',11,'Interpreter','latex')
                hold on
                set(gcf,'Units','inches', 'Position', [18 3 3 2.5],...
                    'PaperUnits', 'Inches', 'PaperSize', [3 2.5]);

                filename = [name_evnt,'V_',cmp,'_DIN4150_3_cmpr_flur_',num2str(i_flur),...
                    '_Vs_', num2str(V_s), '.emf'];

                cd SAVE_FIGS
                cd UnitGeom
                saveas(gcf, filename);
                cd ..
                cd ..
            end
            close all
        end

        function plot_DIN4150_2_XYZ(A_eval,n_flur,f_max3D,max_Vxyz3D,V_s,n_stns,stn_vect,name_evnt,ylbl,cmp)
            ha_cl = @colors;
            lcol = {ha_cl('ball blue'),ha_cl('crimson'),...
                ha_cl('gray')};
            scatterHandles = zeros(1, n_stns);
            sz=80;
             
            for i_flur=n_flur
                                
                figure
                y1 = yline(A_eval(1),'-r','Au');
                y1.LabelHorizontalAlignment = 'left';
                y2 = yline(A_eval(2),'-b','Ao');
                y2.LabelHorizontalAlignment = 'left';
                yline(A_eval(3),'-g');
                legend("off")

                for i_stn = 1:n_stns
                    
                    f_max = f_max3D(i_stn,:,i_flur);
                    max_Vxyz = max_Vxyz3D(i_stn,:,i_flur);
                                       
                    hold on
                    scatterHandles(i_stn) = scatter(f_max,max_Vxyz,sz,'MarkerEdgeColor','k',...
                        'MarkerFaceColor',lcol{i_stn}, 'MarkerFaceAlpha', 0.6, 'MarkerEdgeAlpha', 0.6);
                    
                end
                x_upper_lim_for_plot = max(max(f_max3D(:,:,i_flur)));
                y_upper_lim_for_plot = max(max(max_Vxyz3D(:,:,i_flur)));
                legend(scatterHandles,stn_vect ,'Box', 'off','FontSize',8,'Interpreter','latex');
                %ylim([0 5e-3])
                if y_upper_lim_for_plot >= A_eval(2)/2
                    y_upper_lim_for_plot = max(max(A_eval),y_upper_lim_for_plot);
                end

                ylim([0 y_upper_lim_for_plot*1.2])
                xlim([0 floor(x_upper_lim_for_plot)+2])
                grid off;
                hold off;
                set(gca,'FontSize',10, 'Box', 'on','LineWidth',1,...
                    'TickLabelInterpreter','latex',...
                    'TickLength',[0.01,0.01]);
                xlabel({'time,~s'},'FontSize',11,...
                    'Interpreter','latex')
                ylabel(ylbl,'FontSize',11,'Interpreter','latex')
                hold on
                set(gcf,'Units','inches', 'Position', [18 3 3 2.5],...
                    'PaperUnits', 'Inches', 'PaperSize', [3 2.5]);

                filename = [name_evnt,'V_',cmp,'_DIN4150_2_cmpr_flur_',num2str(i_flur),...
                    '_Vs_', num2str(V_s), '.emf'];

                cd SAVE_FIGS
                cd UnitGeom
                saveas(gcf, filename);
                cd ..
                cd ..                
            end
            close all
        end

        function plot_DIN4150_2_XYZ_wallconfig( A_eval, n_flur,f_max3D,max_Vxyz3D,V_s,n_stns,stn_vect,name_evnt,ylbl,cmp)
            ha_cl = @colors;
            lcol = {ha_cl('ball blue'),ha_cl('crimson'),...
                ha_cl('gray')};
            scatterHandles = zeros(1, n_stns);

            sz=80;

            f_max = zeros(length(f_max3D),1);
            max_Vxyz = zeros(length(f_max3D),1);
            f_upper_limit = zeros(1, n_stns);
            v_upper_limit = zeros(1, n_stns);

               folderName = sprintf('Vary_wall');
               for i_flur= 1:n_flur
                   sub_folderName = sprintf('%s%d', 'i_flur', i_flur);               
                   figure
                   y1 = yline(A_eval(1),'-r','Au');
                   y1.LabelHorizontalAlignment = 'left';

                   y2 = yline(A_eval(2),'-b','Ao');
                   y2.LabelHorizontalAlignment = 'left';

                   yline(A_eval(3),'-g');
                   legend("off")
                   for i_stn = 1:n_stns
                       for i_wallconfig = 1:length(f_max3D)
                           f_max(i_wallconfig) = f_max3D{i_wallconfig}(i_stn,:,i_flur);
                           max_Vxyz(i_wallconfig) = max_Vxyz3D{i_wallconfig}(i_stn,:,i_flur);
                       end 
                       f_upper_limit(i_stn)=max(f_max);
                       v_upper_limit(i_stn)=max(max_Vxyz);
                                            
                       hold on
                       scatterHandles(i_stn) = scatter(f_max,max_Vxyz,sz,'MarkerEdgeColor','k',...
                           'MarkerFaceColor',lcol{i_stn}, 'MarkerFaceAlpha', 0.6, 'MarkerEdgeAlpha', 0.6);

                   end
                   x_upper_lim_for_plot = max(f_upper_limit);
                   y_upper_lim_for_plot = max(v_upper_limit);
                   legend(scatterHandles, stn_vect,'Box', 'off','FontSize',8,'Interpreter','latex');
                   
                   %ylim([0 1e-3])
                   if y_upper_lim_for_plot >= A_eval(2)/2
                        y_upper_lim_for_plot = max(max(A_eval),y_upper_lim_for_plot);
                   end
                   ylim([0 y_upper_lim_for_plot*1.2])
                   xlim([0 floor(x_upper_lim_for_plot)+2])

                   
                   grid off;
                   hold off;
                   set(gca,'FontSize',10, 'Box', 'on','LineWidth',1,...
                       'TickLabelInterpreter','latex',...
                       'TickLength',[0.01,0.01]);
                   xlabel({'time,~s'},'FontSize',11,...
                       'Interpreter','latex')
                   ylabel(ylbl,'FontSize',11,'Interpreter','latex')
                   hold on
                   set(gcf,'Units','inches', 'Position', [18 3 3 2.5],...
                       'PaperUnits', 'Inches', 'PaperSize', [3 2.5]);
   
                   filename = ['VaryWall_',name_evnt,'_V_',cmp,'_DIN4150_2_cmpr_flur_',num2str(i_flur),...
                       '_Vs_', num2str(V_s), '.emf'];                   
                   cd SAVE_FIGS
                   % Check if the folder already exists
                   if exist(folderName, 'dir')
                       % If the folder exists, change the current directory to it
                       cd(folderName);
                   else
                       % If the folder doesn't exist, create it and then change to it
                       mkdir(folderName);
                       cd(folderName);
                   end
                   
                   if exist(sub_folderName, 'dir')
                       % If the folder exists, change the current directory to it
                       cd(sub_folderName);
                   else
                       % If the folder doesn't exist, create it and then change to it
                       mkdir(sub_folderName);
                       cd(sub_folderName);
                   end
                   saveas(gcf, filename);
                   cd ..
                   cd ..
                   cd ..
               end
               close all
                      
        end

        function plot_DIN4150_3_XYZ_wallconfig(v_ref,n_flur,f_max3D,max_Vxyz3D,V_s,n_stns,stn_vect,name_evnt,ylbl,cmp)
            ha_cl = @colors;
            lcol = {ha_cl('ball blue'),ha_cl('crimson'),...
                ha_cl('gray')};
            scatterHandles = zeros(1, n_stns);
            sz=80;
            f_max = zeros(length(f_max3D),1);
            max_Vxyz = zeros(length(f_max3D),1);

            f_upper_limit = zeros(1, n_stns);
            v_upper_limit = zeros(1, n_stns);
           
            folderName = sprintf('Vary_wall');

                for i_flur= 1:n_flur
                    sub_folderName = sprintf('%s%d', 'i_flur', i_flur);
                                    [f,v,v_db]=fns_unitgeomdb.DIN4150_3_lims(v_ref,i_flur);
                    figure
                    for i_stn = 1:n_stns
                       for i_wallconfig = 1:length(f_max3D)
                           f_max(i_wallconfig) = f_max3D{i_wallconfig}(i_stn,:,i_flur);
                           max_Vxyz(i_wallconfig) = max_Vxyz3D{i_wallconfig}(i_stn,:,i_flur);
                       end 

                       f_upper_limit(i_stn)=max(f_max);
                       v_upper_limit(i_stn)=max(max_Vxyz);

                       plot(f,v)
                       hold on

                       scatterHandles(i_stn) = scatter(f_max,max_Vxyz,sz,'MarkerEdgeColor','k',...
                           'MarkerFaceColor',lcol{i_stn}, 'MarkerFaceAlpha', 0.6, 'MarkerEdgeAlpha', 0.6);
                    end
                    
                    x_upper_lim_for_plot = max(f_upper_limit);
                    y_upper_lim_for_plot = max(v_upper_limit);
                    legend(scatterHandles, stn_vect,'Box', 'off','FontSize',8,'Interpreter','latex');

                    ylim([0 y_upper_lim_for_plot*1.2])
                    xlim([0 floor(x_upper_lim_for_plot)+2])
                    grid off;
                    hold off;
                    set(gca,'FontSize',10, 'Box', 'on','LineWidth',1,...
                        'TickLabelInterpreter','latex',...
                        'TickLength',[0.01,0.01]);
                    xlabel({'Frequency,~Hz'},'FontSize',11,...
                        'Interpreter','latex')
                    ylabel(ylbl,'FontSize',11,'Interpreter','latex')
                    hold on
                    set(gcf,'Units','inches', 'Position', [18 3 3 2.5],...
                        'PaperUnits', 'Inches', 'PaperSize', [3 2.5]);
    
                    filename = ['VaryWall_',name_evnt,'V_',cmp,'_DIN4150_3_cmpr_flur_',num2str(i_flur),...
                        '_Vs_', num2str(V_s), '.emf'];
    
                    cd SAVE_FIGS

                    % Check if the folder already exists
                    if exist(folderName, 'dir')
                        % If the folder exists, change the current directory to it
                        cd(folderName);
                    else
                        % If the folder doesn't exist, create it and then change to it
                        mkdir(folderName);
                        cd(folderName);
                    end

                    if exist(sub_folderName, 'dir')
                        % If the folder exists, change the current directory to it
                        cd(sub_folderName);
                    else
                        % If the folder doesn't exist, create it and then change to it
                        mkdir(sub_folderName);
                        cd(sub_folderName);
                    end
                    saveas(gcf, filename);
                    cd ..
                    cd ..
                    cd ..
                end
                close all
            end
        function plot_DIN4150_3_XYZ_varyDamp(v_ref,n_flur,f_max3D,max_Vxyz3D,V_s,n_stns,stn_vect,name_evnt,ylbl,cmp)
            ha_cl = @colors;
            lcol = {ha_cl('ball blue'),ha_cl('crimson'),...
                ha_cl('gray')};
            scatterHandles = zeros(1, n_stns);
            sz=80;
            f_max = zeros(length(f_max3D),1);
            max_Vxyz = zeros(length(f_max3D),1);

            f_upper_limit = zeros(1, n_stns);
            v_upper_limit = zeros(1, n_stns);
           
            folderName = sprintf('Vary_Damp');

                for i_flur= 1:n_flur
                    sub_folderName = sprintf('%s%d', 'i_flur', i_flur);
                                    [f,v,v_db]=fns_unitgeomdb.DIN4150_3_lims(v_ref,i_flur);
                    figure
                    for i_stn = 1:n_stns
                       for i_damp = 1:length(f_max3D)
                           f_max(i_damp) = f_max3D{i_damp}(i_stn,:,i_flur);
                           max_Vxyz(i_damp) = max_Vxyz3D{i_damp}(i_stn,:,i_flur);
                       end 

                       f_upper_limit(i_stn)=max(f_max);
                       v_upper_limit(i_stn)=max(max_Vxyz);

                       plot(f,v)
                       hold on

                       scatterHandles(i_stn) = scatter(f_max,max_Vxyz,sz,'MarkerEdgeColor','k',...
                           'MarkerFaceColor',lcol{i_stn}, 'MarkerFaceAlpha', 0.6, 'MarkerEdgeAlpha', 0.6);
                    end
                    
                    x_upper_lim_for_plot = max(f_upper_limit);
                    y_upper_lim_for_plot = max(v_upper_limit);
                    legend(scatterHandles, stn_vect,'Box', 'off','FontSize',8,'Interpreter','latex');

                    ylim([0 y_upper_lim_for_plot*1.2])
                    xlim([0 floor(x_upper_lim_for_plot)+2])
                    grid off;
                    hold off;
                    set(gca,'FontSize',10, 'Box', 'on','LineWidth',1,...
                        'TickLabelInterpreter','latex',...
                        'TickLength',[0.01,0.01]);
                    xlabel({'Frequency,~Hz'},'FontSize',11,...
                        'Interpreter','latex')
                    ylabel(ylbl,'FontSize',11,'Interpreter','latex')
                    hold on
                    set(gcf,'Units','inches', 'Position', [18 3 3 2.5],...
                        'PaperUnits', 'Inches', 'PaperSize', [3 2.5]);
    
                    filename = ['VaryWall_',name_evnt,'V_',cmp,'_DIN4150_3_cmpr_flur_',num2str(i_flur),...
                        '_Vs_', num2str(V_s), '.emf'];
    
                    cd SAVE_FIGS

                    % Check if the folder already exists
                    if exist(folderName, 'dir')
                        % If the folder exists, change the current directory to it
                        cd(folderName);
                    else
                        % If the folder doesn't exist, create it and then change to it
                        mkdir(folderName);
                        cd(folderName);
                    end

                    if exist(sub_folderName, 'dir')
                        % If the folder exists, change the current directory to it
                        cd(sub_folderName);
                    else
                        % If the folder doesn't exist, create it and then change to it
                        mkdir(sub_folderName);
                        cd(sub_folderName);
                    end
                    saveas(gcf, filename);
                    cd ..
                    cd ..
                    cd ..
                end
                close all
            end
        function plot_DIN4150_2_XYZ_varyDamp(A_eval, n_flur,f_max3D,max_Vxyz3D,V_s,n_stns,stn_vect,name_evnt,ylbl,cmp)
            ha_cl = @colors;
            lcol = {ha_cl('ball blue'),ha_cl('crimson'),...
                ha_cl('gray')};
            scatterHandles = zeros(1, n_stns);

            sz=80;

            f_max = zeros(length(f_max3D),1);
            max_Vxyz = zeros(length(f_max3D),1);

            f_upper_limit = zeros(1, n_stns);
            v_upper_limit = zeros(1, n_stns);

               folderName = sprintf('Vary_Damp');
               for i_flur= 1:n_flur
                   sub_folderName = sprintf('%s%d', 'i_flur', i_flur);
                                 
                   figure
                   y1 = yline(A_eval(1),'-r','Au');
                   y1.LabelHorizontalAlignment = 'left';

                   y2 = yline(A_eval(2),'-b','Ao');
                   y2.LabelHorizontalAlignment = 'left';

                   yline(A_eval(3),'-g');
                   legend("off")                   
                   for i_stn = 1:n_stns
                       for i_damp = 1:length(f_max3D)
                           f_max(i_damp) = f_max3D{i_damp}(i_stn,:,i_flur);
                           max_Vxyz(i_damp) = max_Vxyz3D{i_damp}(i_stn,:,i_flur);
                       end 
                       f_upper_limit(i_stn)=max(f_max);
                       v_upper_limit(i_stn)=max(max_Vxyz);

                       hold on
                       scatterHandles(i_stn) = scatter(f_max,max_Vxyz,sz,'MarkerEdgeColor','k',...
                           'MarkerFaceColor',lcol{i_stn}, 'MarkerFaceAlpha', 0.6, 'MarkerEdgeAlpha', 0.6);

                   end
                   x_upper_lim_for_plot = max(f_upper_limit);
                   y_upper_lim_for_plot = max(v_upper_limit);
                   legend(scatterHandles, stn_vect,'Box', 'off','FontSize',8,'Interpreter','latex');
                   %ylim([0 1e-3])
                   if y_upper_lim_for_plot >= A_eval(2)/2
                        y_upper_lim_for_plot = max(max(A_eval),y_upper_lim_for_plot);
                   end
                   ylim([0 y_upper_lim_for_plot*1.2])
                   xlim([0 floor(x_upper_lim_for_plot)+2])
                   grid off;
                   hold off;
                   set(gca,'FontSize',10, 'Box', 'on','LineWidth',1,...
                       'TickLabelInterpreter','latex',...
                       'TickLength',[0.01,0.01]);
                   xlabel({'time,~s'},'FontSize',11,...
                       'Interpreter','latex')
                   ylabel(ylbl,'FontSize',11,'Interpreter','latex')
                   hold on
                   set(gcf,'Units','inches', 'Position', [18 3 3 2.5],...
                       'PaperUnits', 'Inches', 'PaperSize', [3 2.5]);
   
                   filename = ['VaryWall_',name_evnt,'_V_',cmp,'_DIN4150_2_cmpr_flur_',num2str(i_flur),...
                       '_Vs_', num2str(V_s), '.emf'];                   
                   cd SAVE_FIGS
                   % Check if the folder already exists
                   if exist(folderName, 'dir')
                       % If the folder exists, change the current directory to it
                       cd(folderName);
                   else
                       % If the folder doesn't exist, create it and then change to it
                       mkdir(folderName);
                       cd(folderName);
                   end
                   
                   if exist(sub_folderName, 'dir')
                       % If the folder exists, change the current directory to it
                       cd(sub_folderName);
                   else
                       % If the folder doesn't exist, create it and then change to it
                       mkdir(sub_folderName);
                       cd(sub_folderName);
                   end
                   saveas(gcf, filename);
                   cd ..
                   cd ..
                   cd ..
               end
               close all
                      
        end
    end
end