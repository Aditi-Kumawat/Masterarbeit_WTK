classdef fns_scatter_vary_damping
    methods (Static)
        %%
        function [f_vect,TFamp_mat,TFcmplx_mat,lb_comb]=...
                get_TF_scatter(n_str, n_rx, n_ry,l_vect, b_vect,...
                ftyp, V_s, L_f, B_f,damping_array,bf_nm,i_str,cmpt,n_c,r_fldr,cols)

            % Generate all possible combinations of l and b such that b<=l
            lb_comb_idx = [];
            lb_comb = [];
            for i_l = 1:length(l_vect)
                for i_b = 1:i_l
                    l = l_vect(i_l);
                    b = b_vect(i_b);
                    if b <= l
                        lb_comb_idx(end+1,:) = [i_l, i_b];
                        lb_comb(end+1,:) = [l, b];
                    end
                end
            end
            
            % Loop over all combinations of l and b
            for i_cmb = 1:size(lb_comb_idx, 1)
                i_l = lb_comb_idx(i_cmb, 1);
                i_b = lb_comb_idx(i_cmb, 2);
                l = l_vect(i_l);
                b = b_vect(i_b);
                
                for damping_num = 1:length(damping_array)
                    damping_ratio = damping_array(damping_num);
                    % Get the folder name for the specific damping_numuration
                    fldr = fns_plot.get_fldrnm_vary_DR(n_str,...
                        n_rx, n_ry, l, b, ftyp, V_s, L_f, B_f, num2str(damping_ratio));
                    fl_nm1 = arrayfun(@(x) sprintf(bf_nm, x{1}, i_str,...
                        l, b),cmpt, 'UniformOutput', false);
                    cd ..
                    cd Results_Ansys
                    fil_pth = fullfile(r_fldr, fldr, fl_nm1);
                    U_all = cellfun(@(x) readtable(x), fil_pth,...
                        'UniformOutput', false);
                    cd ..
                    cd Matlab_codes
                    
                    for i_component = 1:n_c
                        U = U_all{i_component};
                        U.Properties.VariableNames = cols;
                        if i_cmb == 1
                            f_vect = U.Freq;
                            TFamp_mat{damping_num,i_component} = U.AMPL;
                            TFr_mat{damping_num,i_component} = U.REAL;
                            TFim_mat{damping_num,i_component} = U.IMAG;
                            TFcmplx_mat{damping_num,i_component} = U.REAL+1i.*U.IMAG;
                        else
                            TFamp_mat{damping_num,i_component}(:, i_cmb) = U.AMPL;
                            TFr_mat{damping_num,i_component}(:, i_cmb)  = U.REAL;
                            TFim_mat{damping_num,i_component}(:, i_cmb)  = U.IMAG;
                            TFcmplx_mat{damping_num,i_component}(:, i_cmb)  = ...
                                U.REAL+1i.*U.IMAG;
                        end
                    end
                end
            end
        end

        
        %%
        function plt_scatter(TFabs_zcell,n_str,f_vect,rf_fldr,plt_cmp,y_lim,lb_combs,V_s)
            TF_abs_Zmat=TFabs_zcell{n_str+1};
            n_blds = size(TF_abs_Zmat, 2);
            [max_vals, max_indices] = max(TF_abs_Zmat);
            f_max_vals = f_vect(max_indices);
            max_vals_all = [];
            f_max_vals_all = [];
            max_vals_all = [max_vals_all, max_vals];
            f_max_vals_all = [f_max_vals_all, f_max_vals];

            % Convert lb_combs to string format
            lb_combs_str = cellstr(num2str(lb_combs));
            % Convert lb_combs to string format with the desired label format
            lb_combs_str1 = cell(size(lb_combs, 1), 1);
            for i = 1:size(lb_combs, 1)
                lb_combs_str1{i} = sprintf('%dmx%dm', lb_combs(i, 1), lb_combs(i, 2));
            end
            % Get unique values from lb_combs_str
            unique_lb_combs_str = unique(lb_combs_str);
            unique_lb_combs_str1 = unique(lb_combs_str1);

            % Determine the corresponding color indices for each unique lb_combs_str value
            color_indices = zeros(size(lb_combs_str));
            for i = 1:numel(unique_lb_combs_str)
                idx = strcmp(lb_combs_str, unique_lb_combs_str{i});
                color_indices(idx) = i;
            end
            % create horizontal line for the scatter plot
            y = ones(size(max_vals_all));

            % create colormap based on maximum values
            c = max_vals_all;
            cmap = jet(length(c));
            % Create a discrete colormap based on unique_lb_combs_str
            discrete_cmap = colormap(jet(numel(unique_lb_combs_str)));

            % Assign colors based on color_indices
            c = color_indices;

            % Make scatter plot with discrete colormap
            figure;
            s = scatter(f_max_vals_all, max_vals_all, 80, c, 'filled');
            colormap(discrete_cmap);
            s.MarkerEdgeColor = 'k';
            s.LineWidth = 1;
            s.MarkerFaceAlpha = 0.8;
            % Add colorbar with unique_lb_combs_str as tick labels
            cbar = colorbar('Ticks', 1:numel(unique_lb_combs_str1), 'TickLabels', unique_lb_combs_str1);

            %             cbar.Label.String = 'lb_combs';
            cbar.Label.FontSize = 12;
            cbar.TickLabelInterpreter = 'latex';

            % adjust plot properties
            xlim([min(f_vect), y_lim]);
            %             ylim([2, 12]);
            xlabel('Frequency (Hz)', 'FontSize', 10,...
                'Interpreter', 'latex');
            ylabel('max(Trasfer function)', 'FontSize', 10,...
                'Interpreter', 'latex');
            %             title('Scatter plot of Maximum values vs Frequency',...
            %                 'FontSize', 14, 'FontWeight', 'bold',...
            %                 'Interpreter', 'latex');
            set(gca, 'FontSize', 10, 'Box', 'on', 'LineWidth', 1,...
                'TickLabelInterpreter', 'latex', 'TickLength',[0.01,0.01]);
            set(gcf, 'Units', 'inches', 'Position', [18 3 4 5.5],...
                'PaperUnits', 'Inches', 'PaperSize', [4 5.5]);

            filnm = ['TF_Scatter_Plot_',plt_cmp,num2str(V_s), '.pdf'];
%             filnm1 = ['TF_Scatter_Plot_',plt_cmp,num2str(V_s), '.emf'];
            filnm2 = ['TF_Scatter_Plot_',plt_cmp,num2str(V_s), '.png'];
            cd SAVE_FIGS
            if ~exist(rf_fldr, 'dir')
                mkdir(rf_fldr);
            end
            saveas(gcf, fullfile(rf_fldr, filnm));
%             saveas(gcf, fullfile(rf_fldr, filnm1));
            saveas(gcf, fullfile(rf_fldr, filnm2));
            cd ..
            cd ..
            cd Matlab_codes
        end
        %%
        function [x,y_mean,y_low,y_up]=get_mean_std(TF_Zmat,f_vect,df)
            n_blds = size(TF_Zmat, 2);
            % Initialize variables to store the peak values and f
            max_vals = zeros(1, size(TF_Zmat, 2));
            f_max_vals = zeros(1, size(TF_Zmat, 2));

            % Find the peaks in each column of the matrix
            for i = 1:size(TF_Zmat, 2)
                [peaks, locs] = findpeaks(TF_Zmat(:,i));

                if isempty(peaks)
                    max_val = NaN;
                    f_max_val = NaN;
                else
                    % Find the index of the first peak
                    [~, idx] = min(locs);

                    % Get the amplitude and frequency of the first peak
                    max_val = peaks(idx);
                    f_max_val = f_vect(locs(idx));
                end

                % Store the peak values and frequencies for this column
                max_vals(i) = max_val;
                f_max_vals(i) = f_max_val;
            end
            % Generate building labels for legend
            bld_lbls = cell(1, n_blds);
            for i = 1:n_blds
                bld_lbls{i} = sprintf('Building %d', i);
            end

            % Normalize the f_range
            f_vect_norm = f_vect ./ max(f_vect);

            % Find mean transfer function across all buildings and storeys
            TF_mean = mean(TF_Zmat, 2);

            % Shift the f_range
            f_nrm=zeros(length(f_vect),n_blds);
            for i_nb=1:n_blds

                f_nrm(:,i_nb)=f_vect./f_max_vals(i_nb);
                if any(isnan(f_nrm(:,i_nb))) || any(isnan(TF_Zmat(:,i_nb)))
                    error('Input data contains NaN values')
                end
            end

            f_norm_out=f_nrm(1,1):df:f_nrm(end,end);

            for i_nb=1:n_blds
                TF_interp(:,i_nb) = interp1(f_nrm(:,i_nb),...
                    TF_Zmat(:,i_nb), f_norm_out, 'linear', 'extrap');
            end
            % Find mean transfer function across all buildings and storeys
            % Calculate the mean and standard deviation of TF_interp
            TF_mean = mean(TF_interp, 2);
            TF_std = std(TF_interp, 0, 2);

            % Define the x and y values for the plot
            x = f_norm_out';
            y_mean = TF_mean;
            y_low = TF_mean - TF_std;
            y_up = TF_mean + TF_std;
        end
        %%
        function [x,y_mean,y_low,y_up] = get_mean_std1(V_rms_mat,f_cenVect)

            n_blds = length(V_rms_mat);

            max_vals = zeros(1, n_blds);
            f_max_vals = zeros(1, n_blds);

            for i = 1:n_blds
                current_data = V_rms_mat(:,i);
                current_data(end+1) = 0;
                [peaks, locs] = findpeaks(current_data);

                if isempty(peaks)
                    max_val = NaN;
                    f_max_val = NaN;
                else
                    [~, idx] = min(locs);
                    max_val = peaks(idx);
                    f_max_val = f_cenVect(locs(idx));
                end

                max_vals(i) = max_val;
                f_max_vals(i) = f_max_val;
                if any(isnan(f_max_val))
                    error('f_max_val contains NaN values')
                end
            end

            f_nrm=zeros(length(f_cenVect),n_blds);
            for i_nb=1:n_blds
                f_nrm(:,i_nb)=f_cenVect./f_max_vals(i_nb);
                if any(isnan(f_nrm(:,i_nb)))
                    error('f_nrm contains NaN values')
                end
            end

            f_norm_out = logspace(log10(f_nrm(1,1)), log10(f_nrm(end,end)), length(f_cenVect));

            for i_nb=1:n_blds
                current_data = V_rms_mat(:,i_nb);
                data_interp(:,i_nb) = interp1(f_nrm(:,i_nb),...
                    current_data, f_norm_out, 'linear', 'extrap');
            end

            data_mean = mean(data_interp, 2);
            data_std = std(data_interp, 0, 2);

            x = f_norm_out';
            y_mean = data_mean;
            y_low = data_mean - data_std;
            y_up = data_mean + data_std;
        end


        %%
        function plt_TF_MeanSD(x,y_mean,y_low,y_up,xlim_1,xlim_2,...
                ylim_1,ylim_2,i_fl,V_s,r_fldr,cmpt)
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
            xlabel('$f/f_0$','FontSize',10,'Interpreter','latex')
            ylabel('Transfer~Function','FontSize',10,'Interpreter','latex')
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
            filnm = ['TF_Mean_Std_plot_',cmpt,'_',num2str(i_fl),...
                '_Vs_', num2str(V_s), '.pdf'];
            %             filnm1 = ['TF_Mean_Std_plot_',cmpt,'_',num2str(i_fl),...
            %                 '_Vs_', num2str(V_s), '.emf'];
            filnm2 = ['TF_Mean_Std_plot_',cmpt,'_',num2str(i_fl),...
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
        function plt_TF_MeanSD_noXYlim(x,y_mean,y_low,y_up,i_fl,V_s,r_fldr)
            % Create the plot
            figure
            hold on
            plot(x, y_mean, 'k', 'LineWidth', 1.5)
            fill([x; flipud(x)], [y_low; flipud(y_mean)], 'b',...
                'EdgeColor', 'none', 'FaceAlpha', 0.3)
            fill([x; flipud(x)], [y_mean; flipud(y_up)], 'r',...
                'EdgeColor', 'none', 'FaceAlpha', 0.3)
            xlabel('$f/f_0$','FontSize',11,'Interpreter','latex')
            ylabel('Transfer~Function','FontSize',11,'Interpreter','latex')
            legend('Mean', 'Standard~Deviation~Below~Mean',...
                'Standard~Deviation~Above~Mean')
            set(gca,'XTickLabelMode','auto');
            set(gca,'YTickLabelMode','auto');
            legend show
            legend('Box','off','Interpreter','latex','FontSize',11)
            hold on
            set(gcf,'Units','inches', 'Position', [18 3 6 4],...
                'PaperUnits', 'Inches', 'PaperSize', [7.25, 9.125]);
            filnm = ['TF_Mean_Std_plot_Full_length_',num2str(i_fl),...
                '_Vs_', num2str(V_s), '.png'];

            cd SAVE_FIGS
            if ~exist(r_fldr, 'dir')
                mkdir(r_fldr);
            end
            saveas(gcf, fullfile(r_fldr, filnm));
            cd ..
            cd ..
            cd Matlab_codes
        end
    end
end