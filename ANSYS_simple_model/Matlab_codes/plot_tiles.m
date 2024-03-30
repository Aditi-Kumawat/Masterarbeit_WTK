% Create the figure and subplot layout
classdef plot_tiles
    methods (Static)
        function plot_tiles_set_3(num_cols,num_rows,data1,data2,data3,x1,x2,x3,x_l_t,y_l_t,lg)
            figure;
            subplot_layout = reshape(1:num_cols*num_rows,num_cols, num_rows)';

            % Loop over the columns of the subplot
            for i = 1:num_rows
                % Plot v1 in the first row of subplot
                subplot(num_rows,num_cols,subplot_layout(i,1));
                plot(x1,data1(:, i),'DisplayName',lg{i}, 'LineWidth', 1.2);
                %                 title(['v(initial)',',~Component:',num2str(i)],'Interpreter','latex','FontSize',12);
                xlabel(x_l_t,'FontSize',11,'Interpreter','latex')
                ylabel(y_l_t,'FontSize',11,'Interpreter','latex')
                legend show
                legend('Box','off','Interpreter','latex','FontSize',11)

                % Plot v2 in the second row of subplot
                subplot(num_rows,num_cols,subplot_layout(i,2));
                plot(x2,data2(:, i),'DisplayName',lg{i}, 'LineWidth', 1.2);
                %                 title(['v(resampled)',',~Component:',num2str(i)],'Interpreter','latex','FontSize',12);
                xlabel(x_l_t,'FontSize',11,'Interpreter','latex')
                ylabel(y_l_t,'FontSize',11,'Interpreter','latex')
                legend show
                legend('Box','off','Interpreter','latex','FontSize',11)

                % Plot v3 in the third row of subplot
                subplot(num_rows,num_cols,subplot_layout(i,3));
                plot(x3,data3(:, i),'DisplayName',lg{i}, 'LineWidth', 1.2);
                %                 title(['v(IFFT)',',~Component:',num2str(i)],'Interpreter','latex','FontSize',12);
                xlabel(x_l_t,'FontSize',11,'Interpreter','latex')
                ylabel(y_l_t,'FontSize',11,'Interpreter','latex')
                legend show
                legend('Box','off','Interpreter','latex','FontSize',11)

            end
            set(gcf,'Units','inches', 'Position', [18 2 24 12],...
                'PaperUnits', 'Inches', 'PaperSize', [7.25, 9.125]);
        end

        function plot_tiles_set_2(num_cols,num_rows,data1,data2,x1,x2,x_l_f,y_l_f,x_l_t,y_l_t,lg)
            figure;
            subplot_layout = reshape(1:num_cols*num_rows,num_cols, num_rows)';

            % Loop over the columns of the subplot
            for i = 1:num_rows
                % Plot v1 in the first row of subplot
                subplot(num_rows,num_cols,subplot_layout(i,1));
                plot(x1,data1(:, i),'DisplayName',lg{i}, 'LineWidth', 1.2);
                xlabel(x_l_t,'FontSize',11,'Interpreter','latex')
                ylabel(y_l_t,'FontSize',11,'Interpreter','latex')
                legend show
                legend('Box','off','Interpreter','latex','FontSize',11)

                % Plot v2 in the second row of subplot
                subplot(num_rows,num_cols,subplot_layout(i,2));
                plot(x2,data2(:, i),'DisplayName',lg{i}, 'LineWidth', 1.2);
                xlabel(x_l_f,'FontSize',11,'Interpreter','latex')
                ylabel(y_l_f,'FontSize',11,'Interpreter','latex')
                legend show
                legend('Box','off','Interpreter','latex','FontSize',11)

            end
            set(gcf,'Units','inches', 'Position', [18 3 12 6],...
                'PaperUnits', 'Inches', 'PaperSize', [7.25, 9.125]);
            filename = ['disp_t_plus_fft', '.png'];
            saveas(gcf, fullfile(filename));
        end
    end
end
