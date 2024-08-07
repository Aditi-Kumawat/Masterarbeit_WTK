% Example Data
categories = {'[3, 0.75]', '[3, 1]', '[4, 0.75]', '[4, 1]'};
data = [25, 50, 40, 133];  % Bar chart data
avgData = [315.20,1794.88,1209.34,22236.11];  % Average values
maxData = [393.88, 2105.25, 1345.3, 24676.22];  % Maximum values
minData = [270.95,1585.72,1085.08,20990.95];   % Minimum values

avgData_M = [94.66,477.01,342.81,6537.87];  % Average values
maxData_M = [125.87,595.36,382.86,7214.79];  % Maximum values
minData_M = [74.53,364.39,320.15,6005.58];   % Minimum values

% Calculate Errors
upperError =log(maxData) - log(avgData);  % Errors above the average
lowerError =log(avgData) - log(minData);  % Errors below the average

% Calculate Errors
upperError_M = log(maxData_M) - log(avgData_M);  % Errors above the average
lowerError_M = log(avgData_M) - log(minData_M);  % Errors below the average

% Create a figure
figure;

% Plot Bar Chart on the right y-axis
yyaxis right;
bar(data, 'FaceColor', [0.8500 0.3250 0.0980], 'FaceAlpha', 0.3);
ylabel('Number of coefficients', 'Interpreter', 'latex'); % Y-axis label for bar chart
ylim([0,200]);

% Switch to the left y-axis
yyaxis left;
hold on;
% Plot Average Line with Error Bars
errorbar(1:length(avgData), log(avgData), lowerError, upperError, 'ko-', ...
    'MarkerFaceColor', 'k', 'LineWidth', 2, 'CapSize', 10);

errorbar(1:length(avgData), log(avgData_M), lowerError_M, upperError_M, 'ko-.', ...
    'MarkerFaceColor', 'k', 'LineWidth', 2, 'CapSize', 10);

ylabel('log($t_{train}$) (sec)', 'Interpreter', 'latex'); % Y-axis label for average and error bars

hold off;

% Customize Plot
set(gca, 'XTick', 1:length(categories), 'XTickLabel', categories); % Set X-axis labels
xlabel('[$p$, $q_{norm}  ]$', 'Interpreter', 'latex'); % X-axis label
%title('Time comparison of running in single / multiple cores'); % Title
legend('Single core', 'Multiple cores (num = 6)','Number of coefficients', 'Location', 'Best', 'Interpreter', 'latex'); % Legend
grid on
% Adjust figure size (optional)
%set(gcf, 'Position', [100, 100, 800, 600]); % Adjust figure size

% Ensure x-axis labels are displayed at the bottom
ax = gca;
ax.XAxisLocation = 'bottom';
