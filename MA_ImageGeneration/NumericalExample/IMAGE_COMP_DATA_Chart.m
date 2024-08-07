% Example Data
categories = {'100', '200', '400', '800', '1600'};
data = [127, 138, 133, 149, 152];  % Bar chart data
avgData = [4551.67, 5668.36, 6537.87, 13605.35, 46854.96];  % Average values
maxData = [5316.83, 6393.03, 7214.79, 17041.74, 61060.44];  % Maximum values
minData = [3857.18, 5291.1, 6005.58, 10195.13, 35333.2];   % Minimum values

% Calculate Errors
upperError =log(maxData) - log(avgData);  % Errors above the average
lowerError =log(avgData) - log(minData);  % Errors below the average

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
ylabel('log($t_{train}$) (sec)', 'Interpreter', 'latex'); % Y-axis label for average and error bars
hold off;

% Customize Plot
set(gca, 'XTick', 1:length(categories), 'XTickLabel', categories); % Set X-axis labels
xlabel('$N$', 'Interpreter', 'latex'); % X-axis label
%title('p = 4, q = 1'); % Title
legend('Training time (sec) ','Number of coefficients', 'Location', 'Best', 'Interpreter', 'latex'); % Legend
grid on
% Adjust figure size (optional)
%set(gcf, 'Position', [100, 100, 800, 600]); % Adjust figure size

% Ensure x-axis labels are displayed at the bottom
ax = gca;
ax.XAxisLocation = 'bottom';
