% Example Data
categories = {'[2, 0.5]','[2, 0.75]', '[2, 1]','[3, 0.5]','[3, 0.75]', '[3, 1]', '[4, 0.5]','[4, 0.75]', '[4, 1]'};
data = [7    12    23    15    26    59    26    42   149];  % Bar chart data
avgData = 1.0e+04*[0.0031    0.0043    0.0103    0.0085    0.0189    0.0817    0.0237    0.0538    1.3605];  % Average values
maxData = 1.0e+04*[0.0031    0.0046    0.0109    0.0106    0.0196    0.0923    0.0272    0.0693    1.7042];  % Maximum values
minData = 1.0e+04*[0.0030    0.0041    0.0093    0.0067    0.0180    0.0640    0.0182    0.0346    1.0195];   % Minimum values

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
xlabel('[$p$, $q_{norm}  ]$', 'Interpreter', 'latex'); % X-axis label
%title('Time comparison of different p and q'); % Title
legend( 'Training time (sec)','Number of coefficients', 'Location', 'Best', 'Interpreter', 'latex'); % Legend
grid on
% Adjust figure size (optional)
%set(gcf, 'Position', [100, 100, 800, 600]); % Adjust figure size

% Ensure x-axis labels are displayed at the bottom
ax = gca;
ax.XAxisLocation = 'bottom';
