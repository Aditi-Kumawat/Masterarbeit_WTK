% Example Data
categories = {'100', '200', '400', '800', '1600'};
data = [127, 138, 133, 149, 152];  % Bar chart data
avgData = [0.09769996521888315 , 0.044305567724442343, 0.0211143646510912,  0.01384842526264201, 0.00952325964997876];  % Average values
maxData = [0.44114076062214275 , 0.12013716177947316 , 0.0492890037144314, 0.03137255896121777, 0.018527180631563972];  % Maximum values
minData = [0.020240559362036642, 0.011281452251465712, 0.009203369458283775, 0.004121354643810716, 0.002917212842270944];   % Minimum values
q1 = [0.04451119586760369,0.03248632433040529,0.013623310536872902,0.01014122456525703,0.007018817541221238];
q2 = [0.06852359755491487,0.044178362374233476, 0.02058672362989388,0.012560111357730018,0.009024656138673343];
q3 = [0.11688992077987771,0.0520244618470873,0.027815786425438947,0.016238351846649207,0.011748371892041282];

% Calculate Errors
upperError =(maxData) - (avgData);  % Errors above the average
lowerError =(avgData) - (minData);  % Errors below the average


info_matrix = [minData;q1;q2;q3;maxData];
% Create a figure


figure;
% Plot Bar Chart on the right y-axis
%yyaxis right;
%bar(data, 'FaceColor', [0.8500 0.3250 0.0980], 'FaceAlpha', 0.3);
%ylabel('Number of coefficients', 'Interpreter', 'latex'); % Y-axis label for bar chart
%ylim([0,200]);

% Switch to the left y-axis
% Plot Average Line with Error Bars
%errorbar(1:length(avgData), (avgData), lowerError, upperError, 'ko-', ...
%    'MarkerFaceColor', 'k', 'LineWidth', 2, 'CapSize', 10);


semilogy(1:length(avgData),avgData,'ko-')

hold on
boxplot(info_matrix,'Whisker',2)

grid on
ylabel('$\epsilon_{val}$', 'Interpreter', 'latex'); % Y-axis label for average and error bars
hold off;
ylim([0.001,1])
% Customize Plot
set(gca, 'XTick', 1:length(categories), 'XTickLabel', categories); % Set X-axis labels
xlabel('$N$', 'Interpreter', 'latex'); % X-axis label
%title('p = 4, q = 1'); % Title
legend('Average error', 'Location', 'Best', 'Interpreter', 'latex'); % Legend

% Adjust figure size (optional)
%set(gcf, 'Position', [100, 100, 800, 600]); % Adjust figure size

% Ensure x-axis labels are displayed at the bottom
ax = gca;
ax.XAxisLocation = 'bottom';
