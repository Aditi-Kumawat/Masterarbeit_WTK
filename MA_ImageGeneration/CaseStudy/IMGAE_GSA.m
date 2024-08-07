% Example Data
categories = {'$M_{L}$','[2, 0.75]', '[2, 1]','[3, 0.5]','[3, 0.75]', '[3, 1]', '[4, 0.5]','[4, 0.75]', '[4, 1]', 'Interpreter', 'latex'};
x1   = 48.08;
x2   = 14.23; 
x3   = 16.58;
x4   = 11.71;
x2x3 = 4.88;
other_x = 100-(x1+x2+x3+x4+x2x3);

y1   = 50.90;
y2   = 13.79; 
y3   = 14.26;
y4   = 16.42;
y2y3 = 2.58;
other_y = 100-(y1+y2+y3+y4+y2y3);

z1   = 50.49;
z2   = 16.32; 
z3   = 17.39;
z4   = 8.19;
z2z3 = 6.40;
other_z = 100-(z1+z2+z3+z4+z2z3);

data_x = [0,x2,0,x4,x2x3,other_x];  % Bar chart data
data_y = [0,y2,y3,0,y2y3,other_y];  % Bar chart data
data_z = [0,z2,0,z4,z2z3,other_z];  % Bar chart data

color_x = [x1,0,x3,0,0,0];
color_y = [y1,0,0,y4,0,0];
color_z = [z1,0,z3,0,0,0];

% Create a figure
figure;
subplot(3, 1, 1)
bar(data_x, 'FaceColor', [0 0.4470 0.7410], 'FaceAlpha', 0.3);
hold on
bar(color_x, 'FaceColor', [0.8500 0.3250 0.0980], 'FaceAlpha', 0.5);
ylabel('Number of coefficients', 'Interpreter', 'latex'); % Y-axis label for bar chart
% Step 3: Customize X-Axis Labels
xticks(1:6); % Set the tick locations to match the x data points
xticklabels({'$M_{L}$', '$R_{x}$', '$R_{y}$','$\omega_{g}/2\pi$','$(R_{x},R_{y})$','$Others$'}); % Set the custom tick labels with LaTeX
ax = gca;
ax.TickLabelInterpreter = 'latex';
ylabel('Percent (\%)', 'Interpreter', 'latex');
legend('Less significant','Top two significant')
grid on;
xlabel('GSA in x-direction', 'Interpreter', 'latex');


subplot(3, 1, 2)
bar(data_y, 'FaceColor', [0 0.4470 0.7410], 'FaceAlpha', 0.3);
hold on 
bar(color_y, 'FaceColor', [0.8500 0.3250 0.0980], 'FaceAlpha', 0.5);
ylabel('Number of coefficients', 'Interpreter', 'latex'); % Y-axis label for bar chart
% Step 3: Customize X-Axis Labels
xticks(1:6); % Set the tick locations to match the x data points
xticklabels({'$M_{L}$', '$R_{x}$', '$R_{y}$','$\omega_{g}/2\pi$','$(R_{x},R_{y})$','$Others$'}); % Set the custom tick labels with LaTeX
ax = gca;
ax.TickLabelInterpreter = 'latex';
ylabel('Percent (\%)', 'Interpreter', 'latex');
legend('Less significant','Top two significant')
grid on;
xlabel('GSA in y-direction', 'Interpreter', 'latex');


subplot(3, 1, 3)
bar(data_z, 'FaceColor', [0 0.4470 0.7410], 'FaceAlpha', 0.4);
hold on 
bar(color_z, 'FaceColor', [0.8500 0.3250 0.0980], 'FaceAlpha', 0.5);
ylabel('Number of coefficients', 'Interpreter', 'latex'); % Y-axis label for bar chart
% Step 3: Customize X-Axis Labels
xticks(1:6); % Set the tick locations to match the x data points
xticklabels({'$M_{L}$', '$R_{x}$', '$R_{y}$','$\omega_{g}/2\pi$','($R_{x},R_{y})$','$Others$'}); % Set the custom tick labels with LaTeX
ax = gca;
ax.TickLabelInterpreter = 'latex';
legend('Less significant','Top two significant')
ylabel('Percent (\%)', 'Interpreter', 'latex');
grid on;
xlabel('GSA in z-direction', 'Interpreter', 'latex');







