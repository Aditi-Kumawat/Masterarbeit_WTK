% Define the grid size
n = 7;

% Create a meshgrid for the grid points
[X, Y] = meshgrid(1:n, 1:n);

% Initialize figure
figure


% Plot the grid points as red crosses
%plot(0:n,0,'-', 'LineWidth', 1)
%plot(0:n,n,'-', 'LineWidth', 1)

subplot(1,3,1)
plot(X, Y, 'rx', 'MarkerSize', 10, 'LineWidth', 1);
hold on;
plot(1:n,1,'bo', 'MarkerSize', 3, 'LineWidth', 9)
plot(1,1:n,'bo', 'MarkerSize', 3, 'LineWidth', 9)
plot(2,1:n-1,'bo', 'MarkerSize', 3, 'LineWidth', 9)
plot(3,1:n-2,'bo', 'MarkerSize', 3, 'LineWidth', 9)
plot(4,1:n-3,'bo', 'MarkerSize', 3, 'LineWidth', 9)
plot(5,1:n-4,'bo', 'MarkerSize', 3, 'LineWidth', 9)
plot(6,1:n-5,'bo', 'MarkerSize', 3, 'LineWidth', 9)
% Set the aspect ratio to be equal
axis equal;
% Force the plot to be square
axis square;
% Remove the axes ticks
set(gca, 'XTick', []);
set(gca, 'YTick', []);
xlim([1,n]);
ylim([1,n]);
% Add grid
grid on;
% Hold off to finish the plotting
hold off;

% Set xlabel with LaTeX interpreter
hXLabel = xlabel('Standard truncation $\mathcal{A}^{M,p}$', 'Interpreter', 'latex','FontSize',12);
% Get current position of xlabel
xlabelPosition = get(hXLabel, 'Position');
% Adjust the y-position to shift it down (e.g., -0.5 units)
xlabelPosition(2) = xlabelPosition(2) - 0.5;
% Set the new position
set(hXLabel, 'Position', xlabelPosition);


subplot(1,3,2)
plot(X, Y, 'rx', 'MarkerSize', 10, 'LineWidth', 1);
hold on;
plot(1:n,1,'bo', 'MarkerSize', 3, 'LineWidth', 9)
plot(1,1:n,'bo', 'MarkerSize', 3, 'LineWidth', 9)
plot(2,1:n-3,'bo', 'MarkerSize', 3, 'LineWidth', 9)
plot(3,1:n-5,'bo', 'MarkerSize', 3, 'LineWidth', 9)
plot(4,1:n-5,'bo', 'MarkerSize', 3, 'LineWidth', 9)
plot(5,1:n-6,'bo', 'MarkerSize', 3, 'LineWidth', 9)
%plot(6,1:n-5,'bo', 'MarkerSize', 3, 'LineWidth', 9)
% Set the aspect ratio to be equal
axis equal;
% Force the plot to be square
axis square;
% Remove the axes ticks
set(gca, 'XTick', []);
set(gca, 'YTick', []);
xlim([1,n]);
ylim([1,n]);
% Add grid
grid on;
% Hold off to finish the plotting
hold off;
% Set xlabel with LaTeX interpreter
hXLabel = xlabel('Hyperbolic truncation $\mathcal{A}^{M,p,q}$', 'Interpreter', 'latex','FontSize',12);
% Get current position of xlabel
xlabelPosition = get(hXLabel, 'Position');
% Adjust the y-position to shift it down (e.g., -0.5 units)
xlabelPosition(2) = xlabelPosition(2) - 0.5;
% Set the new position
set(hXLabel, 'Position', xlabelPosition);


subplot(1,3,3)
plot(X, Y, 'rx', 'MarkerSize', 10, 'LineWidth', 1);
hold on;
plot(1:n,1,'bo', 'MarkerSize', 3, 'LineWidth', 9)
plot(1,1:n,'bo', 'MarkerSize', 3, 'LineWidth', 9)
plot(2,1:n-3,'bo', 'MarkerSize', 3, 'LineWidth', 9)
plot(3,1:n-5,'bo', 'MarkerSize', 3, 'LineWidth', 9)
plot(4,1:n-5,'bo', 'MarkerSize', 3, 'LineWidth', 9)
plot(5,1:n-6,'bo', 'MarkerSize', 3, 'LineWidth', 9)

plot(2,2,'wo', 'MarkerSize', 3, 'LineWidth', 9)
plot(2, 2, 'rx', 'MarkerSize', 10, 'LineWidth', 1);

plot(1,3,'wo', 'MarkerSize', 3, 'LineWidth', 9)
plot(1, 3, 'rx', 'MarkerSize', 10, 'LineWidth', 1);

plot(3,1,'wo', 'MarkerSize', 3, 'LineWidth', 9)
plot(3,1, 'rx', 'MarkerSize', 10, 'LineWidth', 1);

plot(4,1,'wo', 'MarkerSize', 3, 'LineWidth', 9)
plot(4,1, 'rx', 'MarkerSize', 10, 'LineWidth', 1);

plot(5,1,'wo', 'MarkerSize', 3, 'LineWidth', 9)
plot(5,1, 'rx', 'MarkerSize', 10, 'LineWidth', 1);

plot(7,1,'wo', 'MarkerSize', 3, 'LineWidth', 9)
plot(7,1, 'rx', 'MarkerSize', 10, 'LineWidth', 1);
% Set the aspect ratio to be equal
axis equal;
% Force the plot to be square
axis square;
% Remove the axes ticks
set(gca, 'XTick', []);
set(gca, 'YTick', []);
xlim([1,n]);
ylim([1,n]);
% Add grid
grid on;
% Hold off to finish the plotting
hold off;
% Set xlabel with LaTeX interpreter
hXLabel = xlabel('Sparse truncation $\mathcal{A}^{\ast}$', 'Interpreter', 'latex','FontSize',12);
% Get current position of xlabel
xlabelPosition = get(hXLabel, 'Position');
% Adjust the y-position to shift it down (e.g., -0.5 units)
xlabelPosition(2) = xlabelPosition(2) - 0.5;
% Set the new position
set(hXLabel, 'Position', xlabelPosition);





