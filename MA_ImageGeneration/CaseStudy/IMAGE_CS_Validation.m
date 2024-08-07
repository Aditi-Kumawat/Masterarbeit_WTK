clear;
clc;
close all;

addpath('C:\Users\v196m\Desktop\master_project\Masterarbeit\TESTING_ENV\PredictionResult')
Valid = load('Valid_sets_X.mat');
Pred = load('Pred_sets_X.mat');
[f, xi] = ksdensity(Pred.Y(:,7),'Bandwidth',0.5);

figure;
subplot(3, 1, 1)
histogram(Valid.Y(:,7),'Normalization','pdf','FaceColor',[0 0.4470 0.7410])
hold on
plot(xi, f,'-.', 'LineWidth', 2);
xlabel('$ln(v_{max})$ for $\mathbf{X}_{sb}$ in x-direction (mm/s), N=500', 'Interpreter', 'latex');
ylabel('Normalized frequency', 'Interpreter', 'latex');
legend('Validation sets','SPCE Prediction')
xlim([-7,8])
grid on;


Valid = load('Valid_sets_Y.mat');
Pred = load('Pred_sets_Y.mat');
[f, xi] = ksdensity(Pred.Y(:,7),'Bandwidth',0.5);
subplot(3, 1, 2)
histogram(Valid.Y(:,7),'Normalization','pdf','FaceColor',[0 0.4470 0.7410])
hold on
plot(xi, f,'-.', 'LineWidth', 2);
xlabel('$ln(v_{max})$  for $\mathbf{X}_{sb}$ in y-direction (mm/s), N=500', 'Interpreter', 'latex');
ylabel('Normalized frequency', 'Interpreter', 'latex');
legend('Validation sets','SPCE Prediction')
grid on;
xlim([-7,8])


Valid = load('Valid_sets_RAND.mat');
Pred = load('Pred_sets_RAND.mat');
[f, xi] = ksdensity(Pred.Y(:,7),'Bandwidth',0.5);

subplot(3, 1, 3)
histogram(Valid.Y(:,7),'Normalization','pdf','FaceColor',[0 0.4470 0.7410])
hold on
plot(xi, f,'-.', 'LineWidth', 2);
xlabel('$ln(v_{max})$ for $\mathbf{X}_{sb}$ in z-direction (mm/s), N=500', 'Interpreter', 'latex');
ylabel('Normalized frequency', 'Interpreter', 'latex');
legend('Validation sets','SPCE Prediction')
grid on;
xlim([-7,8])





function X_sb = Iso2phy(X_spce)
    X_sb = zeros(1,6);
    X_sb(1) = exp(-0.378+ 0.53*X_spce(1));
    X_sb(2) = 8+ 7.5*X_spce(2);
    X_sb(3) = 8+ 7.5*X_spce(3);
    X_sb(4) = exp(2.76+ 0.37*X_spce(4));
    X_sb(5) = 12+1*X_spce(5);
    X_sb(6) = exp(-3.2+0.1*X_spce(6));
end
