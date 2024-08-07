clear;
clc;
close all;

addpath('C:\Users\v196m\Desktop\master_project\Masterarbeit\TESTING_ENV\PredictionResult')
Valid = load('Valid_sets_RAND.mat');
Pred = load('Pred_sets_RAND.mat');
[f, xi] = ksdensity(Pred.Y(:,7),'Bandwidth',0.5);

figure;
histogram(Valid.Y(:,7),'Normalization','pdf','FaceColor',[0 0.4470 0.7410])
hold on
plot(xi, f,'-.', 'LineWidth', 2);
xlabel('$ln(v_{max})$ for $\mathbf{X}_{sb}$, N = 500', 'Interpreter', 'latex');
ylabel('Normalized frequency', 'Interpreter', 'latex');
legend('Validation sets','SPCE Prediction')
grid on;

%% 
Valid_1 = load('Valid_sets_1.mat');
Valid_2 = load('Valid_sets_2.mat');
Valid_3 = load('Valid_sets_3.mat');
Valid_4 = load('Valid_sets_4.mat');
Valid_5 = load('Valid_sets_5.mat');

Pred_1 = load('Pred_sets_1.mat');
Pred_2 = load('Pred_sets_2.mat');
Pred_3 = load('Pred_sets_3.mat');
Pred_4 = load('Pred_sets_4.mat');
Pred_5 = load('Pred_sets_5.mat');

[f_1, xi_1] = ksdensity(Pred_1.Y(:,7),'Bandwidth',0.5);
[f_2, xi_2] = ksdensity(Pred_2.Y(:,7),'Bandwidth',0.5);
[f_3, xi_3] = ksdensity(Pred_3.Y(:,7),'Bandwidth',0.5);
[f_4, xi_4] = ksdensity(Pred_4.Y(:,7),'Bandwidth',0.5);
[f_5, xi_5] = ksdensity(Pred_5.Y(:,7),'Bandwidth',0.5);


%%

data_1 = Valid_1.Y(50,:);
data_2 = Valid_1.Y(150,:);
data_3 = Valid_1.Y(250,:);
data_4 = Valid_1.Y(350,:);
data_5 = Valid_1.Y(450,:);


%%
figure
subplot(4, 1, 1); % Create the first subplot in the first position
histogram(Valid_1.Y(:,7),'Normalization','pdf','FaceColor',[0 0.4470 0.7410])
hold on
plot(xi_1, f_1,'-.', 'LineWidth', 2);

X_sb = Iso2phy(data_1);
X_sb = round(X_sb,2);
Z_str = sprintf('%10.2f,', X_sb);
Z_str = ['[', Z_str(1:end-1) , ']']; 
xlabel(['$ln(v_{max})$ for $X_{sb,1} = $' Z_str ', N = 500'], 'Interpreter', 'latex');

ylabel('Normalized frequency', 'Interpreter', 'latex');
legend('Validation sets','SPCE Prediction')
grid on;


subplot(4, 1, 2); % Create the first subplot in the first position
histogram(Valid_2.Y(:,7),'Normalization','pdf','FaceColor',[0 0.4470 0.7410])
hold on
plot(xi_2, f_2,'-.', 'LineWidth', 2);

X_sb = Iso2phy(data_2);
X_sb = round(X_sb,2);
Z_str = sprintf('%10.2f,', X_sb);
Z_str = ['[', Z_str(1:end-1) , ']']; 
xlabel(['$ln(v_{max})$ for $X_{sb,2} = $' Z_str ', N = 500'], 'Interpreter', 'latex');

ylabel('Normalized frequency', 'Interpreter', 'latex');
legend('Validation sets','SPCE Prediction')
grid on;


subplot(4, 1, 3); % Create the first subplot in the first position
histogram(Valid_3.Y(:,7),'Normalization','pdf','FaceColor',[0 0.4470 0.7410])
hold on
plot(xi_3, f_3,'-.', 'LineWidth', 2);
X_sb = Iso2phy(data_3);
X_sb = round(X_sb,2);
Z_str = sprintf('%10.2f,', X_sb);
Z_str = ['[', Z_str(1:end-1) , ']']; 
xlabel(['$ln(v_{max})$ for $X_{sb,3} = $' Z_str ', N = 500'], 'Interpreter', 'latex');
ylabel('Normalized frequency', 'Interpreter', 'latex');
legend('Validation sets','SPCE Prediction')
grid on;


subplot(4, 1, 4); % Create the first subplot in the first position
histogram(Valid_4.Y(:,7),'Normalization','pdf','FaceColor',[0 0.4470 0.7410])
hold on
plot(xi_4, f_4,'-.', 'LineWidth', 2);
X_sb = Iso2phy(data_4);
X_sb = round(X_sb,2);
Z_str = sprintf('%10.2f,', X_sb);
Z_str = ['[', Z_str(1:end-1) , ']']; 
xlabel(['$ln(v_{max})$ for $X_{sb,4} = $' Z_str ', N = 500'], 'Interpreter', 'latex');
ylabel('Normalized frequency', 'Interpreter', 'latex');
legend('Validation sets','SPCE Prediction')
grid on;


function X_sb = Iso2phy(X_spce)
    X_sb = zeros(1,6);
    X_sb(1) = exp(-0.378+ 0.53*X_spce(1));
    X_sb(2) = 8+ 7.5*X_spce(2);
    X_sb(3) = 8+ 7.5*X_spce(3);
    X_sb(4) = exp(2.76+ 0.37*X_spce(4));
    X_sb(5) = 12+1*X_spce(5);
    X_sb(6) = exp(-3.2+0.1*X_spce(6));
end
