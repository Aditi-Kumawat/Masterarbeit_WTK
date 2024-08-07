clear;
clc;
close all;



addpath('C:\Users\v196m\Desktop\master_project\Masterarbeit\TESTING_ENV\PredictionResult')
respone = load('Response_Y_FC2_3.mat');
mean_response = mean(respone.Y,3);
std_response = std(respone.Y,0,3);


% Reshape the data to fit a 10x10 grid
X = reshape(exp(-0.378+ 0.53*mean_response(:,1)), [10, 10]);
Y = reshape(8+7.5*mean_response(:,2), [10, 10]);
Z = reshape((mean_response(:,3)), [10, 10]);
%Z_up = reshape((mean_response(:,3)+std_response(:,3)), [10, 10]);
%Z_down = reshape((mean_response(:,3)-std_response(:,3)), [10, 10]);


% Create the surface plot
figure;
surf(X, Y, Z,'EdgeColor','interp','FaceColor','interp','FaceAlpha',0.9);
%hold on 
%surf(X, Y, Z_up,'EdgeColor','k','FaceColor','none','FaceAlpha',0.5);
%surf(X, Y, Z_down,'EdgeColor','k','FaceColor','none','FaceAlpha',0.5);
xlabel('$M_{L}$', 'Interpreter', 'latex');
ylabel('$R$', 'Interpreter', 'latex');
zlabel('$p_{f}$', 'Interpreter', 'latex');
xlim([0.26,2.5])
ylim([0,14])
colorbar;