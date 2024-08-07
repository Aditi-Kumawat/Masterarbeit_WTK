clear;
clc;
close all;

path = load('Statistic_info_type1_Z.mat');
all_path = path.statictic_info_type1;

%% Parameter evaluation
data_t = fns_import_time_data(all_path{5}{1},'txt',[]);
GMG = cls_GM_generator([],data_t, 100);
[GM,FRF_info,Time_info,GM_info] = GMG.generateTimeNonStaPesudoGMbyFit("Hu_S0",[200,1,0.5,GMG.PGA],[],1,1);

%% Perfrom response spectra analysis
[T_o,Spa_o,Spv_o,Sd_o] = fns_response_spectra(data_t.time(3)-data_t.time(2),data_t.ampl,5,9.81,10);


%num_repeat = 50;
%Acc_pesuedo = zeros(num_repeat,2000);
%Disp_pesuedo = zeros(num_repeat,2000);
%for i = 1:num_repeat
%    %Generate Pesuedo Ground Motioin based on those parameters
%    %PGA = GMG.PGA + GMG.PGA*0.3*randn(1);
%    [time,ampl,vel,disp] = fns_generateGM_Params([],data_t.time,100,"Hu_S0",...
%                              FRF_info,...
%                              Time_info,...
%                              PGA);
%    [T_p,Spa_p,Spv_p,Sd_p] = fns_response_spectra(time(3)-time(2),ampl,5,9.81,10);
%    Acc_pesuedo(i,:) = transpose(Spa_p);
%    Disp_pesuedo(i,:) = transpose(Sd_p);
%end
%
%%% Plot Spectra
%figure;
%subplot(2,1,1)
%%figure('Name','Spectral Displacement','NumberTitle','off')
%semilogx(T_o,Sd_o,'LineWidth',2., Color="#0072BD")
%hold on 
%semilogx(T_p,Disp_pesuedo,'LineWidth',2., Color="#EDB120")
%semilogx(T_o,Sd_o,'LineWidth',2., Color="#0072BD")
%grid on
%legend('Recorded','Simulated');
%xlabel('Period (sec)','FontSize',13);
%ylabel('Sd (m)','FontSize',13);
%title('Displacement Spectrum','FontSize',13)
%
%
%subplot(2,1,2)
%%figure('Name','Pseudo Acceleration Spectrum','NumberTitle','off')
%semilogx(T_o,Spa_o,'LineWidth',2., Color="#0072BD")
%hold on 
%semilogx(T_p,Acc_pesuedo,'LineWidth',2., Color="#EDB120")
%semilogx(T_o,Spa_o,'LineWidth',2., Color="#0072BD")
%grid on
%legend('Recorded','Simulated');
%xlabel('Period (sec)','FontSize',13);
%ylabel('Spa (g)','FontSize',13);
%title('Pseudo Acceleration Spectrum','FontSize',13);
