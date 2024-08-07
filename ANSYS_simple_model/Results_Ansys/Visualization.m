%% Initialization
clear;clc;close all;

% Define the number of storeys, rooms in x- y-direction
n_str = 3;
n_rx = 3;
n_ry = 4;


% Define the type of foundation as either 'PLATE' or 'FOOTING'
ftyp = 'FOOTING';

% Define the velocity of the excitation
V_s = 450;

% Define the size of the elements
n_esize = 0.5;

% Calculate the length and width of the footing based on the
% foundation type
if strcmp(ftyp,'PLATE')
    B_f = n_esize/2;
    L_f = n_esize/2;
else
    B_f = 0.75;
    L_f = 0.75;
end

%DR_index = 1;

floor_num = 2;

num_real = 501;



figure;
dir = 'Z';
ax1 = subplot(3,1,3);
for DR_index = 2:num_real
    folder_name = ['./DataFromServer/n_storeys_',num2str(n_str),'_n_rooms_X_',num2str(n_rx),'_n_rooms_Y_',num2str(n_ry),'_ftyp_',ftyp,'_Vs_',num2str(V_s),'_Lf_',num2str(L_f),'_Bf_',num2str(B_f),'_DR_',num2str(DR_index)];
    file_name = ['Disp_Center_',dir,'_',num2str(3),'.csv'];
    path = fullfile(folder_name,file_name );
    
    FRF = readtable(path);

    FRF.Freq(1) = 0;
    %FRF_R = interp1(FRF.Freq,FRF.REAL,f);
    %FRF_I = interp1(FRF.Freq,FRF.IMAG,f);
    FRF_complex = FRF.REAL + 1i*FRF.IMAG;
    
    % Differential, Disp FRF -> Vel FRF
    FRF_vel_complex = 2*pi*1i*FRF_complex.*FRF.Freq;

    hold on
    plot(ax1,FRF.Freq, abs(FRF_complex));
    hold on
end
title("FRFs in z-direction, 3rd floor ", 'Interpreter', 'latex')
xlabel("frequency (Hz)", 'Interpreter', 'latex')
ylabel("Amplitude ", 'Interpreter', 'latex')
grid on

dir = 'Z';
ax2 = subplot(3,1,2);
for DR_index = 2:num_real
    folder_name = ['./DataFromServer/n_storeys_',num2str(n_str),'_n_rooms_X_',num2str(n_rx),'_n_rooms_Y_',num2str(n_ry),'_ftyp_',ftyp,'_Vs_',num2str(V_s),'_Lf_',num2str(L_f),'_Bf_',num2str(B_f),'_DR_',num2str(DR_index)];
    file_name = ['Disp_Center_',dir,'_',num2str(2),'.csv'];
    path = fullfile(folder_name,file_name );
    
    FRF = readtable(path);

    FRF.Freq(1) = 0;
    %FRF_R = interp1(FRF.Freq,FRF.REAL,f);
    %FRF_I = interp1(FRF.Freq,FRF.IMAG,f);
    FRF_complex = FRF.REAL + 1i*FRF.IMAG;
    
    % Differential, Disp FRF -> Vel FRF
    FRF_vel_complex = 2*pi*1i*FRF_complex.*FRF.Freq;

    hold on
    plot(ax2,FRF.Freq, abs(FRF_complex));
    hold on
end
title("FRFs in z-direction, 2nd floor  ", 'Interpreter', 'latex')
xlabel("frequency (Hz)", 'Interpreter', 'latex')
ylabel("Amplitude", 'Interpreter', 'latex')
grid on

dir = 'Z';
ax3 = subplot(3,1,1);
for DR_index = 2:num_real
    folder_name = ['./DataFromServer/n_storeys_',num2str(n_str),'_n_rooms_X_',num2str(n_rx),'_n_rooms_Y_',num2str(n_ry),'_ftyp_',ftyp,'_Vs_',num2str(V_s),'_Lf_',num2str(L_f),'_Bf_',num2str(B_f),'_DR_',num2str(DR_index)];
    file_name = ['Disp_Center_',dir,'_',num2str(1),'.csv'];
    path = fullfile(folder_name,file_name );
    
    FRF = readtable(path);

    FRF.Freq(1) = 0;
    %FRF_R = interp1(FRF.Freq,FRF.REAL,f);
    %FRF_I = interp1(FRF.Freq,FRF.IMAG,f);
    FRF_complex = FRF.REAL + 1i*FRF.IMAG;
    
    % Differential, Disp FRF -> Vel FRF
    FRF_vel_complex = 2*pi*1i*FRF_complex.*FRF.Freq;

    hold on
    plot(ax3,FRF.Freq, abs(FRF_complex));
    hold on
end
title("FRFs in z-direction, 1st floor ", 'Interpreter', 'latex')
xlabel("frequency (Hz)", 'Interpreter', 'latex')
ylabel("Amplitude", 'Interpreter', 'latex')
grid on
%title("200 Realzation of FRF of velocity, Y dirs at 2 EG")

% Convert A4 size from mm to inches
%a4_width = 0.8*210 / 25.4;  % A4 width in inches
%a4_height = 0.8*297  / (25.4); % A4 height in inches

% Set the figure properties to fit an A4 page
%set(gcf, 'PaperUnits', 'inches');
%set(gcf, 'PaperSize', [a4_width, a4_height]);
%set(gcf, 'PaperPosition', [0, 0, a4_width, a4_height]);
%set(gcf, 'Position', [100, 100, a4_width*100, a4_height*100]);  % Display size for on-screen view

% Print the figure to a file, e.g., a PDF
% Print the figure to an SVG file
%print('plot_a4', '-dsvg');