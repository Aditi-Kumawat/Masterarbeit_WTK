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
dir = 'Y';
floor_num = 1;


figure;

for DR_index = 1:10
    folder_name = ['./Nonuni_Y_7pt5/n_storeys_',num2str(n_str),'_n_rooms_X_',num2str(n_rx),'_n_rooms_Y_',num2str(n_ry),'_ftyp_',ftyp,'_Vs_',num2str(V_s),'_Lf_',num2str(L_f),'_Bf_',num2str(B_f),'_DR_',num2str(DR_index)];
    file_name = ['Disp_Center_',dir,'_',num2str(floor_num),'.csv'];
    path = fullfile(folder_name,file_name );
    
    FRF = readtable(path);
    FRF.AMPL(1:10,1)
    hold on
    plot(FRF.Freq, FRF.AMPL)
    hold on

end