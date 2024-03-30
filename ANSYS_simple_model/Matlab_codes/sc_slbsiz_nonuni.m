
%% Initialization
clear;clc;close all;

% Define the number of storeys, rooms in x-,y-direction
n_str = 2;
n_rx = 2;
n_ry = 3;

% Define the length, width, and height of the building
l_vect_off=5;
b_vect_off=4;
h = 3;
l_vect=5;
b_vect=5;
% Define the type of foundation as either 'PLATE' or 'FOOTING'
ftyp = 'FOOTING';
% Define the velocity of the excitation
V_s = 450;

% Define the size of the elements
n_esize = 0.25;

% Calculate the length and width of the footing based on the
% foundation type
if strcmp(ftyp,'PLATE')
    B_f = n_elesize/2;
    L_f = n_elesize/2;
else
    B_f = 0.75;
    L_f = 0.75;
end
%%
room_off='Y2_2m';

%% Importing Transfer Function


% Define the name of the folder where the results are stored
rf_fldr_slbsizVary = 'Nonuni_Y_7pt5';
rf_fldr = 'Nonuni_Y_7pt5';
bf_nm = 'Disp_Center_%s_%d_l%d_b%d';

% Define the column names in the results table
cols1 = {'Freq', 'AMPL','PHASE','REAL','IMAG'};

% Define the components to extract from the results
cmpt = {'X', 'Y', 'Z'};
%%
n_c = length(cmpt);
vel_abs_mat_Z_cell = cell(1, n_str+1);
fref_cmplx_mat_1=cell(1, n_c);
DISP_cmplx_mat=cell(1, n_c);
VEL_abs_mat=cell(1, n_c);
fref_vel_amp_mat_1=cell(1, n_c);

for i_str = 0:n_str
    [f_off,TFamp_off,~]=fns_slbsiz_nonuni.get_TFslbsiz_nonuni...
        (n_str, n_rx, n_ry,l_vect_off, b_vect_off, ftyp,...
        V_s, L_f, B_f,bf_nm,...
        i_str,cmpt,n_c,rf_fldr_slbsizVary,...
        cols1,room_off);
    [f_vect,TFamp,TFcpmlx]=fns_imprtdata.get_TF(n_str, n_rx, n_ry,...
        l_vect, b_vect, ftyp, V_s, L_f, B_f,...
        bf_nm,i_str,cmpt,n_c,rf_fldr,cols1);
    for i_c = 1:n_c
        %% Plotting Transfer Function
%         fns_plot.plt_TF(f_off,TFamp{i_c}, TFamp_off{i_c},...
%                     i_str, n_rx, n_ry, l_vect, b_vect, ftyp,...
%                     V_s, L_f, B_f, i_c, cmpt);
        fns_slbsiz_nonuni.plt_TFcmpr_slbsiz_nonuni(f_vect,f_off,...
            TFamp{i_c},TFamp_off{i_c},...
            i_str, n_rx, n_ry, l_vect, b_vect, ftyp,...
            V_s, L_f, B_f, i_c, cmpt,...
            room_off,l_vect_off, b_vect_off,rf_fldr_slbsizVary);
    end
end
