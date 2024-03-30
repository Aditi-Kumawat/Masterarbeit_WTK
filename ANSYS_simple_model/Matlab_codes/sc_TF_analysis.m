
%% Initialization
clear;clc;close all;

% Define the number of storeys, rooms in x- y-direction
n_str = 1;
n_rx = 1;
n_ry = 1;

% Define the length, width, and height of the building
l_vect=[2 3 4 5 6 7 8];
b_vect=[2 3 4 5 6 7 8];
% l_vect=[3 5 7];
% b_vect=[3 5 7];
h = 3;

% Define the type of foundation as either 'PLATE' or 'FOOTING'
ftyp = 'FOOTING';

% Define the velocity of the excitation
V_s = 450;

% Define the size of the elements
n_esize = 0.25;

% Calculate the length and width of the footing based on the
% foundation type
if strcmp(ftyp,'PLATE')
    B_f = n_esize/2;
    L_f = n_esize/2;
else
    B_f = 0.75;
    L_f = 0.75;
end
%% Importing Transfer Function
% Define the name of the folder where the results are stored
rf_fldr = 'UnitBld_GeomVary';
bf_nm = 'Disp_Center_%s_%d_l%d_b%d';

% Define the column names in the results table
cols = {'Freq', 'AMPL','PHASE','REAL','IMAG'};

% Define the components to extract from the results
cmpt = {'X', 'Y', 'Z'};
%%
n_c = length(cmpt);
Vabs_zCell = cell(1, n_str+1);
fref_cmplx_mat_1=cell(1, n_c);
DISP_cmplx_mat=cell(1, n_c);
VEL_abs_mat=cell(1, n_c);
fref_vel_amp_mat_1=cell(1, n_c);

for i_str = 0:n_str
    [f_vect,TF_amp_mat,TF_cpmlx_mat,lb_combs]=...
        fns_scatter.get_TF_scatter(n_str, n_rx, n_ry,...
        l_vect, b_vect, ftyp, V_s, L_f, B_f,...
        bf_nm,i_str,cmpt,n_c,rf_fldr,cols);

    %%
    TFabs_zcell{i_str+1} = TF_amp_mat{3}.*1e-3;
    TFabs_xcell{i_str+1} = TF_amp_mat{1}.*1e-3;
end
%%
plt_cmp='Z';
ylim=50;
fns_scatter.plt_scatter(TFabs_zcell,n_str,f_vect,rf_fldr,plt_cmp,ylim,lb_combs,V_s);
df=f_vect(3)-f_vect(2);
uzxlim_1_vect=[0 0];
uzxlim_2_vect=[2 2];
if V_s==450
    uzylim_1_vect=[0.4 0].*1e-3;
    uzylim_2_vect=[2.2 15].*1e-3;
elseif V_s==100
    uzylim_1_vect=[0.4 0].*1e-3;
    uzylim_2_vect=[2.2 15].*1e-3;
end
for i_flur = 1:2
    TF_abs_Zmat = TFabs_zcell{i_flur};
    [uz_x,uz_mean,uz_SDlow,uz_SDup]=...
        fns_scatter.get_mean_std(TF_abs_Zmat,...
        f_vect,df);
    uzxlim_1=uzxlim_1_vect(i_flur);
    uzxlim_2=uzxlim_2_vect(i_flur);
    uzylim_1=uzylim_1_vect(i_flur);
    uzylim_2=uzylim_2_vect(i_flur);
    fns_scatter.plt_TF_MeanSD(uz_x,uz_mean,uz_SDlow,uz_SDup,...
        uzxlim_1,uzxlim_2,uzylim_1,uzylim_2,i_flur,V_s,rf_fldr,plt_cmp)
    %     fns_scatter.plt_TF_MeanSD_noXYlim(uz_x,uz_mean,uz_SDlow,uz_SDup,...
    %         i_flur,V_s,rf_fldr)
end
%%
plt_cmp='X';
ylim=20;

fns_scatter.plt_scatter(TFabs_xcell,n_str,f_vect,rf_fldr,plt_cmp,ylim,lb_combs,V_s);

uxxlim_1_vect=[0 0];
uxxlim_2_vect=[2 2];
if V_s==450
    uxylim_1_vect=[0.4 0];
    uxylim_2_vect=[2.2 15];
elseif V_s==100
    uxylim_1_vect=[0.4 0];
    uxylim_2_vect=[2.2 15];
end
for i_flur = 1:2
    TF_abs_Xmat = TFabs_xcell{i_flur};
    [ux_x,ux_mean,ux_SDlow,ux_SDup]=...
        fns_scatter.get_mean_std(TF_abs_Xmat,...
        f_vect,df);
    uxxlim_1=uxxlim_1_vect(i_flur);
    uxxlim_2=uxxlim_2_vect(i_flur);
    uxylim_1=uxylim_1_vect(i_flur);
    uxylim_2=uxylim_2_vect(i_flur);
    fns_scatter.plt_TF_MeanSD(ux_x,ux_mean,ux_SDlow,ux_SDup,...
        uxxlim_1,uxxlim_2,uxylim_1,uxylim_2,i_flur,V_s,rf_fldr,plt_cmp)
    %     fns_scatter.plt_TF_MeanSD_noXYlim(ux_x,ux_mean,ux_SDlow,ux_SDup,...
    %         i_flur,V_s,rf_fldr)
end