
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
V_s = 100;

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
%% Importing Data
ff_fldr = 'GM/GM_UH/UH1_Part1';
% Define the base file name for the U center results in ANSYS
bf_nm_u = 'fftd_%d_%s_%s_%s';
bf_nm_v = 'fftv_%d_%s_%s_%s';
stn='UH1';
date='2013_04_16';
time='21_51_42';
% Define the column names in the results table
cols = {'Freq', 'Re','Im','Amp'};

% Define the components to extract from the results
s_dir = [1 2 3];
% length of free field data for each sensor is different
n_snr = numel(s_dir);

[f_inpt_U,ff_Uamp_mat,ff_Ur_mat,ff_UIm_mat,ff_Ucmplx_mat]=...
    fns_imprtdata.get_ff_inpt(bf_nm_u,s_dir,...
    stn,date, time,n_snr,ff_fldr,cols);
[f_inpt_V,ff_Vamp_mat,ff_Vr_mat,ff_VIm_mat,ff_Vcmplx_mat]=...
    fns_imprtdata.get_ff_inpt(bf_nm_v,s_dir,...
    stn,date, time,n_snr,ff_fldr,cols);
%% Importing Transfer Function
% Define the name of the folder where the results are stored
rf_fldr = 'UnitBld_GeomVary';
bf_nm = 'Disp_Center_%s_%d_l%d_b%d';

% Define the column names in the results table
cols1 = {'Freq', 'AMPL','PHASE','REAL','IMAG'};

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
        bf_nm,i_str,cmpt,n_c,rf_fldr,cols1);
    for i_c = 1:n_c
        %% Calculating Velocity
        TFcpmlx_intrp{i_c}=interp1(f_vect,TF_cpmlx_mat{i_c},...
            f_inpt_U{i_c},'linear','extrap');
        % Ucpmlx_intrp{i_cmp}=Ucpmlx_mat{i_cmp};
        Ucmplx_mat{i_c}=ff_Ucmplx_mat{i_c}.*TFcpmlx_intrp{i_c};

        Vabs_mat{i_c}=abs(Ucmplx_mat{i_c}.*1i.*f_inpt_U{i_c}*2*pi);
        Vcmplx_mat{i_c}=(Ucmplx_mat{i_c}.*1i.*f_inpt_U{i_c}*2*pi);
        Vdb_mat{i_c}=20*log10(Vabs_mat{i_c}(4:end,:)./...
                        5e-9);
        %         fns_saveSbplt.plt_Sbplt_lb(f_inpt_U,Vcmplx_mat,i_c,...
        %             i_str,y_lbl,l_vect,b_vect)
    end
    %     %%
    Vabs_zCell{i_str+1} = Vabs_mat{3};
    ff_VzMat=ff_Vamp_mat{3};
    Vabs_xCell{i_str+1} = Vabs_mat{1};
    ff_VxMat=ff_Vamp_mat{1};
    f_in=f_inpt_V{3};
    f_dB=f_inpt_V{3}(4:end);

end
df=f_in(10)-f_in(9);
[Vratio_dbCell,f_linVect,Vratio_rms_cell,f_cenVect]=...
    fns_Octve.get_V_octave(Vabs_zCell,ff_VzMat,n_str,f_in,size(lb_combs,1), df);
[f_iso,n_octBands]=fns_Octve.get_fvect_iso(f_cenVect);
ustr_vect={'20log10($v_{z0}/v_{ff}$)~(dB)',...
    '20log10($v_{z1}/v_{z0}$)~(dB)',...
    '20log10($v_{z2}/v_{z0}$)~(dB)'};
for i_str = 0:n_str
    Vratio_rms_mat=Vratio_rms_cell{i_str+1};
    Vratio_db_mat=Vratio_dbCell{i_str+1};
    % Plotting velocity levels Octave Bands (in octave 1/3 bands)
    fns_plot.plt_VampZ_Oct(f_cenVect, Vratio_rms_mat,...
        i_str, n_rx, n_ry, l_vect, b_vect, ftyp,...
        V_s, L_f, B_f,3,cmpt,f_iso,ustr_vect)
    % Plotting velocity levels in db over linear frequency range
    %     fns_plot.plt_VampZ_db(f_linVect, vel_db_mat,...
    %         i_str, n_rx, n_ry, l_vect, b_vect, ftyp,...
    %         V_s, L_f, B_f,3,cmpt)

end
%%
plt_cmp='Z';
ylim=50;
% fns_scatter.plt_scatter(TFabs_zcell,n_str,f_vect,rf_fldr,plt_cmp,ylim,lb_combs,V_s);
f_inpt_U=f_linVect;
df=f_inpt_U(3)-f_inpt_U(2);
vzxlim_1_vect=[0 0];
vzxlim_2_vect=[2.5 2.5];
% if V_s==450
vzylim_1_vect=[-6 -1];
vzylim_2_vect=[6 30];
% elseif V_s==100
%     vzylim_1_vect=[-0.5 0.5];
%     vzylim_2_vect=[-1 25];
% end
for i_flur = 1:2
    V_abs_Zmat = Vdb_mat{i_flur};
    [uz_x,uz_mean,uz_SDlow,uz_SDup]=...
        fns_scatter.get_mean_std(V_abs_Zmat,...
        f_dB,df);
    vzxlim_1=vzxlim_1_vect(i_flur);
    vzxlim_2=vzxlim_2_vect(i_flur);
    vzylim_1=vzylim_1_vect(i_flur);
    vzylim_2=vzylim_2_vect(i_flur);
    fns_scatter.plt_TF_MeanSD(uz_x,uz_mean,uz_SDlow,uz_SDup,...
        vzxlim_1,vzxlim_2,vzylim_1,vzylim_2,i_flur,V_s,rf_fldr,plt_cmp)
    %     fns_scatter.plt_TF_MeanSD_noXYlim(uz_x,uz_mean,uz_SDlow,uz_SDup,...
    %         i_flur,V_s,rf_fldr)
end

for i_flur = 1:2
    V_abs_Zmat = Vratio_dbCell{i_flur};
    [uz_x,uz_mean,uz_SDlow,uz_SDup]=...
        fns_scatter.get_mean_std(V_abs_Zmat,...
        f_inpt_U,df);
    vzxlim_1=vzxlim_1_vect(i_flur);
    vzxlim_2=vzxlim_2_vect(i_flur);
    vzylim_1=vzylim_1_vect(i_flur);
    vzylim_2=vzylim_2_vect(i_flur);
    fns_scatter.plt_TF_MeanSD(uz_x,uz_mean,uz_SDlow,uz_SDup,...
        vzxlim_1,vzxlim_2,vzylim_1,vzylim_2,i_flur,V_s,rf_fldr,plt_cmp)
    %     fns_scatter.plt_TF_MeanSD_noXYlim(uz_x,uz_mean,uz_SDlow,uz_SDup,...
    %         i_flur,V_s,rf_fldr)
end
for i_flur = 1:2
    V_rms_Zmat = Vratio_rms_cell{i_flur};
    [uz_x,uz_mean,uz_SDlow,uz_SDup]=...
        fns_scatter.get_mean_std1(V_rms_Zmat,...
        f_cenVect);
    vzxlim_1=vzxlim_1_vect(i_flur);
    vzxlim_2=vzxlim_2_vect(i_flur);
    vzylim_1=vzylim_1_vect(i_flur);
    vzylim_2=vzylim_2_vect(i_flur);
    fns_scatter.plt_TF_MeanSD(uz_x,uz_mean,uz_SDlow,uz_SDup,...
        vzxlim_1,vzxlim_2,vzylim_1,vzylim_2,i_flur,V_s,rf_fldr,plt_cmp)
    %     fns_scatter.plt_TF_MeanSD_noXYlim(uz_x,uz_mean,uz_SDlow,uz_SDup,...
    %         i_flur,V_s,rf_fldr)
end

%%
plt_cmp='X';
ylim=20;

% fns_scatter.plt_scatter(TFabs_xcell,n_str,f_vect,rf_fldr,plt_cmp,ylim,lb_combs,V_s);

vxxlim_1_vect=[0 0];
vxxlim_2_vect=[2 2];
if V_s==450
    vxylim_1_vect=[0.4 0];
    vxylim_2_vect=[2.2 15];
elseif V_s==100
    vxylim_1_vect=[0.4 0];
    vxylim_2_vect=[2.2 15];
end
for i_flur = 1:2
    V_abs_Xmat = Vabs_xcell{i_flur};
    [ux_x,ux_mean,ux_SDlow,ux_SDup]=...
        fns_scatter.get_mean_std(V_abs_Xmat,...
        f_inpt_U,df);
    vxxlim_1=vxxlim_1_vect(i_flur);
    vxxlim_2=vxxlim_2_vect(i_flur);
    vxylim_1=vxylim_1_vect(i_flur);
    vxylim_2=vxylim_2_vect(i_flur);
    fns_scatter.plt_TF_MeanSD(ux_x,ux_mean,ux_SDlow,ux_SDup,...
        vxxlim_1,vxxlim_2,vxylim_1,vxylim_2,i_flur,V_s,rf_fldr,plt_cmp)
    %     fns_scatter.plt_TF_MeanSD_noXYlim(ux_x,ux_mean,ux_SDlow,ux_SDup,...
    %         i_flur,V_s,rf_fldr)
end