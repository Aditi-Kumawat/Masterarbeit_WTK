%% Initialization
clear;clc;close all;

n_str = 3;
n_rx = 2;
n_ry = 3;

l_vect=[2 3 4 5 6 7 8];
b_vect=[2 3 4 5 6 7 8];
h = 3;
ftyp = 'PLATE';

V_s = 100
n_esize = 0.5;

if strcmp(ftyp,'PLATE')
    B_f = n_esize/2;
    L_f = n_esize/2;
else
    B_f = 0.75;
    L_f = 0.75;
end
%% Importing Data
bf_nm_u = 'fftd_%d_%s_%s_%s';
bf_nm_v = 'fftv_%d_%s_%s_%s';
cols = {'Freq', 'Re','Im','Amp'};
s_dir = [1 2 3];
n_snr = numel(s_dir);
%%
name_evnt='Poing'
%%
if strcmp(name_evnt, 'Poing')
    evnt='Po2016';
    stn='POI01';
    date='2016_12_20';
    time='03_30_51';
    nzero=6;
    ff_fldr = fullfile('GM','GM_POI2016',stn);
elseif strcmp(name_evnt, 'Unterhaching')
    evnt='Part1';
    stn='UH1';
    date='2013_04_16';
    time='21_51_42';
    nzero=4;
    fldr_nm = [stn, '_', evnt];
    ff_fldr = fullfile('GM', 'GM_UH',fldr_nm);
end

[f_inpt,ff_Uamp_mat,ff_Ur_mat,ff_UIm_mat,ff_Ucmplx_mat]=...
    fns_imprtdata.get_ff_inpt(bf_nm_u,s_dir,...
    stn,date, time,n_snr,ff_fldr,cols);
[f_inpt_V,ff_Vamp_mat,ff_Vr_mat,ff_VIm_mat,ff_Vcmplx_mat]=...
    fns_imprtdata.get_ff_inpt(bf_nm_v,s_dir,...
    stn,date, time,n_snr,ff_fldr,cols);
% fns_plot.plt_ff(f_inpt_V, ff_Vamp_mat,bf_nm_v,123,...
%     stn,date, time,'Velocity~(m/s)','initial')
%% Importing Transfer Function
rf_fldr = 'MultiUnitBld_GeomVary';
bf_nm = 'Disp_Center_%s_%d_l%d_b%d';
cols1 = {'Freq', 'AMPL','PHASE','REAL','IMAG'};
cmpt = {'X', 'Y', 'Z'};
%%
n_c = length(cmpt);
Vabs_zCell = cell(1, n_str+1);
fref_cmplx_mat_1=cell(1, n_c);
DISP_cmplx_mat=cell(1, n_c);
VEL_abs_mat=cell(1, n_c);
fref_vel_amp_mat_1=cell(1, n_c);
v_ref=5e-8;
for i_str = 0:n_str
    [f_vect,TF_amp_mat,TF_cpmlx_mat,lb_combs]=...
        fns_scatter.get_TF_scatter(n_str, n_rx, n_ry,...
        l_vect, b_vect, ftyp, V_s, L_f, B_f,...
        bf_nm,i_str,cmpt,n_c,rf_fldr,cols1);
    for i_c = 1:n_c
        %% Calculating Velocity
        TFcpmlx_intrp{i_c}=interp1(f_vect,TF_cpmlx_mat{i_c},...
            f_inpt{i_c},'linear','extrap');
        Ucmplx_mat{i_c}=2*ff_Ucmplx_mat{i_c}.*TFcpmlx_intrp{i_c};

        Vabs_mat{i_c}=abs(Ucmplx_mat{i_c}.*1i.*f_inpt{i_c}*2*pi);
        Vcmplx_mat{i_c}=(Ucmplx_mat{i_c}.*1i.*f_inpt{i_c}*2*pi);
        Vdb_mat{i_c}=20*log10(Vabs_mat{i_c}(nzero:end,:)./v_ref);
    end
    %%
    Vdb_zCell{i_str+1} = Vdb_mat{3};
    Vdb_xCell{i_str+1} = Vdb_mat{1};
    Vdb_yCell{i_str+1} = Vdb_mat{2};
    f_dBz=f_inpt_V{3}(nzero:end);
    f_dBx=f_inpt_V{1}(nzero:end);
end

filename_x = sprintf('Vdb_Cell_Vs_%d_%d_stn%s_cmp%s',n_str, V_s, stn, 'X');
filename_y = sprintf('Vdb_Cell_Vs_%d_%d_stn%s_cmp%s', V_s, stn, 'Y');
filename_z = sprintf('Vdb_Cell_Vs_%d_%d_stn%s_cmp%s', V_s, stn, 'Z');

%%
ylblvectz = {'$v_{z}$,~dB'; '[ref: $5 \times 10^{-8}$ m/s]'};
ylblvectx = {'$v_{x}$,~dB'; '[ref: $5 \times 10^{-8}$ m/s]'};
ylblvect = {'PPV,~dB'; '[ref: $5 \times 10^{-8}$ m/s]'};

dfz=f_dBz(3)-f_dBz(2);
dfx=f_dBx(3)-f_dBx(2);
num_lb=length(lb_combs);
for i_flur = 1:n_str+1
%     [f,v,v_db]=fns_unitgeomdb.DIN4150_3_lims(v_ref,i_flur);
    Vdb_Zmat = Vdb_zCell{i_flur};
    Vdb_Xmat = Vdb_xCell{i_flur};
    Vdb_Ymat = Vdb_yCell{i_flur};
    min_l = min([length(Vdb_Xmat), length(Vdb_Ymat), length(Vdb_Zmat)]);
    Vdb_mat=(Vdb_Zmat(1:min_l,:).^2+Vdb_Xmat(1:min_l,:).^2+Vdb_Ymat(1:min_l,:).^2).^0.5;
    for j=1:num_lb
        Vdb_vect=Vdb_mat(:,j);
        [Fun_rms_vect, f_cenVect] = fns_Octve.get_octBand(...
            Vdb_vect, f_dBz, dfz);
        V_rms_mat(:, j)=Fun_rms_vect;

%         Vdb_Zvect=Vdb_Zmat(:,j);
%         [Fun_rms_vectz, f_cenVectz] = fns_Octve.get_octBand(...
%             Vdb_Zvect, f_dBz, dfz);
%         V_rms_Zmat(:, j)=Fun_rms_vectz;
        %         Vdb_Xvect=Vdb_Xmat(:,j);
        %         [Fun_rms_vectx, f_cenVectx] = fns_Octve.get_octBand(...
        %             Vdb_Xvect, f_dBx, dfx);
        %         V_rms_Xmat(:, j)=Fun_rms_vectx;
    end
    %     fns_unitgeomdb.plt_Vrms(f_cenVect,V_rms_Zmat,f_iso,i_flur,V_s,ylblvect{i_flur})
    %     fns_unitgeomdb.plt_Vrms_stats(f_cenVectz,V_rms_Zmat,i_flur,V_s,ylblvectz,'Z',stn,v_db,f);
    V_rms_mean=fns_unitgeomdb.plt_Vrms_stats(f_cenVect,V_rms_mat,i_flur,V_s,ylblvect,'PPV',stn,n_str);
    %     fns_unitgeomdb.plt_Vrms_stats(f_cenVectx,V_rms_Xmat,i_flur,V_s,ylblvectx,'X',stn,v_db,f)
    %% plot velocity results in db for individual direction for linear f scale
    %     Vzdb_mean = mean(Vdb_Zmat, 2);
    %     Vzdb_std = std(Vdb_Zmat, 0, 2);
    %     vz_mean = Vzdb_mean;
    %     vz_SDlow = Vzdb_mean - Vzdb_std;
    %     vz_SDup = Vzdb_mean + Vzdb_std;
    %     %     fns_unitgeomdb.plt_Vdb_MeanSD(f_inptz,vz_mean,vz_SDlow,vz_SDup,...
    %     %         0,100,-20,120,i_flur,V_s,rf_fldr,'Z','$v_z$ (dB; ref: 1nm/s)')
    %     Vdb_Xmat = Vdb_xCell{i_flur};
    %     Vxdb_mean = mean(Vdb_Xmat, 2);
    %     Vxdb_std = std(Vdb_Xmat, 0, 2);
    %     vx_mean = Vxdb_mean;
    %     vx_SDlow = Vxdb_mean - Vxdb_std;
    %     vx_SDup = Vxdb_mean + Vxdb_std;
    %     fns_unitgeomdb.plt_Vdb_MeanSD(f_inptx,vx_mean,vx_SDlow,vx_SDup,...
    %         0,100,-20,120,i_flur,V_s,rf_fldr,'X','$v_x$ (dB; ref: 1nm/s)')
end

