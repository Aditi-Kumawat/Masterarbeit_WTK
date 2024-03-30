clear;clc;close all
%% Importing Data
% evnt={'Po2016', 'Po2017'};
evnt='Po2016'
rf_fldr = 'input_Data';
if evnt == "Po2017"
    stn_vect={'HWMRS', 'LP01S', 'MS1', 'Poing', 'RHS26',...
        'SCH6S', 'SIS21', 'WS15S'};
    date='2017_09_09';
    time='17_20_29';
elseif evnt == "Po2016"
    stn_vect={'POI01', 'POI02', 'POI03'};
    date='2016_12_20';
    time='03_30_51';
end

n_stns=length(stn_vect);

bf_nm_u = 'fftd_%d_%s_%s_%s';
bf_nm_v = 'fftv_%d_%s_%s_%s';
cols = {'Freq', 'Re','Im','Amp'};
s_dir = [1 2 3];
n_snr = numel(s_dir);
vlbl_vect={'$v_x$ (m/s)' '$v_y$ (m/s)' '$v_z$ (m/s)'};
ulbl_vect={'$u_x$ (m)' '$u_y$ (m)' '$u_z$ (m)'};

figure
for i=1:n_stns
    stn=stn_vect{i};
    fldr_nm = ['GM_', stn, '_', date, '_', time];
    ff_fldr = fullfile('GM', 'GM_Poing_Sync', ['GM_', date, '_', time],...
        fldr_nm);
    [f_inpt_V,ff_Vamp_mat,ff_Vr_mat,ff_VIm_mat,ff_Vcmplx_mat]=...
        fns_imprtdata.get_ff_inpt(bf_nm_v,s_dir,...
        stn,date, time,n_snr,ff_fldr,cols);
    fns_plot.plt_ff_svrlstns(f_inpt_V, ff_Vamp_mat,bf_nm_v,123,...
        stn,date, time,vlbl_vect,{'f (Hz)'},'initial',evnt)
end

figure
for i=1:n_stns
    stn=stn_vect{i};
    fldr_nm = ['GM_', stn, '_', date, '_', time];
    ff_fldr = fullfile('GM', 'GM_Poing_Sync', ['GM_', date, '_', time],...
        fldr_nm);
    [f_inpt_U,ff_Uamp_mat,ff_Ur_mat,ff_UIm_mat,ff_Ucmplx_mat]=...
        fns_imprtdata.get_ff_inpt(bf_nm_u,s_dir,...
        stn,date, time,n_snr,ff_fldr,cols);
    fns_plot.plt_ff_svrlstns(f_inpt_U, ff_Uamp_mat,bf_nm_u,123,...
        stn,date, time,ulbl_vect,{'f (Hz)'},'initial',evnt)
end
%%
bf_nm_ut = 'd_%d_%s_%s_%s';
bf_nm_vt = 'v_%d_%s_%s_%s';
cols_t = {'tim', 'val'};

figure
for i=1:n_stns
    stn=stn_vect{i};
    fldr_nm = ['GM_', stn, '_', date, '_', time];
    ff_fldr = fullfile('GM', 'GM_Poing_Sync', ['GM_', date, '_', time],...
        fldr_nm);
    [t_in,ff_Vt]=...
        fns_imprtdata.get_ff_tim(bf_nm_vt,s_dir,...
        stn,date, time,n_snr,ff_fldr,cols_t);
    fns_plot.plt_ff_svrlstns(t_in, ff_Vt,bf_nm_vt,123,...
        stn,date, time,vlbl_vect,{'t (s)'},'initial',evnt)
end

figure
for i=1:n_stns
    stn=stn_vect{i};
    fldr_nm = ['GM_', stn, '_', date, '_', time];
    ff_fldr = fullfile('GM', 'GM_Poing_Sync', ['GM_', date, '_', time],...
        fldr_nm);
    [t_in,ff_Ut]=...
        fns_imprtdata.get_ff_tim(bf_nm_ut,s_dir,...
        stn,date, time,n_snr,ff_fldr,cols_t);
    fns_plot.plt_ff_svrlstns(t_in, ff_Ut,bf_nm_ut,123,...
        stn,date, time,ulbl_vect,{'t (s)'},'initial',evnt)
end