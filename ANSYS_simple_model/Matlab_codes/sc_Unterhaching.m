clear;clc;close all
%% Importing Data
% evnt={'UH_part1', 'UH_part2'};
evnt='Part1'
rf_fldr = 'input_Data';
stn_vect={'UH1', 'UH2', 'UH3'};
date='2013_04_16';
time='21_51_42';

n_stns=length(stn_vect);

bf_nm_u = 'fftd_%d_%s_%s_%s';
bf_nm_v = 'fftv_%d_%s_%s_%s';
bf_nm_a = 'ffta_%d_%s_%s_%s';
cols = {'Freq', 'Re','Im','Amp'};
s_dir = [1 2 3];
n_snr = numel(s_dir);
vlbl_vect={'$v_x$,~m/s' '$v_y$,~m/s' '$v_z$,~m/s'};
ulbl_vect={'$u_x$,~m' '$u_y$,~m' '$u_z$,~m'};
albl_vect={'$a_x$ (m)' '$a_y$ (m)' '$a_z$ (m)'};
figure
for i=1:n_stns
    stn=stn_vect{i};
    fldr_nm = [stn, '_', evnt];
    ff_fldr = fullfile('GM', 'GM_UH',fldr_nm);

    [f_inpt_V,ff_Vamp_mat,ff_Vr_mat,ff_VIm_mat,ff_Vcmplx_mat]=...
        fns_imprtdata.get_ff_inpt(bf_nm_v,s_dir,...
        stn,date, time,n_snr,ff_fldr,cols);
    fns_plot.plt_ff_svrlstns(f_inpt_V, ff_Vamp_mat,bf_nm_v,123,...
        stn,date, time,vlbl_vect,{'Frequency,~Hz'},'initial',evnt)
if stn=="UH1"
        v_x_UH1=ff_Vamp_mat{1};
        v_y_UH1=ff_Vamp_mat{2};
        v_z_UH1=ff_Vamp_mat{2};
        f=f_inpt_V{1};
    end
end
Vinpt_PPV=(v_x_UH1.^2+v_y_UH1.^2+v_z_UH1.^2).^0.5;
max_Vinpt_PPV=max(Vinpt_PPV)
[maxrow_indices, ~] = find(bsxfun(@eq, Vinpt_PPV, max_Vinpt_PPV));

f_idx=f(maxrow_indices)
figure
plot(f,Vinpt_PPV)
% figure
% for i=1:n_stns
%     stn=stn_vect{i};
%     fldr_nm = [stn, '_', evnt];
%     ff_fldr = fullfile('GM', 'GM_UH',fldr_nm);
%     [f_inpt_U,ff_Uamp_mat,ff_Ur_mat,ff_UIm_mat,ff_Ucmplx_mat]=...
%         fns_imprtdata.get_ff_inpt(bf_nm_u,s_dir,...
%         stn,date, time,n_snr,ff_fldr,cols);
%     fns_plot.plt_ff_svrlstns(f_inpt_U, ff_Uamp_mat,bf_nm_u,123,...
%         stn,date, time,ulbl_vect,{'f (Hz)'},'initial',evnt)
% end

% figure
% for i=1:n_stns
%     stn=stn_vect{i};
%     fldr_nm = [stn, '_', evnt];
%     ff_fldr = fullfile('GM', 'GM_UH',fldr_nm);
%     [f_inpt_A,ff_Aamp_mat,ff_Ar_mat,ff_AIm_mat,ff_Acmplx_mat]=...
%         fns_imprtdata.get_ff_inpt(bf_nm_a,s_dir,...
%         stn,date, time,n_snr,ff_fldr,cols);
%     fns_plot.plt_ff_svrlstns(f_inpt_A, ff_Aamp_mat,bf_nm_a,123,...
%         stn,date, time,albl_vect,{'f (Hz)'},'initial',evnt)
% end
%% Time domain
% Fs=200;    %sampling rate
% t_vect=0:0.005:3.995;
% nfft = 2^nextpow2(length(t_vect));
% for i=1:n_stns
%     stn=stn_vect{i};
%     fldr_nm = [stn, '_', evnt];
%     ff_fldr = fullfile('GM', 'GM_UH',fldr_nm);
%     [f_inpt_V,ff_Vamp_mat,ff_Vr_mat,ff_VIm_mat,ff_Vcmplx_mat]=...
%         fns_imprtdata.get_ff_inpt(bf_nm_v,s_dir,...
%         stn,date, time,n_snr,ff_fldr,cols);
%     y_lbl='Velocity~(m/s)';
%     [vt_xyz]=fns_tdomn_eval.get_tdmain_rslt(ff_Vr_mat,ff_VIm_mat,...
%         n_snr,Fs,nfft,t_vect,y_lbl,rf_fldr);
%     vt_xyz = cell2mat(vt_xyz');
%     fns_data_process.save_data_time('v', stn, date, time,ff_fldr,t_vect,vt_xyz);
%     %%
%     y_lbl='Displacement~(m)';
%     [f_inpt_U,ff_Uamp_mat,ff_Ur_mat,ff_UIm_mat,ff_Ucmplx_mat]=...
%         fns_imprtdata.get_ff_inpt(bf_nm_u,s_dir,...
%         stn,date, time,n_snr,ff_fldr,cols);
%     [ut_xyz]=fns_tdomn_eval.get_tdmain_rslt(ff_Ur_mat,ff_UIm_mat,...
%         n_snr,Fs,nfft,t_vect,y_lbl,rf_fldr);
%     ut_xyz = cell2mat(ut_xyz');
%     fns_data_process.save_data_time('d', stn, date, time,ff_fldr,t_vect,ut_xyz);
%     %%
%     [f_inpt_A,ff_Aamp_mat,ff_Ar_mat,ff_AIm_mat,ff_Acmplx_mat]=...
%         fns_imprtdata.get_ff_inpt(bf_nm_a,s_dir,...
%         stn,date, time,n_snr,ff_fldr,cols);
%     y_lbl='Acceleration~(m/s^2)';
%     [at_xyz]=fns_tdomn_eval.get_tdmain_rslt(ff_Ar_mat,ff_AIm_mat,...
%         n_snr,Fs,nfft,t_vect,y_lbl,rf_fldr);
%     at_xyz = cell2mat(at_xyz');
%     fns_data_process.save_data_time('a', stn, date, time,ff_fldr,t_vect,at_xyz);
% end

%%
bf_nm_ut = 'u_%d_%s_%s_%s';
bf_nm_vt = 'v_%d_%s_%s_%s';
cols_t = {'tim', 'val'};

figure
for i=1:n_stns
    stn=stn_vect{i};
    fldr_nm = [stn, '_', evnt];
    ff_fldr = fullfile('GM', 'GM_UH',fldr_nm);

    [t_in,ff_Vt]=...
        fns_imprtdata.get_ff_tim(bf_nm_vt,s_dir,...
        stn,date, time,n_snr,ff_fldr,cols_t);
    fns_plot.plt_ff_svrlstns(t_in, ff_Vt,bf_nm_vt,123,...
        stn,date, time,vlbl_vect,{'t (s)'},'initial',evnt)
end
% 
% figure
% for i=1:n_stns
%     stn=stn_vect{i};
%     fldr_nm = [stn, '_', evnt];
%     ff_fldr = fullfile('GM', 'GM_UH',fldr_nm);
% 
%     [t_in,ff_Ut]=...
%         fns_imprtdata.get_ff_tim(bf_nm_ut,s_dir,...
%         stn,date, time,n_snr,ff_fldr,cols_t);
%     fns_plot.plt_ff_svrlstns(t_in, ff_Ut,bf_nm_ut,123,...
%         stn,date, time,ulbl_vect,{'t (s)'},'initial',evnt)
% end