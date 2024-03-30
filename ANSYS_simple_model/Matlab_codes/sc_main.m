clear;clc;
close all
% Define the number of storeys, rooms in x-direction, and y-direction
n_str = 2;
n_rx = 2;
n_ry = 3;

% Define the length, width, and height of the building
l_vect=[3 4 5 6];
b_vect=[3 4 5 6];
h = 3;

% Define the type of foundation as either 'PLATE' or 'FOOTING'
% ftyp = 'PLATE';
ftyp = 'FOOTING';

% Define the foundation behaviour and analysis type
rec=936;

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
%% Importing Transfer Function

% Define the name of the folder where the results are stored
% rf_fldr = 'Results_Freq_Indepn_Inpt/SeisSol/data_Vs30_Bld';
% rf_fldr = 'Results_Freq_Indepn_TF';
rf_fldr = 'MultiUnitBld_GeomVary';
% Define the base file name for the U center results in ANSYS
bf_nm = 'Disp_Center_%s_%d_l%d_b%d';

% Define the column names in the results table
cols1 = {'Freq', 'AMPL','PHASE','REAL','IMAG'};

% Define the components to extract from the results
cmpt = {'X', 'Y', 'Z'};
n_c = length(cmpt);
%% Importing Data
ff_fldr = 'GM/SeisSol/data_Vs30_Bld';
% Define the base file name for the U center results in ANSYS
bf_nm_u = 'fftd_%d_%s_%d';
bf_nm_v = 'fftv_%d_%s_%d';
stn='rec';
% Define the column names in the results table
cols = {'Freq', 'Re','Im','Amp'};

% Define the components to extract from the results
s_dir = [1 2 3];
% length of free field data for each sensor is different
n_snr = numel(s_dir);

[f_inpt_U,ff_Uamp_mat,ff_Ur_mat,ff_UIm_mat,ff_Ucmplx_mat]=...
    fns_imprtdata.get_inpt_seisl(bf_nm_u,s_dir,...
    stn,rec,n_snr,ff_fldr,cols);
[f_inpt_V,ff_Vamp_mat,ff_Vr_mat,ff_VIm_mat,ff_Vcmplx_mat]=...
    fns_imprtdata.get_inpt_seisl(bf_nm_v,s_dir,...
    stn,rec,n_snr,ff_fldr,cols);
%%
ss_fldr = 'GM/SeisSol/data_Vs30_Bld';
rec_b_vect=[1324 1325 1326];
[bldUampmat,bldUcmplx_mat]=fns_imprtdata.get_bldata_seisl(bf_nm_u,...
    stn,rec_b_vect,s_dir,ss_fldr,cols);
[bldVampmat,bldVcmplx_mat]=fns_imprtdata.get_bldata_seisl(bf_nm_v,...
    stn,rec_b_vect,s_dir,ss_fldr,cols);

% Plotting Data
date='936';
time='936';
% fns_plot.plt_ff(f_inpt_V, ff_Vamp_mat,bf_nm_v,123,...
%     stn,date, time,'Velocity~(m/s)','initial')
% fns_plot.plt_ff(f_inpt_U, ff_Uamp_mat,bf_nm_u,123,...
%     stn,date, time,'Displacement~(m)','initial')
%% % Time domain
Fs=200;    %sampling rate
t_vect=0:0.005:3.995;
nfft = 2^nextpow2(length(t_vect));
y_lbl='Velocity~(m/s)';
fns_tdomn_eval.get_tdmain_rslt(ff_Vr_mat,ff_VIm_mat,...
    n_snr,Fs,nfft,t_vect,y_lbl,rf_fldr);
y_lbl='Displacement~(m)';
fns_tdomn_eval.get_tdmain_rslt(ff_Ur_mat,ff_UIm_mat,...
    n_snr,Fs,nfft,t_vect,y_lbl,rf_fldr);
%%

Vabs_xCell = cell(1, n_str+1);
Ucmplx_mat=cell(1, n_c);
Vabs_mat=cell(1, n_c);
Vcmplx_mat=cell(1, n_c);
y_lbl='Velocity~(m/s)';

for i_str = 0:n_str
    if strcmp(rf_fldr, 'Results_Freq_Indepn_TF')...
            ||...
            strcmp(rf_fldr, 'MultiUnitBld_GeomVary')
        [f_out,Uamp_mat,Ucpmlx_mat]=fns_imprtdata.get_TF(...
            n_str, n_rx, n_ry,...
            l_vect, b_vect, ftyp, V_s, L_f, B_f,...
            bf_nm,i_str,cmpt,n_c,rf_fldr,cols1);
        %         for i_c = 1:n_c
        %             % Plotting Transfer Function
        %             fns_plot.plt_TF(f_out,Uamp_mat{i_c},i_str, n_rx,...
        %                 n_ry, l_vect, b_vect, ftyp,...
        %                 V_s, L_f, B_f,i_c, cmpt,rf_fldr);
        %         end
    else
        [f_out,Uamp_mat,Ucpmlx_mat]=fns_imprtdata.get_U_rec(...
            n_str,n_rx, n_ry,l_vect, b_vect, ftyp, V_s, L_f,...
            B_f,rec,bf_nm,i_str,cmpt,n_c,rf_fldr,cols1);
    end
    figure
    for i_c = 1:n_c
        %% Calculating Velocity
        Ucpmlx_intrp{i_c}=interp1(f_out,Ucpmlx_mat{i_c},...
            f_inpt_U{i_c},'linear','extrap');
        % Ucpmlx_intrp{i_cmp}=Ucpmlx_mat{i_cmp};

        if strcmp(rf_fldr, 'Results_Freq_Indepn_TF')...
                ||...
                strcmp(rf_fldr, 'MultiUnitBld_GeomVary')
            Ucmplx_mat{i_c}=ff_Ucmplx_mat{i_c}.*Ucpmlx_intrp{i_c};
        else
            Ucmplx_mat{i_c}=Ucpmlx_intrp{i_c};
        end
        Vabs_mat{i_c}=abs(Ucmplx_mat{i_c}.*1i.*f_inpt_U{i_c}*2*pi);
        Vcmplx_mat{i_c}=(Ucmplx_mat{i_c}.*1i.*f_inpt_U{i_c}*2*pi);
        fns_saveSbplt.plt_Sbplt_SS_lb(f_inpt_U,Vcmplx_mat,i_c,...
            i_str,y_lbl,bldVampmat,l_vect,b_vect)
        %         fns_saveData.fn_saveUF(Ucmplx_mat,f_inpt_U,i_c,i_str,rf_fldr)
        %         fns_saveData.fn_saveVF(Vcmplx_mat,f_inpt_U,i_c,i_str,rf_fldr)

    end
    fil_nm = ['vel_cmpr_lb_',num2str(i_str),'.png'];
    fns_saveSbplt.saveFig(rf_fldr,fil_nm)
    %%
    Vabs_zCell{i_str+1} = Vabs_mat{3};
    ff_VzMat=ff_Vamp_mat{3};
    Vabs_xCell{i_str+1} = Vabs_mat{1};
    ff_VxMat=ff_Vamp_mat{1};
    f_in=f_inpt_V{3};
    %
    y_lbl='Displacement~(m)';
    [u_ifft_mat,t]=fns_tdomn_eval.get_tdmain_rslt_MAT(...
        Ucmplx_mat,bldUcmplx_mat,n_snr,Fs,nfft,t_vect,l_vect,...
        b_vect,i_str,ftyp,V_s,L_f,B_f,y_lbl,rf_fldr);
    y_lbl='Velocity~(m/s)';
    [v_ifft_mat,t]=fns_tdomn_eval.get_tdmain_rslt_MAT(...
        Vcmplx_mat,bldVcmplx_mat,n_snr,Fs,nfft,t_vect,l_vect,...
        b_vect,i_str,ftyp,V_s,L_f,B_f,y_lbl,rf_fldr);
    fname_1 = strcat(sprintf('vt_SeisSol_ifft_%d',i_str), '.txt');
    %     fns_saveData.fn_saveVifft(v_ifft_mat,t,rf_fldr,fname_1)

end
%% Calculating and Plotting Velocity in Octave Bands
% % This part of the code calculates the octave bands for the vertical
% % velocity levels (Z-components for each story of each building).
% % The octave bands are calculated using the get_vel_on_octave_scale
% % function in the funs_Octave_analysis module.
df=f_in(10)-f_in(9);
[Vratio_dbCell,f_linVect,Vratio_rms_cell,f_cenVect]=...
    fns_Octve.get_V_octave(Vabs_zCell,ff_VzMat,n_str,f_in,l_vect, df);
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
% %%
% [Vratio_dbCell,f_linVect,Vratio_rms_cell,f_cenVect]=...
%     fns_Octve.get_V_octave(Vabs_xCell,ff_VxMat,n_str,f_in,l_vect, df);
% [f_iso,n_octBands]=fns_Octve.get_fvect_iso(f_cenVect);
% ustr_vect={'20log10($v_{x0}/v_{ff}$)~(dB)',...
%     '20log10($v_{x1}/v_{x0}$)~(dB)',...
%     '20log10($v_{x2}/v_{x0}$)~(dB)'};
% for i_str = 0:n_str
%     Vratio_rms_mat=Vratio_rms_cell{i_str+1};
%     Vratio_db_mat=Vratio_dbCell{i_str+1};
%     if i_str==0
%         ylim([-2,4])
%     else
%         ylim([-5,25])
%     end
%     % Plotting velocity levels Octave Bands (in octave 1/3 bands)
%     fns_plot.plt_VampZ_Oct(f_cenVect, Vratio_rms_mat,...
%         i_str, n_rx, n_ry, l_vect, b_vect, ftyp,...
%         V_s, L_f, B_f,3,cmpt,f_iso,ustr_vect,y_lim)
%     %     % Plotting velocity levels in db over linear frequency range
%     %     fns_plot.plt_VampZ_db(f_linVect, vel_db_mat,...
%     %         i_str, n_rx, n_ry, l_vect, b_vect, ftyp,...
%     % V_s, L_f, B_f,3,cmpt)
%
% end