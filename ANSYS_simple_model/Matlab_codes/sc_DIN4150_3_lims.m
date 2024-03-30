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
%%
name_evnt='Unterhaching'
%%
if strcmp(name_evnt, 'Poing')
    evnt='Po2016';
    stn_vect={'POI01', 'POI02', 'POI03'};
    date='2016_12_20';
    time='03_30_51';
elseif strcmp(name_evnt, 'Unterhaching')
    evnt='Part1';
    stn_vect={'UH1', 'UH2', 'UH3'};
    date='2013_04_16';
    time='21_51_42';
end

n_stns=length(stn_vect);

%% Importing Data
bf_nm_u = 'fftd_%d_%s_%s_%s';
bf_nm_v = 'fftv_%d_%s_%s_%s';
nzero=6;
cols = {'Freq', 'Re','Im','Amp'};
s_dir = [1 2 3];
n_snr = numel(s_dir);
%% Importing Transfer Function
rf_fldr = 'UnitBld_GeomVary';
bf_nm = 'Disp_Center_%s_%d_l%d_b%d';
cols1 = {'Freq', 'AMPL','PHASE','REAL','IMAG'};
cmpt = {'X', 'Y', 'Z'};

%%
bf_nm_vt = 'v_%d_%s_%s_%s';
cols_t = {'tim', 'val'};

for i_stn=1:n_stns
    stn=stn_vect{i_stn};
    display(stn)
    if strcmp(name_evnt, 'Poing')
        ff_fldr = fullfile('GM','GM_POI2016',stn);
    elseif strcmp(name_evnt, 'Unterhaching')
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
    %%

    [t_in,ff_Vt]=...
        fns_imprtdata.get_ff_tim(bf_nm_vt,s_dir,...
        stn,date, time,n_snr,ff_fldr,cols_t);
    t_in=t_in{1};

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
            Vss_mat{i_c}=20*log10(Vabs_mat{i_c}(nzero:end,:)./v_ref);
        end
        %%
        Vss_zCell{i_str+1} = Vcmplx_mat{3};
        Vss_xCell{i_str+1} = Vcmplx_mat{1};
        Vss_yCell{i_str+1} = Vcmplx_mat{2};
        freq=f_inpt_V{3}(nzero:end);
    end
    %%
    Fs = 200;
    nfft = 2^nextpow2(length(t_in));
    freq = Fs / 2 * linspace(0, 1, nfft/2+1);
    dfz=freq(3)-freq(2);
    num_lb=length(lb_combs);
    for i_flur = 1:2
        Vss_Zmat = Vss_zCell{i_flur};
        Vss_Xmat = Vss_xCell{i_flur};
        Vss_Ymat = Vss_yCell{i_flur};
        Vzss_fft_pad = [Vss_Zmat; conj(flipud(Vss_Zmat(2:end-1,:)))];
        Vxss_fft_pad = [Vss_Xmat; conj(flipud(Vss_Xmat(2:end-1,:)))];
        Vyss_fft_pad = [Vss_Ymat; conj(flipud(Vss_Ymat(2:end-1,:)))];
        Vz_ifft = 0.5*ifft(Vzss_fft_pad*Fs, nfft, 1, 'symmetric');
        Vx_ifft = 0.5*ifft(Vxss_fft_pad*Fs, nfft, 1, 'symmetric');
        Vy_ifft = 0.5*ifft(Vyss_fft_pad*Fs, nfft, 1, 'symmetric');
        Vz_ifft = Vz_ifft(1:length(t_in),:);
        Vx_ifft = Vx_ifft(1:length(t_in),:);
        Vy_ifft = Vy_ifft(1:length(t_in),:);
        t = (0:length(Vz_ifft)-1) / Fs;
        max_Vxyzmat=[max(Vx_ifft); max(Vy_ifft); max(Vz_ifft)];
        max_Vxyz=max(max_Vxyzmat);
        max_Vxyz_1(i_stn,:,i_flur)=max_Vxyz;
        [maxrow_indices, ~] = find(bsxfun(@eq, max_Vxyzmat, max_Vxyz));
        %         f_max=zeros(1,num_lb);
        for ilb=1:num_lb
            max_rowin=maxrow_indices(ilb);
            if max_rowin==1
                Vss_Xvect=Vss_Xmat(:,ilb);
                [value, index] = max(Vss_Xvect);
                f_max(i_stn,ilb,i_flur)=freq(index);
            elseif max_rowin==2
                Vss_Yvect=Vss_Ymat(:,ilb);
                [value, index] = max(Vss_Yvect);
                f_max(i_stn,ilb,i_flur)=freq(index);
            elseif max_rowin==3
                Vss_Zvect=Vss_Zmat(:,ilb);
                [value, index] = max(Vss_Zvect);
                f_max(i_stn,ilb,i_flur)=freq(index);
            end
        end
    end
end
fns_unitgeomdb.plot_DIN4150_3(v_ref,2,f_max,max_Vxyz_1,V_s,n_stns,stn_vect,name_evnt)
