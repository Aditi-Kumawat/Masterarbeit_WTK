clear;clc;close all
%% Importing Data
% evnt={'Po2016', 'Po2017'};
evnt='Part1'
rf_fldr = 'input_Data';
stn_vect={'UH1', 'UH2', 'UH3'};
date='2013_04_16';
time_evnt='21_51_42';


n_stns=length(stn_vect);


s_dir = [1 2 3];
n_snr = numel(s_dir);
vlbl_vect={'$v_x$ (m/s)' '$v_y$ (m/s)' '$v_z$ (m/s)'};
ulbl_vect={'$u_x$ (m)' '$u_y$ (m)' '$u_z$ (m)'};
%

%%
bf_nm_ut = 'd_%d_%s_%s_%s';
bf_nm_vt = 'v_%d_%s_%s_%s';
cols_t = {'tim', 'val'};

% figure
% for i=1:n_stns
%     stn=stn_vect{i};
%     fldr_nm = [stn, '_', evnt];
%     ff_fldr = fullfile('GM', 'GM_UH',fldr_nm);
%     [t_in,ff_Vt]=...
%         fns_imprtdata.get_ff_tim(bf_nm_vt,s_dir,...
%         stn,date, time_evnt,n_snr,ff_fldr,cols_t);
%     fns_plot.plt_ff_svrlstns(t_in, ff_Vt,bf_nm_vt,123,...
%         stn,date, time_evnt,vlbl_vect,{'t (s)'},'initial',evnt)
% end

figure;
for i = 1:n_stns
    stn = stn_vect{i};
    fldr_nm = [stn, '_', evnt];
    ff_fldr = fullfile('GM', 'GM_UH',fldr_nm);
    [t_in, Ut] = fns_imprtdata.get_ff_tim(bf_nm_ut, s_dir, stn, date, time_evnt, n_snr, ff_fldr, cols_t);

    for j=1:n_snr
        t_cut=   t_in{j,:};
        mask = (t_cut >= 115) & (t_cut <= 130);
        t_cut = t_cut(mask) - 115;
        t_in{j,:}=t_cut;
        Ut_cut = Ut{:,j}.';
        Ut_cut=Ut_cut(:,mask);
        Utin(:,j)=Ut_cut.';

    end
    Fs = 200;
    nfft = 2^nextpow2(length(t_cut));
    freq = Fs / 2 * linspace(0, 1, nfft/2+1);
    u_fft = fft(Utin, nfft) * (1/Fs);
    u_fft_ss = 2 * u_fft(1:nfft/2+1,:);
    f_matrix = repmat(freq', 1, 3);
    %     u_fft_ss=v_fft_ss./1i./(2*pi*f_matrix);

    u_fft_pad = [u_fft_ss; conj(flipud(u_fft_ss(2:end-1,:)))];
    u_ifft = (0.5)*ifft(u_fft_pad*Fs, nfft, 1, 'symmetric');
    u_ifft = u_ifft(1:length(t_cut),:); % Truncate to same length as v
    t = (0:length(u_ifft)-1) / Fs;
    %% Plotting
    leg_vect={'X','Y','Z'};

    x_l_f='Frequency~(Hz)';
    y_l_f='u(f)';
    x_l_t='Time~(s)';
    y_l_t='u(t)';
    plot_tiles.plot_tiles_set_2(2,3,u_ifft,abs(u_fft_ss),t,freq,x_l_f,y_l_f,x_l_t,y_l_t,leg_vect)

    %     fns_plot.plt_ff_svrlstns(t_in, Utin, bf_nm_ut, 123, stn, date, time_evnt, ulbl_vect, {'t (s)'}, 'initial', evnt);
end

