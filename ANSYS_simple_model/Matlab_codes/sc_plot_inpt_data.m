clear;clc;close all
%% Mexican Hat
% Parameters
center_freq = 4; % Center frequency in Hz
t_resoltn = 0.001; % Time resolution (sampling interval) in seconds
T = 10; % Time window in seconds

% Time vector
t = (0:t_resoltn:T);

% Generate the Mexican hat wavelet using the mexihat function
wavelet = fns_mexihat.def_mexihat(t, center_freq);
% Compute the frequency axis for the Fourier transform
N = length(t);
Fs = 1 / t_resoltn;
freq = (0:N-1) * Fs / N;
wavelet_cell = cell(1, 3);
wavelet_fft_ss_cell = cell(1, 3);

for row = 1:3
    %     currentMatrix = sig_t_xyz{row};
    %     peakAmplitudesVec = max(abs(currentMatrix));
    peakAmplitudesVec =5e-7
    current_peak_amp = max(abs(wavelet));
    scale_factor = peakAmplitudesVec / current_peak_amp;
    wavelet = wavelet * scale_factor;
    wavelet_cell{row} = wavelet;
    % Compute the Fourier transform of the wavelet
    wavelet_fft = fft(wavelet);

    % Single-sided spectrum
    wavelet_fft_ss = wavelet_fft(1:N/2+1);
    wavelet_fft_ss(2:end-1) = 2 * wavelet_fft_ss(2:end-1);
    wavelet_fft_ss_cell{row} = wavelet_fft_ss;

end
% Frequency vector for the single-sided spectrum
freq_ss = freq(1:N/2+1);
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
        stn,date, time,'Velocity~(m/s)','initial')
    fns_mexihat.plt_ff_svrlstns_mhat(f_inpt_V, ff_Vamp_mat,...
        stn,'Velocity~(m/s)',freq_ss,wavelet_fft_ss_cell)
end

% figure
% for i=1:n_stns
%     stn=stn_vect{i};
%     fldr_nm = ['GM_', stn, '_', date, '_', time];
%     ff_fldr = fullfile('GM', 'GM_Poing_Sync', ['GM_', date, '_', time],...
%         fldr_nm);
%     [f_inpt_U,ff_Uamp_mat,ff_Ur_mat,ff_UIm_mat,ff_Ucmplx_mat]=...
%         fns_imprtdata.get_ff_inpt(bf_nm_u,s_dir,...
%         stn,date, time,n_snr,ff_fldr,cols);
%     fns_plot.plt_ff_svrlstns(f_inpt_U, ff_Uamp_mat,bf_nm_u,123,...
%         stn,date, time,'Displacement~(m)','initial')
% end
%% Time domain
Fs=200;    %sampling rate
t_vect=0:0.005:3.995;
nfft = 2^nextpow2(length(t_vect));
y_lbl='Velocity~(m/s)';
fns_tdomn_eval.get_tdmain_rslt(ff_Vr_mat,ff_VIm_mat,...
    n_snr,Fs,nfft,t_vect,y_lbl,rf_fldr);
y_lbl='Displacement~(m)';
[sig_t_xyz]=fns_tdomn_eval.get_tdmain_rslt(ff_Ur_mat,ff_UIm_mat,...
    n_snr,Fs,nfft,t_vect,y_lbl,rf_fldr);
%%

%Plot the Mexican hat wavelet in time domain
figure;
subplot(2, 1, 1);
plot(t, wavelet_cell{1}, 'r', 'LineWidth', 1.5);
hold on
plot(t, wavelet_cell{2}, 'g', 'LineWidth', 1.5);
hold on
plot(t, wavelet_cell{3}, 'b', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Amplitude');
xlim([0,1])
title('Mexican Hat Wavelet in Time Domain');
grid on;

% Plot the Mexican hat wavelet in frequency domain (single-sided)
subplot(2, 1, 2);
plot(freq_ss, abs(wavelet_fft_ss_cell{1}), 'r', 'LineWidth', 1.5);
hold on
plot(freq_ss, abs(wavelet_fft_ss_cell{2}), 'g', 'LineWidth', 1.5);
hold on
plot(freq_ss, abs(wavelet_fft_ss_cell{3}), 'b', 'LineWidth', 1.5);
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title('Mexican Hat Wavelet in Frequency Domain (Single-Sided)');
grid on;

fns_mexihat.plt_ff_mexihat(f_inpt_U, ff_Uamp_mat,bf_nm_u,123,...
    stn,date, time,'Displacement~(m)','initial',freq_ss,wavelet_fft_ss_cell)
