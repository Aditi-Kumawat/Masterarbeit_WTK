% Parameters
center_freq = 10; % Center frequency in Hz
t_resoltn = 0.001; % Time resolution (sampling interval) in seconds
T = 10; % Time window in seconds
peak_amp = 1e-5;

% Time vector
t = (-T:t_resoltn:T);

% Generate the Mexican hat wavelet using the mexihat function
wavelet = fn_mexihat(t, center_freq);

% Scale the wavelet to have the desired peak amplitude
current_peak_amp = max(abs(wavelet));
scale_factor = peak_amp / current_peak_amp;
wavelet = wavelet * scale_factor;

% Compute the frequency axis for the Fourier transform
N = length(t);
Fs = 1 / t_resoltn;
freq = (0:N-1) * Fs / N;

% Compute the Fourier transform of the wavelet
wavelet_fft = fft(wavelet);

% Single-sided spectrum
wavelet_fft_ss = wavelet_fft(1:N/2+1);
wavelet_fft_ss(2:end-1) = 2 * wavelet_fft_ss(2:end-1);

% Frequency vector for the single-sided spectrum
freq_ss = freq(1:N/2+1);

% Plot the Mexican hat wavelet in time domain
figure;
subplot(2, 1, 1);
plot(t, wavelet, 'b', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Amplitude');
title('Mexican Hat Wavelet in Time Domain');
grid on;

% Plot the Mexican hat wavelet in frequency domain (single-sided)
subplot(2, 1, 2);
plot(freq_ss, imag(wavelet_fft_ss), 'r', 'LineWidth', 1.5);
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title('Mexican Hat Wavelet in Frequency Domain (Single-Sided)');
grid on;
xlim([0, 50]); % Set the frequency axis limit for better visualization