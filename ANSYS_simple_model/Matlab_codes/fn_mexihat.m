function wavelet = fn_mexihat(t, center_freq)
t = t - 0.5;
% Parameters for the Mexican hat wavelet (Ricker wavelet)
sigma = 1 / (2 * pi * center_freq); % Standard deviation to control the width of the wavelet

% Calculate the wavelet
wavelet = (2 / (sqrt(3 * sigma) * (pi^(1/4)))) * (1 - (t / sigma).^2) .* exp(-(t.^2) / (2 * sigma^2));
end

