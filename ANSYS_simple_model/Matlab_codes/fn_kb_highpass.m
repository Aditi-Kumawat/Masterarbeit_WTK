%% compute KB value
%input: 1. freq of your data
%       2. data in frequency domain
%       3. threshold of the filter, e.g. 5.6Hz

function KB = fn_kb_highpass(freq, signal_freq, highpass_threshold)
    KB = signal_freq./(sqrt(1+power((highpass_threshold./freq),2)));
end