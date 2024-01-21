import numpy as np
import matplotlib.pyplot as plt


def FFT_half_positive(time,data_t, iFFT_check = False, plot_ = False):
    #time = data_time[0,:] 
    ampl =  data_t#data_time[1,:] 
    fs = int(1 / (time[3] - time[2]))
    
    # Perform FFT
    fft_result = np.fft.fft(ampl)
    frequencies_fft = np.fft.fftfreq(len(fft_result), 1/fs)
    fft_result_HP = fft_result[frequencies_fft > 0]
    fft_freq_HP = frequencies_fft[frequencies_fft > 0]
    #fft_result_HP = fft_result_HP[:,0]
    

    if iFFT_check:
        # Perform IFFT on the positive frequencies
        ifft_result = np.fft.ifft(np.concatenate(([0],fft_result_HP,[0], np.conj(fft_result_HP[::-1]))))


        plt.plot(time, ampl, label='origianl data')
        plt.plot(time, np.real(ifft_result), label='reconstructed data')
        plt.title('Reconstructed Signal from Positive Frequencies')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()

        plt.show()

    if plot_:
        # Plot the original signal
        plt.subplot(2, 1, 1)
        plt.plot(time, ampl)
        plt.title('Original Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        
        # Plot the FFT result
        plt.subplot(2, 1, 2)
        plt.plot(fft_freq_HP, np.abs(fft_result_HP))
        plt.title('FFT Result')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.show()

    return fft_freq_HP, fft_result_HP


def signal_resample_inter1(freq,cmplx_data, resamples_type = '', num_samples = 0, resample_list = np.array([]), resample_check = False):
    freq_HP = freq
    data_f_HP = cmplx_data
  
    if resamples_type == 'num':
        f_num_samples_list = np.linspace(freq_HP[0],freq_HP[-1],num_samples)
    elif resamples_type == 'array':
        f_num_samples_list = resample_list
    else:
        print("ERROR")
    
        
    resample_signal_R = np.interp(f_num_samples_list, freq_HP, data_f_HP.real)
    resample_signal_I = np.interp(f_num_samples_list, freq_HP, data_f_HP.imag)
    resample_signal = resample_signal_R + 1j * resample_signal_I

    if resample_check:
        plt.plot(freq_HP, np.abs(data_f_HP),label='original data')
        plt.plot(f_num_samples_list, np.abs(resample_signal),label='resample data')
        plt.legend()
        plt.show()
    
    return f_num_samples_list , resample_signal

  

def IFFT_half_positive(cmplx_data, time = np.array([]), plot_ = False):
    # Perform IFFT on the positive frequencies
    ifft_result = np.fft.ifft(np.concatenate(([0],cmplx_data,[0], np.conj(cmplx_data[::-1]))))
    time_adjust = np.linspace(time[0],time[-1],len(ifft_result))
    
    if plot_:
        plt.plot(time_adjust, np.real(ifft_result), label='reconstructed data')
        plt.title('Reconstructed Signal from Positive Frequencies')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()

        plt.show()

    return time_adjust, np.real(ifft_result)


#t = np.arange(0,5,1/200)  # Time vector
#freq_range = np.linspace(0,100,3000)
##
#frequencies = [5, 10, 80]  # Frequencies of sinusoidal components
#signal = np.sin(2 * np.pi * frequencies[0] * t) + np.sin(2 * np.pi * frequencies[1] * t) + np.sin(2 * np.pi * frequencies[2] * t)


#
#freq_, data_freq = FFT_half_positive(t,signal,True,True)
#
#re_freq, re_ampl = signal_resample_inter1(freq_, data_freq, resamples_type = 'num',num_samples=3000,resample_check= True)
#
#
#IFFT_half_positive(data_freq,t,True)
#IFFT_half_positive(re_ampl,t,True)




## Generate some sample data
#fs = 200  # Sampling frequency
#t = np.linspace(0, 1, 10*fs, endpoint=False)  # Time vector
#frequencies = [10, 50, 100]  # Frequencies of sinusoidal components
#signal = np.sin(2 * np.pi * frequencies[0] * t) + np.sin(2 * np.pi * frequencies[1] * t) + np.sin(2 * np.pi * frequencies[2] * t)
#
## Perform FFT
#fft_result = np.fft.fft(signal)
#frequencies_fft = np.fft.fftfreq(len(fft_result), 1/fs)
#fft_result_HP = fft_result[frequencies_fft > 0]
#fft_freq_HP = frequencies_fft[frequencies_fft > 0]
#
## Perform IFFT on the positive frequencies
#ifft_result = np.fft.ifft(np.concatenate(([0],fft_result_HP, np.conj(fft_result_HP[::-1]))))
#
## Adjust the time vector
#t_ifft = np.linspace(0, 1, len(ifft_result), endpoint= False)
#
#
## Plot the original signal
##plt.subplot(3, 1, 1)
#plt.plot(t, signal)
##plt.title('Original Signal')
##plt.xlabel('Time (s)')
##plt.ylabel('Amplitude')
#
### Plot the FFT result
##plt.subplot(3, 1, 2)
##plt.plot(fft_freq_HP, np.abs(fft_result_HP))
##plt.title('FFT Result')
##plt.xlabel('Frequency (Hz)')
##plt.ylabel('Amplitude')
#
#
##plt.subplot(3, 1, 3)
#plt.plot(t_ifft, np.real(ifft_result))
#plt.title('Reconstructed Signal from Positive Frequencies')
#plt.xlabel('Time (s)')
#plt.ylabel('Amplitude')

