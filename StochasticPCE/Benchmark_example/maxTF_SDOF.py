import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

def sdof_frequency_domain(X,plot_ = False):
    mass = X[:,0:1]
    damping_ratio = X[:,1:2]
    stiffness = X[:,2:3]

    freq = np.arange(0.1,100,0.1)

    num_freq = 2*np.pi*freq
    output = np.zeros_like(mass)
    # Run the simulation
    transfer_function = 1 / (-mass * num_freq.T**2 + 1j * damping_ratio *num_freq.T + stiffness )
    ampl = np.abs(transfer_function)
    output = np.max(ampl, axis = 1)
    if plot_:
        # Create a 3D plot
        fig = plt.figure()
        #Plot the 3D data
        for i in range(1,10):
            plt.plot(freq, ampl[i,:])
        # Add a title
        plt.title("10 Realiazation of TF of SDOF system")
        # Add x and y labels
        plt.xlabel("freq (Hz)")
        plt.ylabel("Amplitude")
        plt.show()
    return output


def realization_SDOF(X_path, mean_array, std_array, import_range = [], plot_ = False):
    data = sio.loadmat(X_path)
    X = np.array([data["X"]])[0]
    #X = X[0]
    X = mean_array + std_array*X
    if import_range != []:
        X = X[import_range[0]:import_range[1],:]
    Y = sdof_frequency_domain(X,plot_)
    return Y