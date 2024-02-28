import scipy.io as sio
import json
import sys
sys.path.append(r"C:/Users/v196m/Desktop/master_project/Masterarbeit/StochasticPCE/Benchmark_example/")
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import SPCE_1ver as SPCE
from scipy.stats import wasserstein_distance
config_file_path = r"C:/Users/v196m/Desktop/master_project/Masterarbeit/StochasticPCE/config/config_SDOF_GM.json"
Y_path = r"C:/Users/v196m/Desktop/master_project/Masterarbeit/StochasticPCE/InputData/Realization/SDOF_AGM_Y_1000.mat"
save_dir = r"C:/Users/v196m/Desktop/master_project/Masterarbeit/StochasticPCE/TrainResult"

Model = SPCE.StochasticPCE(config_file_path, numDOE= None)

Y = sio.loadmat(Y_path)
Realization = np.array([Y["Y"]])[0]

Model.InputRealization(Realization)

#Model.BuildModel_SPCE(tol_err=1E-8, show_info_ = False, overfit_count_ = 100,  save_result_ = True, result_path_ = save_dir)
#Model.BuildModel_SPCE(tol_err=1E-8, show_info_ = False, overfit_count_ = 100,  save_result_ = True, result_path_ = save_dir)

# 1st try 4, 0.75
#Model.Read_result(save_dir,"SPCE_Result_0228055837.json",show_info_= True)

# 2nd try 4, 1
Model.Read_result(save_dir,"SPCE_Result_0228093851.json",show_info_= True)

#Model.Read_result(save_dir,"SPCE_Result_0219220037.json",show_info_= True)
#Model.SensitivityAnalysis(NumIndices= 6, Qol_based= False,show_info_=True)
#SPCE_Result_0217023956.json
#SPCE_Result_0217035719.jsonmaxiter
#Y_real = np.array([])
#for i in range(20000):
#    Y_real_i = BM_E.Benchmark_BlackScholes(np.array([[-0.6, 0.5333]]))
#    Y_real = np.append(Y_real, Y_real_i)
#Y_real = reshaped_Y .reshape(-1,1)

valid = Model.X_train[199,:]
print(valid)
a = np.zeros((100000,4))
a[:,0] = valid[0]
a[:,1] = valid[1]
a[:,2] = valid[2]
a[:,3] = valid[3]

#X_pred, Y_pred = Model.Predict_SPCE(load_type = "np_arr",Predict_type = "all_X", input_X = a)
X_pred, Y_pred = Model.Predict_SPCE(load_type = "mat_file",Predict_type = "all_X", input_X = Model.X_path)


#print(Model.Sigma)

# Plot the histogram
plt.hist(Realization.flatten(), bins=100, density=True, alpha=0.6, color='b')
plt.hist(Y_pred.flatten(), bins=300, density=True, alpha=0.6, color='r')
#plt.show()
# Add labels and title
#plt.xlabel('Value')
#plt.ylabel('Frequency')
#plt.title('Histogram of Gaussian Distribution')
distance = wasserstein_distance(Realization.flatten(), Y_pred.flatten())
# Add text for the Wasserstein distance
print(distance)


# Show the plot

plt.show()#