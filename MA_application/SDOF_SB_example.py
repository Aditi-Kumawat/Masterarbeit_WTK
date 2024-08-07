import scipy.io as sio
import json
import sys

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import SPCE_ParallelGP as SPCE
#import SPCE_SingleCore as SPCE
from scipy.stats import wasserstein_distance
import time



start_time = time.time()
config_file_path = r"C:/Users/v196m/Desktop/master_project/Masterarbeit/TESTING_ENV/config/config_TEST3_SDOF_SBAGM_6_X.json"
Y_path = r"C:/Users/v196m/Desktop/master_project/Masterarbeit/GroundMotion/GroundMotionGeneration/TEST_CASE/REALIZATION/TEST3_Y_SBAGM_V6_VAL_FIX550_DOE_1000_DIR_Z.mat"
save_dir = r"C:/Users/v196m/Desktop/master_project/Masterarbeit/TESTING_ENV/TrainResult/"

Model = SPCE.StochasticPCE(config_file_path, numDOE= None)
Y = sio.loadmat(Y_path)
Realization = np.array([Y["Y"]])[0]
Model.InputRealization(Realization)
#Model.BuildModel_PCE(tol_err=1E-6, show_info_ = False, overfit_count_ = 2,  save_result_ = True, result_path_ = save_dir)

#Model.BuildModel_SPCE(tol_err=1E-8, show_info_ = False, overfit_count_ = 100,  save_result_ = True, result_path_ = save_dir)
# 1st try 4, 0.75
#Model.Read_result(save_dir,"SPCE_Result_0228055837.json",show_info_= True)
# 2nd try 4, 1
#Model.Read_result(save_dir,"SPCE_Result_0228093851.json",show_info_= True)

# 3rd try 4,1
#Model.Read_result(save_dir,"SPCE_Result_0229065103.json",show_info_= True)


# 4th try 5,1
#Model.Read_result(save_dir,"SPCE_Result_0301011247.json",show_info_= True)

#5th try 4,1 
#Model.Read_result(save_dir,"SPCE_Result_0301015322.json",show_info_= True)



# SBGM try 3, 0.75
#Model.Read_result(save_dir,"SPCE_Result_0307195408.json",show_info_= True)



# SBGM try 3, 1   config_SDOF_SBGM5
#Model.Read_result(save_dir,"SPCE_Result_0308222604.json",show_info_= True)
 
# SBGM try 4, 1   config_SDOF_SBGM5
#Model.Read_result(save_dir,"SPCE_Result_X_3rd.json",show_info_= True)


# SBGM try 4, 1   config_SDOF_SBGM5 PCE
#Model.Read_result(save_dir,"PCE_Result_0415150659.json",show_info_= True)
#Model.Read_result(save_dir,"PCE_Result_0415163940.json",show_info_= True)
#Model.Read_result(save_dir,"SPCE_Result_X_2nd.json",show_info_= True)
#Model.Read_result(save_dir,"SPCE_Result_Y_2nd.json",show_info_= True)

#Model.Read_result(save_dir,"SPCE_Result_0219220037.json",show_info_= True)
#Model.SensitivityAnalysis(NumIndices= 6, Qol_based= True,show_info_=True)
#SPCE_Result_0217023956.json
#SPCE_Result_0217035719.jsonmaxiter
#Y_real = np.array([])
#for i in range(20000):
#    Y_real_i = BM_E.Benchmark_BlackScholes(np.array([[-0.6, 0.5333]]))
#    Y_real = np.append(Y_real, Y_real_i)
#Y_real = reshaped_Y .reshape(-1,1)
valid = Model.X_train[49,:]
print(valid)
#
a = np.zeros((500,6))
a[:,0] = valid[0]
a[:,1] = valid[1]
a[:,2] = valid[2]
a[:,3] = valid[3]
a[:,4] = valid[4]
a[:,5] = valid[5]
#X_pred, Y_pred = Model.Predict_SPCE(load_type = "np_arr",Predict_type = "all_X", input_X = a)
#X_pred, Y_pred = Model.Predict_SPCE(load_type = "mat_file",Predict_type = "all_X", input_X = Model.X_path)

#X_pred, Y_pred = Model.Predict_PCE(load_type = "mat_file", input_X = Model.X_path)
#X_pred, Y_pred = Model.Predict_PCE(load_type = "np_arr", input_X = a)
#noise = np.random.normal(loc=0, scale=Model.error_loo, size=Y_pred.shape)
#Y_pred = Y_pred + noise
#Model.ComputeERROR_WD(Y_pred, Realization)

#print(Model.Sigma)
# Plot the histogram
#plt.hist(Realization.flatten(), bins=50, density=True, alpha=0.6, color='b',label = 'Realization')
#plt.hist(Y_pred.flatten(), bins=50, density=True, alpha=0.6, color='r',label = 'Pred')

# Add labels and title
#plt.xlabel('Value')
#plt.ylabel('Frequency')
#plt.title('Comparison of Histogram, PCE with Residual')
#plt.legend()
#distance = wasserstein_distance(Realization.flatten(), Y_pred.flatten())
# Add text for the Wasserstein distance
#print(distance)
# Show the plot
#plt.show()#


end_time = time.time()

execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")




