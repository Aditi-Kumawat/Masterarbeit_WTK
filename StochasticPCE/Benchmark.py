import scipy.io as sio
import json
import sys
sys.path.append(r"C:/Users/v196m/Desktop/master_project/Masterarbeit/StochasticPCE/Benchmark_example/")
import numpy as np
import Benchmark_Example as BM_E
import matplotlib.pyplot as plt
import SPCE_1ver as SPCE
config_file_path = r"C:/Users/v196m/Desktop/master_project/Masterarbeit/StochasticPCE/config/config_BM_BlackScholes.json"
save_dir = r"C:/Users/v196m/Desktop/master_project/Masterarbeit/StochasticPCE/TrainResult"

Model = SPCE.StochasticPCE(config_file_path, numDOE= None)

#Y = BM_E.Benchmark_BlackScholes(Model.X)
#reshaped_Y = np.reshape(Y, (len(Y), 1))
#Model.InputRealization(reshaped_Y)

Model.BuildModel_SPCE(tol_err=1E-8, show_info_ = False, overfit_count_ = 100,  save_result_ = True, result_path_ = save_dir)
#Model.BuildModel_SPCE(tol_err=1E-8, show_info_ = False, overfit_count_ = 100,  save_result_ = True, result_path_ = save_dir)
#Model.Read_result(save_dir,"SPCE_Result_0219134106.json",show_info_= True)


Model.Read_result(save_dir,"SPCE_Result_0219231452.json",show_info_= True)
#Model.Read_result(save_dir,"SPCE_Result_0219220037.json",show_info_= True)
#Model.SensitivityAnalysis(NumIndices= 6, Qol_based= False,show_info_=True)
#SPCE_Result_0217023956.json
#SPCE_Result_0217035719.jsonmaxiter
#Y_real = np.array([])
#for i in range(20000):
#    Y_real_i = BM_E.Benchmark_BlackScholes(np.array([[-0.6, 0.5333]]))
#    Y_real = np.append(Y_real, Y_real_i)
#Y_real = reshaped_Y .reshape(-1,1)


a = np.zeros((100000,2))
a[:,0] = -0.6
a[:,1] = 0.533333

X_pred, Y_pred = Model.Predict_SPCE(load_type = "np_arr",Predict_type = "all_X", input_X = a)
#X_pred, Y_pred = Model.Predict_SPCE(load_type = "mat_file",Predict_type = "all_X", input_X = Model.X_path)


#print(Model.Sigma)

# Plot the histogram
#plt.hist(Y_real.flatten(), bins=100, density=True, alpha=0.6, color='b')
plt.hist(Y_pred.flatten(), bins=200, density=True, alpha=0.6, color='r')
#plt.show()
# Add labels and title
#plt.xlabel('Value')
#plt.ylabel('Frequency')
#plt.title('Histogram of Gaussian Distribution')

# Show the plot

plt.show()#