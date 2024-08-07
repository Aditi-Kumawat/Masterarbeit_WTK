import scipy.io as sio
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import SPCE_ParallelGP as SPCE
import time


start_time = time.time()
config_file_path = r"C:/Users/v196m/Desktop/master_project/Masterarbeit/TESTING_ENV/config/config_TEST1_SDOF_NUMDATA_COMPARE.json"
Y_path = r"C:/Users/v196m/Desktop/master_project/Masterarbeit/TESTING_ENV/InputData/Realization/TEST1_Y_SBAGM_V6_TR_RAND_DOE_100_DIR_Z_5.mat"
save_dir = r"C:/Users/v196m/Desktop/master_project/Masterarbeit/TESTING_ENV/TrainResult/"

# Initialized 
Model = SPCE.StochasticPCE(config_file_path, numDOE= None)

# Import the realization for training 
Y = sio.loadmat(Y_path)
Realization = np.array([Y["Y"]])[0]
Model.InputRealization(Realization)

# Training 
#Model.BuildModel_SPCE(tol_err=1E-6, show_info_ = False, overfit_count_ = 10,  save_result_ = True, result_path_ = save_dir)

# Import result 
#Model.Read_result(save_dir,"SPCE_Result_DATA1600_1.json",show_info_= True)


# Validation 
#valid = Model.X_train[149,:]
#a = np.zeros((1000,6))
#vary_params = np.zeros((1000,))

# if only modified the sensitive params
#a[:,0] = np.random.normal(loc=0, scale=1, size=vary_params.shape)
#a[:,1] = np.random.uniform(low=-1, high=1,size=vary_params.shape)

#a[:,0] = valid[0]
#a[:,1] = valid[1]
#a[:,2] = valid[2]
#a[:,3] = valid[3]
#a[:,4] = valid[4]
#a[:,5] = valid[5]

#X_pred, Y_pred = Model.Predict_SPCE(load_type = "np_arr",Predict_type = "all_X", input_X = a)
#X_pred, Y_pred = Model.Predict_SPCE(load_type = "mat_file",Predict_type = "all_X", input_X = Model.X_path)
#Model.ComputeERROR_WD(Y_pred, Realization)
# Plot the histogram
#plt.hist(Realization.flatten(), bins=50, density=True, alpha=0.6, color='b',label = 'Realization')
#plt.hist(Y_pred.flatten(), bins=50, density=True, alpha=0.6, color='r',label = 'Pred')
#plt.xlabel('Value')
#plt.ylabel('Frequency')
#plt.title('Comparison of Histogram, PCE with Residual')
#plt.legend()
#plt.show()#

#end_time = time.time()
#execution_time = end_time - start_time
#print(f"Execution time: {execution_time} seconds")

error_1_list = []
error_2_list = []

num_repeat = 30


for i in range(0,num_repeat):
    Model.Read_result(save_dir,"SPCE_Result_DATA100_1.json",show_info_= True)
    X_pred, Y_pred = Model.Predict_SPCE(load_type = "mat_file",Predict_type = "all_X", input_X = Model.X_path)
    error_1, error_2 = Model.ComputeERROR_WD(Y_pred, Realization)
    error_1_list.append(error_1)
    error_2_list.append(error_2)

for i in range(0,num_repeat):
    Model.Read_result(save_dir,"SPCE_Result_DATA100_2.json",show_info_= True)
    X_pred, Y_pred = Model.Predict_SPCE(load_type = "mat_file",Predict_type = "all_X", input_X = Model.X_path)
    error_1, error_2 = Model.ComputeERROR_WD(Y_pred, Realization)
    error_1_list.append(error_1)
    error_2_list.append(error_2)

for i in range(0,num_repeat):
    Model.Read_result(save_dir,"SPCE_Result_DATA100_3.json",show_info_= True)
    X_pred, Y_pred = Model.Predict_SPCE(load_type = "mat_file",Predict_type = "all_X", input_X = Model.X_path)
    error_1, error_2 = Model.ComputeERROR_WD(Y_pred, Realization)
    error_1_list.append(error_1)
    error_2_list.append(error_2)

error_1_arr = np.array(error_1_list)
error_2_arr = np.array(error_2_list)

print(f"Mean: {np.mean(error_1_arr),np.mean(error_2_arr)}")
print(f"q1: {np.percentile(error_1_arr, 25),np.percentile(error_2_arr, 25)}")
print(f"q2: {np.percentile(error_1_arr, 50),np.percentile(error_2_arr, 50)}")
print(f"q3: {np.percentile(error_1_arr, 75),np.percentile(error_2_arr, 75)}")
print(f"Maximum: {np.max(error_1_arr),np.max(error_2_arr)}")
print(f"Minimum: {np.min(error_1_arr),np.min(error_2_arr)}")







