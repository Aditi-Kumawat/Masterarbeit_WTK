import numpy as np
import matplotlib.pyplot as plt
import matlab.engine
from ansys.mapdl.core import launch_mapdl
import HalfPositive_FFT_IFFT as HPFFT
import time as T

def GenerateMaxVelocity(Matlab_Eng_path, FRF_param, Time_param, AriasIntensity, Stucture_params, seed = []):
    
    # Initialized
    Start_time = 0.005
    End_time = Time_param[3] + 5
    Time_interval = 0.005
    Time =  np.arange(Start_time, End_time, Time_interval)
    start_time = T.time()

    Stiffness = Stucture_params[0]
    Mass = Stucture_params[1]

    # Start the matlab engine
    eng = matlab.engine.start_matlab()
    eng.cd(Matlab_Eng_path, nargout=0)
    
    # Generate ground motion
    time,acc,vel,disp = eng.fns_generateGM_Params(seed, Time, 100,"Hu_S0",FRF_param , Time_param, AriasIntensity, nargout=4)
    acc_signal = np.column_stack((time, acc))
    vel_signal = np.column_stack((time, vel))
    disp_signal = np.column_stack((time, disp))

    # Stop the Matlab engine
    eng.quit()

    GM_end_time = T.time()
    # Calculate the elapsed time
    GM_duration = GM_end_time - start_time
    print(f"Generate Ground Motion, Duration: {GM_duration} seconds")

    # Start the ANSYS simulation
    mapdl = launch_mapdl(nproc=4)

    ## PREP7
    mapdl.prep7()
    mapdl.units("SI")  # SI - International system (m, kg, s, K).

    # Create Node
    mapdl.n(1,"0","0","0")
    mapdl.n(2,"0","0","10")

    # Define mass21 element 
    mapdl.et(1, "MASS21", kop3=2)
    mapdl.type(1)
    mapdl.real(1)
    mapdl.r(1,Mass)
    mapdl.e(2)

    mapdl.et(2, "COMBIN14", kop2=3)
    mapdl.type(2)
    mapdl.real(2)
    mapdl.r(2,Stiffness,1000)
    mapdl.e(1,2)

    mapdl.nsel("ALL")
    mapdl.allsel()
    mapdl.nummrg("ALL")

    freq = np.array([0,100])
    U_input = np.array([1,1])
    U_signal = np.column_stack((freq, U_input))
    mapdl.load_table("TF_input", U_signal ,"FREQ")

    ##SOLU
    mapdl.run("/SOLU")
    mapdl.antype(3)  
    mapdl.outres("ALL","NONE")	
    mapdl.outres("NSOL","1")
    mapdl.harfrq("0.1","100")

    substep = int(100/0.2)
    min_sub = int(100/0.1)
    max_sub = int(100/0.5)
    mapdl.nsubst(substep,min_sub,max_sub)

    mapdl.eqslv("SPARSE")  # use sparse solver
    #mapdl.dmprat("0.05")
    mapdl.d("1","UZ","%TF_input%")

    mapdl.solve()

    ## POST26
    mapdl.post26()
    mapdl.nsol("2","1","U","Z")
    mapdl.nsol("3","2","U","Z")

    freq_values = mapdl.post_processing.frequency_values
    freq_values = np.unique(freq_values)

    U_1F_HR = np.empty((0, 3))
    index = 0
    for f in freq_values:
        U_1F_real = mapdl.get("U_1F","VARI","3","REAL",f)
        U_1F_imag = mapdl.get("U_1F","VARI","3","IMAG",f)

        U_1F_HR = np.vstack((U_1F_HR, np.array([f, U_1F_real , U_1F_imag])))
        index = index +1

    mapdl.exit()

    # time and signal from AGT
    time = np.array(time)
    freq,cmplx_values = HPFFT.FFT_half_positive(time,np.squeeze(disp),False,False)

    FRF_freq = U_1F_HR[:,0]
    FRF_real = U_1F_HR[:,1]
    FRF_imag = U_1F_HR[:,2]
    FRF_response = FRF_real + 1j * FRF_imag

    FRF_freq_resample, FRF_values_resample = HPFFT.signal_resample_inter1(FRF_freq,FRF_response, resamples_type = 'array', resample_list = freq, resample_check = False)
    Response = FRF_values_resample*cmplx_values
    Response =  1j*2*np.pi*FRF_freq_resample* Response
    time_res, ampl_res = HPFFT.IFFT_half_positive(Response, time = time, plot_ = False)

    Simu_end_time = T.time()
    # Calculate the elapsed time
    Simu_duration = Simu_end_time - GM_end_time
    print(f"Finish ANSYS simulation, Duration: {Simu_duration} seconds")

    # Record the end time
    end_time = T.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(f"Total Duration: {elapsed_time} seconds")

    return time_res, 1000*ampl_res




