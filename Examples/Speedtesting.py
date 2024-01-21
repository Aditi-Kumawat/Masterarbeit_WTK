from ansys.mapdl.core import launch_mapdl

import matplotlib.pyplot as plt
import numpy as np
import time

# Record the start time
start_time = time.time()
mapdl = launch_mapdl(nproc=4)
print("ACTIVATE")


mapdl.prep7()
mapdl.units("SI")  # SI - International system (m, kg, s, K).


# Create Node
mapdl.n(1,"0","0","0")
mapdl.n(2,"0","0","10")

# Define mass21 element 
mapdl.et(1, "MASS21", kop3=2)
mapdl.type(1)
mapdl.real(1)
mapdl.r(1,100)
mapdl.e(2)

mapdl.et(2, "COMBIN14", kop2=3)
mapdl.type(2)
mapdl.real(2)
k = (4*np.pi**2)*10000
mapdl.r(2,k,1000)
mapdl.e(1,2)


mapdl.nsel("ALL")
mapdl.allsel()
mapdl.nummrg("ALL")

freq = np.array([0,100])
U_input = np.array([1,1])
U_signal = np.column_stack((freq, U_input))
mapdl.load_table("TF_input", U_signal ,"FREQ")


print("SOLUTION")

mapdl.run("/SOLU")
mapdl.antype(3)  

mapdl.outres("ALL","NONE")	
mapdl.outres("NSOL","1")
#mapdl.outres("EPFL")
#mapdl.outres("ESOL")
#mapdl.outres("RSOL")
#mapdl.outres("V","1")
#mapdl.outres("A","1")

mapdl.harfrq("0.1","100")

substep = int(100/0.2)
min_sub = int(100/0.1)
max_sub = int(100/0.5)
mapdl.nsubst(substep,min_sub,max_sub)


#mapdl.eqslv("SPARSE")  # use sparse solver
mapdl.dmprat("0.05")
#mapdl.d("1","ACCZ","%ACC_input%")
mapdl.d("1","UZ","%TF_input%")

mapdl.solve()


print("POST26")
mapdl.post26()
acc_z_EG = mapdl.nsol("2","1","U","Z")
acc_z_1F = mapdl.nsol("3","2","U","Z")

freq_values = mapdl.post_processing.frequency_values
freq_values = np.unique(freq_values)
U_info_EG = np.empty((0, 3))
U_info_1F = np.empty((0, 3))

U_EG = []
U_1F = []
for f in freq_values:
    #print(mapdl.get("U_EG","VARI","2","AMPL",f))
    #print(mapdl.get("U_1F","VARI","3","AMPL",f))
    U_EG_real = mapdl.get("U_EG","VARI","2","REAL",f)
    U_EG_imag = mapdl.get("U_EG","VARI","2","IMAG",f)
    U_1F_real = mapdl.get("U_1F","VARI","3","REAL",f)
    U_1F_imag = mapdl.get("U_1F","VARI","3","IMAG",f)

    #U_info_EG = np.vstack((U_info_EG, np.array([f, U_EG_real , U_EG_imag])))
    #U_info_1F = np.vstack((U_info_1F, np.array([f, U_1F_real , U_1F_imag])))
    
    U_EG.append(mapdl.get("U_EG","VARI","2","AMPL",f))
    U_1F.append(mapdl.get("U_1F","VARI","3","AMPL",f))

# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print(f"Elapsed Time: {elapsed_time} seconds")

plt.plot(freq_values, U_EG, label="ACC EG")
plt.plot(freq_values, U_1F, label="ACC 1F")
#plt.plot(time_values, label="ACC time")
plt.legend()
plt.xlabel("freq (Hz)")
plt.ylabel("U")
plt.show()