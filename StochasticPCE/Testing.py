from threadpoolctl import ThreadpoolController
import time
# basic optimization of the variational functional for a random symmetric matrix

import numpy as np
from numpy.random import uniform
from scipy.optimize import minimize

# generate random symmetric matrix to compute minimal eigenvalue of
N = 1000
H = uniform(-1, 1, [N,N])
H = H + H.T

x0 = uniform(-0.1, 0.1, N)
# variational cost function
def cost(x):
    return (x @ H @ x) / (x @ x)
controller = ThreadpoolController()
for i in range(1, 5):
    t0 = time.time()
    with controller.limit(limits=i, user_api='blas'):
        print(minimize(cost, x0, method='L-BFGS-B'))
    t = time.time()
    print(f"Threads {i}, Duration: {t - t0:.3f}")
