import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.stats import norm

def Uniform_mapping(X, lower, upper):
    X = (X + 1) / 2

    return (upper - lower)*X +  lower

def Benchmark_Ishigami(X,a=7, b=0.1):
    """
    Ishigami function.
    
    Parameters:
    - x: np.array of shape (N, 3)
    - a, b: Parameters (default: a=7, b=0.1)
    
    Returns:
    - y: np.array of shape (N,)
    """
    if X.shape[1] != 3:
        raise ValueError("Input array must have shape (N, 3)")
    
    x1 = X[:, 0]
    x2 = X[:, 1]
    x3 = X[:, 2]
   
    y = (np.sin(x1) + a * np.sin(x2)**2 + b * x3**4 * np.sin(x1))
    return y

def Benchmark_Ishigami_s(X,a=7, b=0.1):
    """
    Ishigami function.
    
    Parameters:
    - x: np.array of shape (N, 3)
    - a, b: Parameters (default: a=7, b=0.1)
    
    Returns:
    - y: np.array of shape (N,)
    """
    if X.shape[1] != 3:
        raise ValueError("Input array must have shape (N, 3)")
    
    x1 = X[:, 0]
    x2 = X[:, 1]
    x3 = X[:, 2]
   
    y = (np.sin(x1) + a * np.sin(x2)**2 + b * x3**4 * np.sin(x1)) 
    y = np.random.normal(0.1*y, 0.05, len(x1))
    return y


def Benchmark_BlackScholes(X):
    """
    Simulate stock price using Geometric Brownian Motion and price European call option using Black-Scholes model.
    
    """
    # Example usage:
    
    # Parameters
    initial_price = 1.0
    dt = 0.00025  # time step
    T = 1.0    # final time
    X1 =  Uniform_mapping(X[:, 0], 0, 0.1)  # Risk-free interest rate (5%)
    X2 =  Uniform_mapping(X[:, 1], 0.1, 0.4)  # Volatility (20%)
    num_steps = int(T / dt)
    prices = np.zeros(num_steps + 1)
    prices[0] = initial_price
    call_price_list = []
    for j in range(np.shape(X)[0]):
        x1 = X1[j]
        x2 = X2[j]
        for i in range(num_steps):
            dW = np.random.normal(0,1)
            prices[i+1] = prices[i] + x1 * prices[i] * dt + x2 * prices[i] *np.sqrt(dt)* dW
        call_price_list.append([prices[-1]])
    return call_price_list



def Benchmark_borehole(X):
    """
    Ishigami function.
    
    Parameters:
    - x: np.array of shape (N, 8)

    Returns:
    - y: np.array of shape (N,)
    """
    if X.shape[1] != 8:
        raise ValueError("Input array must have shape (N, 8)")
    
    # Gaussian
    rw = 0.10 + 0.0161812 *X[:, 0]

    # Uniform
    L  = Uniform_mapping(X[:, 1], 1120, 1680)
    Kw = Uniform_mapping(X[:, 2], 9855, 12045)
    Tu = Uniform_mapping(X[:, 3], 63070, 115600)
    Tl = Uniform_mapping(X[:, 4], 63.1, 116)
    Hu = Uniform_mapping(X[:, 5], 990, 1110)
    Hl = Uniform_mapping(X[:, 6], 700, 820)

    # Lognormal 
    r = np.exp(7.71 + 1.0056* X[:, 7])

    return borehole_function(rw, L ,Kw, Tu, Tl, Hu, Hl, r)

def borehole_function(rw, L ,Kw, Tu, Tl, Hu, Hl, r):
    """
    Compute the flow rate of water through a borehole.
    
    Parameters:
    Kw (float):  borehole hydraulic conductivity
    rw (float): Radius of the borehole (m)
    r (float): Radius of influence (m)
    Tu (float): Transmissivity of upper aquifer (m^2/year)
    Hu (float): Hydraulic head of upper aquifer (m)
    Tl (float): Transmissivity of lower aquifer (m^2/year)
    Hl (float): Hydraulic head of lower aquifer (m)
    L (float): Length of borehole (m)
    
    Returns:
    Q (float): Flow rate of water through the borehole (m^3/year)
    """
    
    frac1 = 2 * np.pi * Tu * (Hu - Hl)
    frac2 = np.log(r / rw) 
    frac3 = (2*L * Tu) / (frac2* Kw* rw**2 )
    frac4 = 1 + (Tu / Tl)

    Q = frac1 / (frac2 * (frac3 + frac4) ) 
    
    return Q


