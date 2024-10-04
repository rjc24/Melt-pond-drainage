# Number of cores to use
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

# functions for calculating vertical temperature profile (initial conditions)

def F_z(T, z, z_B, z_T, T_B, T_T, Cr):

    k_l, k_s = 0.58, 2.0   # liquid/solid conductivity

    v_s = k_s/k_l   # conductivity ratio
    
    lam = (v_s - 1)
    
    F = ((T - T_B)*z_T - (T - T_T)*z_B)/(z_T - z_B) + \
        lam/(z_T - z_B)*((T - T_B)*z_T - (T - T_T)*z_B + \
        Cr*(z_T*np.log((T - Cr)/(T_B - Cr)) - \
        z_B*np.log((T - Cr)/(T_T - Cr)))) - \
        ((T_T - T_B)/(z_T - z_B) + \
        lam/(z_T - z_B)*(T_T - T_B + Cr*np.log((T_T - Cr)/(T_B - Cr))))*z

    return F

def dF_z(T, z, z_B, z_T, T_B, T_T, Cr):
    
    k_l, k_s = 0.58, 2.0   # liquid/solid conductivity

    v_s = k_s/k_l   # diffusivity ratio

    lam = (v_s - 1)

    dF = (1 + lam*(1 + Cr/(T - Cr)))

    return dF

def dF_zdz(T, z, z_B, z_T, T_B, T_T, Cr):
    
    k_l, k_s = 0.58, 2.0   # liquid/solid conductivity

    v_s = k_s/k_l   # diffusivity ratio

    lam = (v_s - 1)

    dF = (T_T - T_B + lam*(T_T - T_B + Cr*np.log((T_T - Cr)/(T_B - Cr))))/\
        ((z_T - z_B)*(1 + lam*(1 + Cr/(T - Cr)))) 

    return dF

