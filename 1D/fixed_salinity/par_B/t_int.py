# Number of cores to use
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

def data_int(t_i, t_0, t, xx, HH_0, HH, TT_0, TT, CC_0, CC, a_0, a):
    
    #=========================================================================
    # Linearly interpolates between data at time t_0 and t to time t_i.
    # Calculates enthalpy (HH), temperature (TT), bulk salinity (CC) and the
    # interface position a_i.
    #=========================================================================
    
    HH_i, TT_i, CC_i = map(np.zeros, 3*(np.size(HH_0), ))
    HH_i[:] = (t - t_i)/(t - t_0)*HH_0 + (t_i - t_0)/(t - t_0)*HH
    TT_i[:] = (t - t_i)/(t - t_0)*TT_0 + (t_i - t_0)/(t - t_0)*TT
    CC_i[:] = (t - t_i)/(t - t_0)*CC_0 + (t_i - t_0)/(t - t_0)*CC
    if a != None and a_0 != None:
        a_i = (t - t_i)/(t - t_0)*a_0 + (t_i - t_0)/(t - t_0)*a
    else:
        a_i = None
    
    return HH_i, TT_i, CC_i, a_i
