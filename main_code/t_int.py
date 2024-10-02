# Number of cores to use
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

def data_int(t_i, t_0, t, HH_0, HH, TT_0, TT, CC_0, CC, pp_0, pp):

    #=========================================================================
    # Linearly interpolates between data at time t_0 and t to time t_i.
    # Calculates enthalpy (HH), temperature (TT), bulk salinity (CC), 
    # pressure (pp)
    #
    #=========================================================================
    
    Nx, Nz = np.shape(HH_0)
    HH_i, TT_i, CC_i, pp_i = map(np.zeros, 4*((Nx, Nz), ))
    
    HH_i[:, :] = (t - t_i)/(t - t_0)*HH_0[:, :] + (t_i - t_0)/(t - t_0)*HH[:, :]
    TT_i[:, :] = (t - t_i)/(t - t_0)*TT_0[:, :] + (t_i - t_0)/(t - t_0)*TT[:, :]
    CC_i[:, :] = (t - t_i)/(t - t_0)*CC_0[:, :] + (t_i - t_0)/(t - t_0)*CC[:, :]
    pp_i[:, :] = (t - t_i)/(t - t_0)*pp_0[:, :] + (t_i - t_0)/(t - t_0)*pp[:, :]
    

    return HH_i, TT_i, CC_i, pp_i
