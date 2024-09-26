# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 12:02:04 2023

@author: Russell
"""

# Number of cores to use
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

def data_int(t_i, t_0, t, xx_0, xx, HH_0, HH, TT_0, TT, CC_0, CC, a_0, a):
    
    xx_i = np.zeros(np.size(xx_0))
    HH_i, TT_i, CC_i = map(np.zeros, 3*(np.size(HH_0), ))
    xx_i[:] = (t - t_i)/(t - t_0)*xx_0 + (t_i - t_0)/(t - t_0)*xx
    HH_i[:] = (t - t_i)/(t - t_0)*HH_0 + (t_i - t_0)/(t - t_0)*HH
    TT_i[:] = (t - t_i)/(t - t_0)*TT_0 + (t_i - t_0)/(t - t_0)*TT
    CC_i[:] = (t - t_i)/(t - t_0)*CC_0 + (t_i - t_0)/(t - t_0)*CC
    if a != None and a_0 != None:
        a_i = (t - t_i)/(t - t_0)*a_0 + (t_i - t_0)/(t - t_0)*a
    else:
        a_i = None
    
    return xx_i, HH_i, TT_i, CC_i, a_i