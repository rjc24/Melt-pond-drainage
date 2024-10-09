# Number of cores to use
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from scipy.linalg import inv

def int_lin(TT, CC, xx, xin, a_0):

    j = (TT >= CC).nonzero()[0][-1]    # Boundary index

    # left hand extrapolation
    a = (xin[j]*(CC[j-1] - TT[j-1]) - xin[j-1]*(CC[j] - TT[j]))/\
        (TT[j] - CC[j] - (TT[j-1] - CC[j-1])) # a1
    if a > xx[-1]:
        a = np.nan

    return a

def int_quad(TT, CC, xx, xin, a_0):

    if not np.isnan(a_0):
        j = (TT >= CC).nonzero()[0][-1]    # Boundary index
        if j >= 2:
            # left hand extrapolation
            # temperature
            A1 = np.array([[xin[j-2]**2, xin[j-2], 1], [xin[j-1]**2, xin[j-1], 1], \
                         [xin[j]**2, xin[j], 1]])
            b1 = np.array([TT[j-2], TT[j-1], TT[j]])
            A1_inv = inv(A1)
            c1 = A1_inv.dot(b1)
            
            # salinity
            A2 = np.array([[xin[j-2]**2, xin[j-2], 1], [xin[j-1]**2, xin[j-1], 1], \
                         [xin[j]**2, xin[j], 1]])
            b2 = np.array([CC[j-2], CC[j-1], CC[j]])
            A2_inv = inv(A2)
            c2 = A2_inv.dot(b2)

            disc1 = (c1[1] - c2[1])**2/(c1[0] - c2[0])**2 - \
                4*(c1[2] - c2[2])/(c1[0] - c2[0])   # discriminant
            if disc1 >= 0:
                a1p, a1m = 0.5*(-(c1[1] - c2[1])/(c1[0] - c2[0]) + np.sqrt(disc1)), \
                           0.5*(-(c1[1] - c2[1])/(c1[0] - c2[0]) - np.sqrt(disc1))
                if abs(a1p - a_0) < abs(a1m - a_0):
                    a = a1p
                else:
                    a = a1m
                if a < 0:
                    a = int_lin(TT, CC, xx, xin, a_0)
            else:
                a = int_lin(TT, CC, xx, xin, a_0)
        elif j == 1:
            a = int_lin(TT, CC, xx, xin, a_0)
        elif j == 0:
            a = 0
        if not np.isnan(a):
            if a > xx[-1]:
                a = np.nan
            elif a < xx[0]:
                a = xx[0]
    else:
        a = np.nan

    return a


