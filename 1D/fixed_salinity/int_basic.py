# Number of cores to use
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from scipy.linalg import inv

def int_lin(TT, CC, xx, xin, a_0):
    
    #=========================================================================
    # Calculates the interface position (a) by linearly extrapolating 
    # temperature and bulk salinity from the liquid
    #=========================================================================    

    j = (TT >= CC).nonzero()[0][-1]    # Boundary index

    # left hand extrapolation
    a = (xin[j]*(CC[j-1] - TT[j-1]) - xin[j-1]*(CC[j] - TT[j]))/\
        (TT[j] - CC[j] - (TT[j-1] - CC[j-1])) # a1
    if a > xx[-1]:
        a = np.nan

    return a

def int_quad(TT, CC, xx, xin, a_0, OPT = 1):
    
    #=========================================================================
    # Calculates the interface position (a) by quadratic extrapolation of
    # temperature and bulk salinity fields in the liquid. Also outputs
    # the temperature at the interface position (T_f), estimates of
    # the temperature in the liquid portion of the cell 
    # containing the interface (T_c).
    #=========================================================================
    
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
                T_f = c1[0]*a**2 + c1[1]*a + c1[2]
                # temp in channel portion of interface cell
                if xx[j+1] < a:
                    x_c = (a + xx[j+1])/2
                else:
                    x_c = (a + xx[j])/2
                T_c = c1[0]*x_c**2 + c1[1]*x_c + c1[2]
                if a < 0:
                    A1 = np.array([[xin[j-1], 1], \
                                   [xin[j], 1]])
                    b1 = np.array([TT[j-1], TT[j]])
                    A1_inv = inv(A1)
                    c1 = A1_inv.dot(b1)
                    
                    # salinity
                    A2 = np.array([[xin[j-1], 1], \
                                   [xin[j], 1]])
                    b2 = np.array([CC[j-1], CC[j]])
                    A2_inv = inv(A2)
                    c2 = A2_inv.dot(b2)
                    
                    a = (c2[1] - c1[1])/(c1[0] - c2[0])
                    T_f = c1[0]*a + c1[1]
                    
                    # temp in channel portion of interface cell
                    if xx[j+1] < a:
                        x_c = (a + xx[j+1])/2
                    else:
                        x_c = (a + xx[j])/2
                    T_c = c1[0]*x_c + c1[1]
            else:
                A1 = np.array([[xin[j-1], 1], \
                               [xin[j], 1]])
                b1 = np.array([TT[j-1], TT[j]])
                A1_inv = inv(A1)
                c1 = A1_inv.dot(b1)
                
                # salinity
                A2 = np.array([[xin[j-1], 1], \
                               [xin[j], 1]])
                b2 = np.array([CC[j-1], CC[j]])
                A2_inv = inv(A2)
                c2 = A2_inv.dot(b2)
                
                a = (c2[1] - c1[1])/(c1[0] - c2[0])
                T_f = c1[0]*a + c1[1]
                
                # temp in channel portion of interface cell
                if xx[j+1] < a:
                    x_c = (a + xx[j+1])/2
                else:
                    x_c = (a + xx[j])/2
                T_c = c1[0]*x_c + c1[1]
        elif j == 1:
            A1 = np.array([[xin[j-1], 1], \
                           [xin[j], 1]])
            b1 = np.array([TT[j-1], TT[j]])
            A1_inv = inv(A1)
            c1 = A1_inv.dot(b1)
            
            # salinity
            A2 = np.array([[xin[j-1], 1], \
                           [xin[j], 1]])
            b2 = np.array([CC[j-1], CC[j]])
            A2_inv = inv(A2)
            c2 = A2_inv.dot(b2)
            
            a = (c2[1] - c1[1])/(c1[0] - c2[0])
            T_f = c1[0]*a + c1[1]

            # temp in channel portion of interface cell
            if xx[j+1] < a:
                x_c = (a + xx[j+1])/2
            else:
                x_c = (a + xx[j])/2
            T_c = c1[0]*x_c + c1[1]
        elif j == 0:
            a = 0
            T_f = 0
            T_c = 0
        if not np.isnan(a):
            if a > xx[-1]:
                a = np.nan
                T_f = np.nan
                T_c = 0
            elif a < xx[0]:
                a = xx[0]
                T_f = 0
                T_c = 0
    else:
        a = np.nan
        T_f = np.nan
        T_c = 0

    if OPT == 1:
        return a
    elif OPT == 2:
        return a, T_f, T_c
