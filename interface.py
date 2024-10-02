# Number of cores to use
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

def int_loc(HH, CC, xx, zz, xx_p, zz_p):

    #========================================================================
    # Calculates the approximate left and right interface position
    # to the closest horizontal cell face (note: greater accuracy can be 
    # achieved by calculating the interface position a posteriori).
    #========================================================================
    
    Nx, Nz = np.shape(HH)
    xx_an, xx_ap = map(np.zeros, 2*([Nx+1, Nz], ))
    aa_p, aa_n = map(np.zeros, 2*([2, Nz], ))

    # identifying cells faces between mush/liquid
    np.putmask(xx_an[1:-1, :], \
          np.logical_and((HH[1:, :] > CC[1:, :]), (HH[:-1, :] <= CC[:-1, :])),\
           1)

    np.putmask(xx_ap[1:-1, :], \
           np.logical_and((HH[1:, :] <= CC[1:, :]), (HH[:-1, :] > CC[:-1, :])),\
           1)

    aa_p[1, :], aa_n[1, :] = zz_p[-1, :], zz_p[-1, :]

    # indices of interface positions (neg/pos)
    i_n = xx_an.nonzero()
    i_p = xx_ap.nonzero()

    # left interface positions
    for i in range(np.shape(i_n)[1]):
        aa_n[0][i_n[1][i]] = xx[i_n[0][i], i_n[1][i]]
    np.putmask(aa_n[0], aa_n[0] == 0, np.nan)

    # right interface positions
    for i in range(np.shape(i_p)[1]):
        aa_p[0][i_p[1][i]] = xx[i_p[0][i], i_p[1][i]]
    np.putmask(aa_p[0], aa_p[0] == 0, np.nan)


    return aa_n, aa_p