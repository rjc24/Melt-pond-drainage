# Number of cores to use
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import phys_params as ppar
from Enthalpy_functions import Var_bounds, Perm_calc
from elliptic_solve_sparse import velocities


def MUSCL(CC, u_b, w_b, xx, xin, zz, zin, CC_BC, FF_CC, alp):
    #=========================================================================
    # Calculates quantities at cell faces using MUSCL-style limiter
    #
    # Inputs:
    #    CC - cell-centred quantity
    #    u_b, w_b - horizontal, vertical velocities (cell faces)
    #    xx, xin, zz, zin - spatial arrays
    #    CC_BC, FF_CC - Dirichlet, Neumann conditions on quantity
    #    alp - boundary condition selctor
    #
    # Outputs:
    #    CC_fh, CC_fv - quantity at cell faces

    #
    #=========================================================================
    
    Nx, Nz = np.shape(xin)[0], np.shape(xin)[1]    # number of grid points

    CC_bh, CC_bv = Var_bounds(CC, xx, xin, zz, zin, CC_BC, FF_CC, alp, EQN = 1)
    
    alp_L, alp_R, alp_B, alp_T = alp[0], alp[1], alp[2], alp[3]
    FF_CCL, FF_CCR, FF_CCB, FF_CCT = FF_CC[0], FF_CC[1], FF_CC[2], FF_CC[3]

    C_GL, C_GR = map(np.zeros, 2*(Nz, ))
    C_GB, C_GT = map(np.zeros, 2*(Nx, ))   # ghost cells
    r_h, r_v = map(np.zeros, 2*([Nx, Nz], ))   # cell ratios
    sig_h, sig_v = map(np.zeros, 2*([Nx, Nz], ))   # limiter
    CC_l, CC_r = map(np.zeros, 2*([Nx+1, Nz], ))   # left, right states
    CC_b, CC_t = map(np.zeros, 2*([Nx, Nz+1], ))   # bottom, top states
    CC_fh = np.zeros([Nx+1, Nz])
    CC_fv = np.zeros([Nx, Nz+1])

    # ghost cells
    C_GL[:] = alp_L*(2*CC_bh[0, :] - CC[0, :]) + \
              (1 - alp_L)*(CC[0, :] - 2*(xin[0, :] - xx[0, :])*FF_CCL[:])
    C_GR[:] = alp_R*(2*CC_bh[-1, :] - CC[-1, :]) + \
              (1 - alp_R)*(CC[-1, :] + 2*(xx[-1, :] - xin[-1, :])*FF_CCR[:])
    C_GB[:] = alp_B*(2*CC_bv[:, 0] - CC[:, 0]) + \
              (1 - alp_B)*(CC[:, 0] - 2*(zin[:, 0] - zz[:, 0])*FF_CCB[:])
    C_GT[:] = alp_T*(2*CC_bv[:, -1] - CC[:, -1]) + \
              (1 - alp_T)*(CC[:, -1] + 2*(zz[:, -1] - zin[:, -1])*FF_CCT[:])

    # consecutive-cell slope ratios
    np.putmask(r_h[1:-1, :], CC[2:, :] != CC[1:-1, :], \
               (CC[1:-1, :] - CC[:-2, :])/(CC[2:, :] - CC[1:-1, :]))
    np.putmask(r_h[0, :], CC[1, :] != CC[0, :], \
               (CC[0, :] - C_GL[:])/(CC[1, :] - CC[0, :]))
    np.putmask(r_h[-1, :], C_GR[:] != CC[-1, :], \
               (CC[-1, :] - CC[-2, :])/(C_GR[:] - CC[-1, :]))
    np.putmask(r_v[:, 1:-1], CC[:, 2:] != CC[:, 1:-1], \
               (CC[:, 1:-1] - CC[:, :-2])/(CC[:, 2:] - CC[:, 1:-1]))
    np.putmask(r_v[:, 0], CC[:, 1] != CC[:, 0], \
               (CC[:, 0] - C_GB[:])/(CC[:, 1] - CC[:, 0]))
    np.putmask(r_v[:, -1], C_GT[:] != CC[:, -1], \
               (CC[:, -1] - CC[:, -2])/(C_GT[:] - CC[:, -1]))

    # limiter
    sig_h[:, :] = np.fmax(0, np.fmin(np.fmin(2*r_h[:, :], (r_h[:, :] + 1)/2), 2))
    sig_v[:, :] = np.fmax(0, np.fmin(np.fmin(2*r_v[:, :], (r_v[:,  :] + 1)/2), 2))

    # constructing left/right, top/bottom states
    CC_l[1:-1, :] = CC[:-1, :] + sig_h[:-1, :]*(CC[1:, :] - CC[:-1, :])/2
    CC_l[0, :] = CC_bh[0, :]
    CC_l[-1, :] = CC[-1, :] + sig_h[-1, :]*(C_GR[:] - CC[-1, :])/2
    CC_r[:-2, :] = CC[:-1, :] - sig_h[:-1, :]*(CC[1:, :] - CC[:-1, :])/2
    CC_r[-2, :] = CC[-1, :] - sig_h[-1, :]*(C_GR[:] - CC[-1, :])/2
    CC_r[-1, :] = CC_bh[-1, :]

    CC_b[:, 1:-1] = CC[:, :-1] + sig_v[:, :-1]*(CC[:, 1:] - CC[:, :-1])/2
    CC_b[:, 0] = CC_bv[:, 0]
    CC_b[:, -1] = CC[:, -1] + sig_v[:, -1]*(C_GT[:] - CC[:, -1])/2
    CC_t[:, :-2] = CC[:, :-1] - sig_v[:, :-1]*(CC[:, 1:] - CC[:, :-1])/2
    CC_t[:, -2] = CC[:, -1] - sig_v[:, -1]*(C_GT[:] - CC[:, -1])/2
    CC_t[:, -1] = CC_bv[:, -1]


    # upwinding face values
    CC_fh[0, :] = np.fmax(np.sign(u_b[0, :]), 0)*CC_bh[0, :] - \
                  np.fmin(np.sign(u_b[0, :]), 0)*CC_r[0, :] + \
                  (1 - abs(np.sign(u_b[0, :])))*CC_bh[0, :]
    CC_fh[1:-1, :] = np.fmax(np.sign(u_b[1:-1, :]), 0)*CC_l[1:-1, :] - \
                     np.fmin(np.sign(u_b[1:-1, :]), 0)*CC_r[1:-1, :] + \
                     (1 - abs(np.sign(u_b[1:-1, :])))*(CC_l[1:-1, :] + CC_r[1:-1, :])/2
    CC_fh[-1, :] = np.fmax(np.sign(u_b[-1, :]), 0)*CC_l[-1, :] - \
                   np.fmin(np.sign(u_b[-1, :]), 0)*CC_bh[-1, :] + \
                   (1 - abs(np.sign(u_b[-1, :])))*CC_bh[-1, :]
    
    CC_fv[:, 0] = np.fmax(np.sign(w_b[:, 0]), 0)*CC_bv[:, 0] - \
                  np.fmin(np.sign(w_b[:, 0]), 0)*CC_t[:, 0] + \
                  (1 - abs(np.sign(w_b[:, 0])))*CC_bv[:, 0]
    CC_fv[:, 1:-1] = np.fmax(np.sign(w_b[:, 1:-1]), 0)*CC_b[:, 1:-1] - \
                     np.fmin(np.sign(w_b[:, 1:-1]), 0)*CC_t[:, 1:-1] + \
                   (1 - abs(np.sign(w_b[:, 1:-1])))*(CC_b[:, 1:-1] + CC_t[:, 1:-1])/2
    CC_fv[:, -1] = np.fmax(np.sign(w_b[:, -1]), 0)*CC_b[:, -1] - \
                   np.fmin(np.sign(w_b[:, -1]), 0)*CC_bv[:, -1] + \
                   (1 - abs(np.sign(w_b[:, -1])))*CC_bv[:, -1]

    return CC_fh, CC_fv


