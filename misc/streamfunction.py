# Number of cores to use
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from elliptic_solve_sparse import p_solve, velocities
from Enthalpy_functions import Var_bounds, Perm_calc
import phys_params as ppar

def jacobian_perm(TT, phi, xin, xx, zin, zz, TT_BC, FF_TT, phi_BC, FF_phi, 
                  alp_T, alp_phi, Da_h):
    
    #========================================================================
    # Calculates jacobian J(T, Pi) where T is temperature, Pi is permeability
    #========================================================================

    Nx, Nz = np.shape(TT)

    dTdx, dTdz = map(np.zeros, 2*([Nx, Nz], ))
    dPidx, dPidz = map(np.zeros, 2*([Nx, Nz], ))
    jac = np.zeros([Nx, Nz])

    # cell face values
    TT_bh, TT_bv = Var_bounds(TT, xx, xin, zz, zin, TT_BC, FF_TT, alp_T, EQN = 1)
    Pi, Pi_bh, Pi_bv = Perm_calc(phi, xx, xin, zz, zin, phi_BC, FF_phi, alp_phi, Da_h)

    # temperature gradients
    dTdx[:, :] = (TT_bh[1:, :] - TT_bh[:-1, :])/(xx[1:, :] - xx[:-1, :])
    dTdz[:, :] = (TT_bv[:, 1:] - TT_bv[:, :-1])/(zz[:, 1:] - zz[:, :-1])

    # permeability gradients
    dPidx[:, :] = (Pi_bh[1:, :] - Pi_bh[:-1, :])/(xx[1:, :] - xx[:-1, :])
    dPidz[:, :] = (Pi_bv[:, 1:] - Pi_bv[:, :-1])/(zz[:, 1:] - zz[:, :-1])

    # jacobian                  
    jac[:, :] = dTdx[:, :]*dPidz[:, :] - dTdz[:, :]*dPidx[:, :]
    
    return jac

def streamfunc(phi, pp, xin, xx, zin, zz, phi_BC, FF_phi, pp_BC, FF_pp, 
               alp_phi, alp_p, Pe, Da_h):
    
    #========================================================================
    # Calculates the streamfunction psi, given solid fraction (phi), 
    # pressure (pp) fields at a given time.
    #========================================================================
    
    Nx, Nz = np.shape(pp)

    # constants
    d = ppar.d   # Hele-shaw width
    Pi_0 = ppar.Pi_0   # reference permeability
    
    # setting up streamfunction BCs
    psi_L, psi_R = map(np.zeros, 2*(Nz, ))
    psi_B, psi_T = map(np.zeros, 2*(Nx, ))
    
    # boundary conditions
    alp_psi = [1, 1, 0, 0]

    # calculating cell-face values of phi, Pi
    phi_bh, phi_bv = Var_bounds(phi, xx, xin, zz, zin, phi_BC, FF_phi, alp_phi, EQN = 2)
    Pi, Pi_bh, Pi_bv = Perm_calc(phi, xx, xin, zz, zin, phi_BC, FF_phi, alp_phi, Da_h)

    # calculating velocities at cell faces
    u_b, w_b = velocities(xx, xin, zz, zin, pp, Pi_bh, Pi_bv, pp_BC, FF_pp, 
                          alp_p, Pe)
    
    dx = xx[1, 0] - xx[0, 0]   # horizontal spacing
    w_rsum = np.sum(w_b, axis = 0)
    w_R = dx*(w_rsum[1:] + w_rsum[:-1])/2   # vertical velocity at lateral boundaries
    psi_L[:], psi_R[:] = -w_R/2, w_R/2   # BCs on psi

    # converting phi BCs to Pi BCs
    FF_phi_L, FF_phi_R, FF_phi_B, FF_phi_T = \
        FF_phi[0], FF_phi[1], FF_phi[2], FF_phi[3]

    # setting psi BCs
    psi_BC = [psi_L, psi_R, psi_B, psi_T]
    FF_psi = FF_pp.copy()

   # setting Pi BCs
    Pi_L, Pi_R, Pi_B, Pi_T = \
        Pi_bh[0, :], Pi_bh[-1, :], Pi_bv[:, 0], Pi_bv[:, -1]
    FF_PiL = d**4*phi_bh[0, :]**2*(1 - phi_bh[0, :])**2/\
        (d**2*phi_bh[0, :]**2 + 12*Pi_0*(1 - phi_bh[0, :])**3)**2*FF_phi_L[:]
    FF_PiR = d**4*phi_bh[-1, :]**2*(1 - phi_bh[-1, :])**2/\
        (d**2*phi_bh[-1, :]**2 + 12*Pi_0*(1 - phi_bh[-1, :])**3)**2*FF_phi_R[:]
    FF_PiB = d**4*phi_bv[:, 0]**2*(1 - phi_bv[:, 0])**2/\
        (d**2*phi_bv[:, 0]**2 + 12*Pi_0*(1 - phi_bv[:, 0])**3)**2*FF_phi_B[:]
    FF_PiT = d**4*phi_bv[:, -1]**2*(1 - phi_bv[:, -1])**2/\
        (d**2*phi_bv[:, -1]**2 + 12*Pi_0*(1 - phi_bv[:, -1])**3)**2*FF_phi_T[:]

    Pi_BC = [Pi_L, Pi_R, Pi_B, Pi_T]
    FF_Pi = [FF_PiL, FF_PiR, FF_PiB, FF_PiT]

    # calculating jacobian
    jac = jacobian_perm(pp, phi, xin, xx, zin, zz, pp_BC, FF_pp, 
                        phi_BC, FF_phi, alp_p, alp_phi, Da_h)
    G_0 = Pe*jac

    ones_h, ones_v = np.ones([Nx+1, Nz]), np.ones([Nx, Nz+1])

    # calculating streamfunction
    psi, u, w = p_solve(xx, xin, zz, zin, ones_h, ones_v, psi_BC, FF_psi, alp_psi, 
                  Pe, G = G_0)


    return psi
