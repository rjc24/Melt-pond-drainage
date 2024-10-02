# Number of cores to use
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import phys_params as ppar
from scipy.sparse import diags

def phi_diff(HH, CC, St, Cr, EQN = 1):
    
    #========================================================================
    # Calculates solid fraction terms for use in jacobian
    #
    # Inputs:
    #    HH, CC - enthalpy, bulk salinity
    #    St, Cr - Stefan number, concentration ratio
    #    EQN - 1 (heat) or 2 (solute)
    #
    # Outputs:
    #    phi_diff - solid fraction terms
    #
    #========================================================================
    
    c_p = ppar.c_p
    
    s_term = np.zeros(np.shape(HH))
    d_s = np.zeros(np.shape(HH))
    phi_diff = np.zeros(np.shape(HH))

    np.putmask(s_term, HH > CC, 0)
    np.putmask(s_term, HH <= CC, ((c_p - 1)*CC - St - Cr + HH)**2 - \
    4*(Cr*(c_p - 1) - St)*(HH - CC))
    

    if EQN == 1:   # heat eqn
        np.putmask(d_s, HH <= CC, 2*((c_p - 1)*CC - St - Cr + HH) - \
             4*(Cr*(c_p - 1) - St))

    elif EQN == 2:
        np.putmask(d_s, HH <= CC, 2*(c_p - 1)*((c_p - 1)*CC - St - Cr + HH) + \
             4*(Cr*(c_p - 1) - St))
    

    np.putmask(phi_diff, HH > CC, 0)
    np.putmask(phi_diff, HH <= CC,  1/(2.0*(Cr*(c_p - 1) - St))*\
        (1 - 1/(2*(np.sqrt(s_term)))*d_s))

        
    return phi_diff


def J_resid(method, HH, CC, phi, xx, xin, zz, zin, dt, 
            k_bh, k_bv, cc_p, HH_BC, phi_BC, FF, FF_phi, 
            St, Cr, alp, D_n, G = None, EQN = 1):
    
    #========================================================================
    # Calculates jacobian (diffusive terms only)
    #
    # Inputs:
    #    method - 'BTCS' (backwards Euler) or 'Crank' (Crank-Nicholson)
    #    HH, CC, phi - enthalpy, bulk salinity, solid frac.
    #    xx, xin, zz, zin - spatial arrays
    #    dt - time step size
    #    k_bh, k_bv - thermal conductivity / solutal diffusivity at
    #                 horizontal, vertical cell faces
    #    cc_p - specific heat capacity / porosity
    #    HH_BC, phi_BC - Dirichlet conditions on enthalpy / bulk salinity,
    #                    solid fraction
    #    FF, FF_phi - Neumann conditions on enthalpy / bulk salinity,
    #                 solid fraction
    #    St, Cr - Stefan number, concentration ratio
    #    alp - boundary condition selector
    #    G - source term
    #    EQN - 1 (heat) or 2 (solute)
    #
    # Outputs:
    #    J -Jacobian matrix
    #
    #========================================================================
    
    # dimensionless quantities
    c_p = ppar.c_p
    k = ppar.k
    D = ppar.D
    D_s = ppar.D_s
    
    Nx, Nz = np.shape(xin)[0], np.shape(xin)[1]    # number of grid points
    N = Nz*Nx
    
    # Diagonal entries of matrix A
    main = np.zeros([Nx, Nz])
    left, right = np.zeros([Nx, Nz]), np.zeros([Nx, Nz])
    lower, upper = np.zeros([Nx, Nz-1]), np.zeros([Nx, Nz-1])
    
    # Defining theta 
    if method == 'BTCS':
        theta = 1        
    elif method == 'Crank':
        theta = 0.5
    elif method == 'FTCS':
        theta = 0

    # alp = 1 - Dirichlet, alp = 0 - Neumann
    alp_L, alp_R, alp_B, alp_T = alp[0], alp[1], alp[2], alp[3]

    # Dirichlet conditions
    # fixed enthalpies/salinities
    HH_L, HH_R, HH_B, HH_T = HH_BC[0], HH_BC[1], HH_BC[2], HH_BC[3]
    # fixed solid fractions
    phi_L, phi_R, phi_B, phi_T = phi_BC[0], phi_BC[1], phi_BC[2], phi_BC[3]
    
    # selecting parameters for equation (1 = heat, 2 = solute)
    if EQN == 1:
        lam = St   # Stefan number
        xi = c_p - 1   # heat cap.
        sig = k - 1   # conductivity
        ES = HH.copy()   # enthalpy
    elif EQN == 2:
        lam = -Cr   # concentration ratio
        xi = -1   # liquid frac.
#        sig = D_s - D   # diffusivity
        sig = 0   # diffusivity
        ES = CC.copy()
    
    # heat capacity (EQN 1) / liquid-fraction (EQN 2) at boundaries
    cc_pL, cc_pR, cc_pB, cc_pT = 1 + xi*phi_L, 1 + xi*phi_R, \
                                 1 + xi*phi_B, 1 + xi*phi_T

    # Neumann conditions
    # enthalpy fluxes
    FF_L, FF_R, FF_B, FF_T = FF[0], FF[1], FF[2], FF[3]
    # solid fraction fluxes
    FF_phi_L, FF_phi_R, FF_phi_B, FF_phi_T = \
        FF_phi[0], FF_phi[1], FF_phi[2], FF_phi[3]
    
#========================== enthalpy/salinity terms ==========================

    dHdx = np.zeros(np.shape(xx))
    dHdz = np.zeros(np.shape(zz))
    
    # spatial derivatives
    dHdx[1:-1, :] = (ES[1:, :]/cc_p[1:, :] - ES[:-1, :]/cc_p[:-1, :])/\
        (xin[1:, :] - xin[:-1, :])
    dHdz[:, 1:-1] = (ES[:, 1:]/cc_p[:, 1:] - ES[:, :-1]/cc_p[:, :-1])/\
        (zin[:, 1:] - zin[:, :-1])

    if np.all(FF_L) == 0:
        dHdx[0, :] = alp_L*(-8*HH_L[:]/cc_pL[:] + \
            9*ES[0, :]/cc_p[0, :] - ES[1, :]/cc_p[1, :])/\
            (6*(xin[0, :] - xx[0, :])) + (1 - alp_L)*FF_L[:]

    if np.all(FF_R) == 0:
        dHdx[-1, :] = alp_R*(8*HH_R[:]/cc_pR[:] - \
            9*ES[-1, :]/cc_p[-1, :] + ES[-2, :]/cc_p[-2, :])/\
            (6*(xx[-1, :] - xin[-1, :])) + (1 - alp_R)*FF_R[:]
            
    if np.all(FF_B) == 0:
        dHdz[:, 0] = alp_B*(-8*HH_B[:]/cc_pB[:] + \
            9*ES[:, 0]/cc_p[:, 0] - ES[:, 1]/cc_p[:, 1])/\
            (6*(zin[:, 0] - zz[:, 0])) + (1 - alp_B)*FF_B[:]

    if np.all(FF_T) == 0:
        dHdz[:, -1] = alp_T*(8*HH_T[:]/cc_pT[:] - \
            9*ES[:, -1]/cc_p[:, -1] + ES[:, -2]/cc_p[:, -2])/\
            (6*(zz[:, -1] - zin[:, -1])) + (1 - alp_T)*FF_T[:]

    
#=========================== solid fraction terms ============================
        
    dphidx = np.zeros(np.shape(xx))
    dphidz = np.zeros(np.shape(zz))
    dphidH = np.zeros(np.shape(xin))

    # spatial derivatives
    dphidx[1:-1, :] = (phi[1:, :]/cc_p[1:, :] - phi[:-1, :]/cc_p[:-1, :])/\
        (xin[1:, :] - xin[:-1, :])
    dphidz[:, 1:-1] = (phi[:, 1:]/cc_p[:, 1:] - phi[:, :-1]/cc_p[:, :-1])/\
        (zin[:, 1:] - zin[:, :-1])

    if np.all(FF_phi_L) == 0:
        dphidx[0, :] = alp_L*(-8*phi_L[:]/cc_pL[:] + \
            9*phi[0, :]/cc_p[0, :] - phi[1, :]/cc_p[1, :])/\
            (6*(xin[0, :] - xx[0, :])) + (1 - alp_L)*FF_phi_L[:]

    if np.all(FF_phi_R) == 0:
        dphidx[-1, :] = alp_R*(8*phi_R[:]/cc_pR[:] - \
            9*phi[-1, :]/cc_p[-1, :] + phi[-2, :]/cc_p[-2, :])/\
            (6*(xx[-1, :] - xin[-1, :])) + (1 - alp_R)*FF_phi_R[:]
            
    if np.all(FF_phi_B) == 0:
        dphidz[:, 0] = alp_B*(-8*phi_B[:]/cc_pB[:] + \
            9*phi[:, 0]/cc_p[:, 0] - phi[:, 1]/cc_p[:, 1])/\
            (6*(zin[:, 0] - zz[:, 0])) + (1 - alp_B)*FF_phi_B[:]

    if np.all(FF_phi_T) == 0:
        dphidz[:, -1] = alp_T*(8*phi_T[:]/cc_pT[:] - \
            9*phi[:, -1]/cc_p[:, -1] + phi[:, -2]/cc_p[:, -2])/\
            (6*(zz[:, -1] - zin[:, -1])) + (1 - alp_T)*FF_phi_T[:]

    dphidH[:] = phi_diff(HH, CC, St, Cr, EQN = EQN)
    

#=============================================================================
#=================================== main body ===============================
#=============================================================================

    # matrix
    main[1:-1, 1:-1] = -dt*theta*(sig/2*((dHdx[2:-1, 1:-1] - dHdx[1:-2, 1:-1] + \
                  lam*(dphidx[2:-1, 1:-1] - dphidx[1:-2, 1:-1]))*\
                  (zz[1:-1, 2:-1] - zz[1:-1, 1:-2]) + \
                  (dHdz[1:-1, 2:-1] - dHdz[1:-1, 1:-2] + \
                  lam*(dphidz[1:-1, 2:-1] - dphidz[1:-1, 1:-2]))*\
                  (xx[2:-1, 1:-1] - xx[1:-2, 1:-1])) + \
                  ((k_bh[2:-1, 1:-1]/(xin[2:, 1:-1] - xin[1:-1, 1:-1]) + \
                  k_bh[1:-2, 1:-1]/(xin[1:-1, 1:-1] - xin[:-2, 1:-1]))*\
                  (zz[1:-1, 2:-1] - zz[1:-1, 1:-2]) + \
                  (k_bv[1:-1, 2:-1]/(zin[1:-1, 2:] - zin[1:-1, 1:-1]) + \
                  k_bv[1:-1, 1:-2]/(zin[1:-1, 1:-1] - zin[1:-1, :-2]))*\
                  (xx[2:-1, 1:-1] - xx[1:-2, 1:-1]))*\
                  (xi*ES[1:-1, 1:-1]/(cc_p[1:-1, 1:-1]**2) - \
                  lam*(1/cc_p[1:-1, 1:-1] - xi*phi[1:-1, 1:-1]/\
                  (cc_p[1:-1, 1:-1]**2))))*dphidH[1:-1, 1:-1]

    left[1:-1, :] = -dt*theta*(sig/2*(-dHdx[1:-2, :] - lam*dphidx[1:-2, :]) + \
                  k_bh[1:-2, :]/(xin[1:-1, :] - xin[:-2, :])*\
                  (-xi*ES[:-2, :]/(cc_p[:-2, :]**2) + \
                  lam*(1/cc_p[:-2, :] - xi*phi[:-2, :]/(cc_p[:-2, :]**2))))*\
                  (zz[1:-1, 1:] - zz[1:-1, :-1])*dphidH[:-2, :]

    right[1:-1, :] = -dt*theta*(sig/2*(dHdx[2:-1, :] + lam*dphidx[2:-1, :]) + \
                k_bh[2:-1, :]/(xin[2:, :] - xin[1:-1, :])*\
                (-xi*ES[2:, :]/(cc_p[2:, :]**2) + \
                lam*(1/cc_p[2:, :] - xi*phi[2:, :]/(cc_p[2:, :]**2))))*\
                (zz[1:-1, 1:] - zz[1:-1, :-1])*dphidH[2:, :]

    lower[:, :-1] = -dt*theta*(sig/2*(-dHdz[:, 1:-2] - lam*dphidz[:, 1:-2]) + \
                  k_bv[:, 1:-2]/(zin[:, 1:-1] - zin[:, :-2])*\
                  (-xi*ES[:, :-2]/(cc_p[:, :-2]**2) + \
                  lam*(1/cc_p[:, :-2] - xi*phi[:, :-2]/(cc_p[:, :-2]**2))))*\
                  (xx[1:, 1:-1] - xx[:-1, 1:-1])*dphidH[:, :-2]
    
    upper[:, 1:] = -dt*theta*(sig/2*(dHdz[:, 2:-1] + lam*dphidz[:, 2:-1]) + \
                k_bv[:, 2:-1]/(zin[:, 2:] - zin[:, 1:-1])*\
                (-xi*ES[:, 2:]/(cc_p[:, 2:]**2) + \
                lam*(1/cc_p[:, 2:] - xi*phi[:, 2:]/(cc_p[:, 2:]**2))))*\
                (xx[1:, 1:-1] - xx[:-1, 1:-1])*dphidH[:, 2:]


#=============================================================================
#================================== boundaries ===============================
#=============================================================================

#=============================== Left boundary ===============================

    main[0, 1:-1] = -dt*theta*(sig*((dHdx[1, 1:-1]/2 - 9*(1 - alp_L)*dHdx[0, 1:-1]/8 + \
        lam*(dphidx[1, 1:-1]/2 - 9*(1 - alp_L)*dphidx[0, 1:-1]/8))*\
        (zz[0, 2:-1] - zz[0, 1:-2]) + \
        (dHdz[0, 2:-1]/2 - dHdz[0, 1:-2]/2 + \
        lam/2*(dphidz[0, 2:-1] - dphidz[0, 1:-2]))*\
        (xx[1, 1:-1] - xx[0, 1:-1])) + \
        ((k_bh[1, 1:-1]/(xin[1, 1:-1] - xin[0, 1:-1]) + \
        9*k_bh[0, 1:-1]*alp_L/(6*(xin[0, 1:-1] - xx[0, 1:-1])))*\
        (zz[0, 2:-1] - zz[0, 1:-2]) + \
        (k_bv[0, 2:-1]/(zin[0, 2:] - zin[0, 1:-1]) + \
        k_bv[0, 1:-2]/(zin[0, 1:-1] - zin[0, :-2]))*\
        (xx[1, 1:-1] - xx[0, 1:-1]))*\
        (xi*ES[0, 1:-1]/(cc_p[0, 1:-1]**2) - \
        lam*(1/cc_p[0, 1:-1] - xi*phi[0, 1:-1]/(cc_p[0, 1:-1]**2))))*dphidH[0, 1:-1]
            
            
    right[0, :] = -dt*theta*(sig*(dHdx[1, :]/2 + (1 - alp_L)*dHdx[0, :]/8 + \
        lam*(dphidx[1, :]/2 + (1 - alp_L)*dphidx[0, :]/8)) + \
        (k_bh[1, :]/(xin[1, :] - xin[0, :]) + \
        k_bh[0, :]*alp_L/(6*(xin[0, :] - xx[0, :])))*(-xi*ES[1, :]/(cc_p[1, :]**2) + \
        lam*(1/cc_p[1, :] - xi*phi[1, :]/(cc_p[1, :]**2))))*\
        (zz[0, 1:] - zz[0, :-1])*dphidH[1, :]
            

#=============================== Right boundary ==============================


    main[-1, 1:-1] = -dt*theta*(sig*((9*(1 - alp_R)*dHdx[-1, 1:-1]/8 - dHdx[-2, 1:-1]/2 + \
        lam*(9*(1 - alp_R)*dphidx[-1, 1:-1]/8 - dphidx[-2, 1:-1]/2))*\
        (zz[-1, 2:-1] - zz[-1, 1:-2]) + \
        (dHdz[-1, 2:-1]/2 - dHdz[-1, 1:-2]/2 + \
        lam/2*(dphidz[-1, 2:-1] - dphidz[-1, 1:-2]))*\
        (xx[-1, 1:-1] - xx[-2, 1:-1])) + \
        ((9*k_bh[-1, 1:-1]*alp_R/(6*(xx[-1, 1:-1] - xin[-1, 1:-1])) + \
        k_bh[-2, 1:-1]/(xin[-1, 1:-1] - xin[-2, 1:-1]))*\
        (zz[-1, 2:-1] - zz[-1, 1:-2]) + \
        (k_bv[-1, 2:-1]/(zin[-1, 2:] - zin[-1, 1:-1]) + \
        k_bv[-1, 1:-2]/(zin[-1, 1:-1] - zin[-1, :-2]))*\
        (xx[-1, 1:-1] - xx[-2, 1:-1]))*\
        (xi*ES[-1, 1:-1]/(cc_p[-1, 1:-1]**2) - \
        lam*(1/cc_p[-1, 1:-1] - xi*phi[-1, 1:-1]/(cc_p[-1, 1:-1]**2))))*dphidH[-1, 1:-1]
    
    
    left[-1, :] = -dt*theta*(sig*(-(1 - alp_R)*dHdx[-1, :]/8 - dHdx[-2, :]/2 + \
        lam*(-(1 - alp_R)*dphidx[-1, :]/8 - dphidx[-2, :]/2)) - \
        (k_bh[-2, :]/(xin[-1, :] - xin[-2, :]) + \
        k_bh[-1, :]*alp_R/(6*(xx[-1, :] - xin[-1, :])))*(xi*ES[-2, :]/(cc_p[-2, :]**2) - \
        lam*(1/cc_p[-2, :] - xi*phi[-2, :]/(cc_p[-2, :]**2))))*\
        (zz[-1, 1:] - zz[-1, :-1])*dphidH[-2, :]
        
        
#============================== Lower boundary ===============================

    main[1:-1, 0] = -dt*theta*(sig*((dHdx[2:-1, 0]/2 - dHdx[1:-2, 0]/2 + \
        lam/2*(dphidx[2:-1, 0] - dphidx[1:-2, 0]))*\
        (zz[1:-1, 1] - zz[1:-1, 0]) + \
        (dHdz[1:-1, 1]/2 - 9*(1 - alp_B)*dHdz[1:-1, 0]/8 + \
        lam*(dphidz[1:-1, 1]/2 - 9*(1 - alp_B)*dphidz[1:-1, 0]/8))*\
        (xx[2:-1, 0] - xx[1:-2, 0])) + \
        ((k_bh[2:-1, 0]/(xin[2:, 0] - xin[1:-1, 0]) + \
        k_bh[1:-2, 0]/(xin[1:-1, 0] - xin[:-2, 0]))*\
        (zz[1:-1, 1] - zz[1:-1, 0]) + \
        (k_bv[1:-1, 1]/(zin[1:-1, 1] - zin[1:-1, 0]) + \
        9*k_bv[1:-1, 0]*alp_B/(6*(zin[1:-1, 0] - zz[1:-1, 0])))*\
        (xx[2:-1, 0] - xx[1:-2, 0]))*\
        (xi*ES[1:-1, 0]/(cc_p[1:-1, 0]**2) - \
        lam*(1/cc_p[1:-1, 0] - xi*phi[1:-1, 0]/(cc_p[1:-1, 0]**2))))*dphidH[1:-1, 0]
            
            
    upper[:, 0] = -dt*theta*(sig*(dHdz[:, 1]/2 + (1 - alp_B)*dHdz[:, 0]/8 + \
        lam*(dphidz[:, 1]/2 + (1 - alp_B)*dphidz[:, 0]/8)) + \
        (k_bv[:, 1]/(zin[:, 1] - zin[:, 0]) + \
        k_bv[:, 0]*alp_B/(6*(zin[:, 0] - zz[:, 0])))*(-xi*ES[:, 1]/(cc_p[:, 1]**2) + \
        lam*(1/cc_p[:, 1] - xi*phi[:, 1]/(cc_p[:, 1]**2))))*\
        (xx[1:, 0] - xx[:-1, 0])*dphidH[:, 1]
            

#=============================== Upper boundary ==============================


    main[1:-1, -1] = -dt*theta*(sig*((dHdx[2:-1, -1]/2 - dHdx[1:-2, -1]/2 + \
        lam/2*(dphidx[2:-1, -1] - dphidx[1:-2, -1]))*\
        (zz[1:-1, -1] - zz[1:-1, -2]) + \
        (9*(1 - alp_T)*dHdz[1:-1, -1]/8 - dHdz[1:-1, -2]/2 + \
        lam*(9*(1 - alp_T)*dphidz[1:-1, -1]/8 - dphidz[1:-1, -2]/2))*\
        (xx[2:-1, -1] - xx[1:-2, -1])) + \
        ((k_bh[2:-1, -1]/(xin[2:, -1] - xin[1:-1, -1]) + \
        k_bh[1:-2, -1]/(xin[1:-1, -1] - xin[:-2, -1]))*\
        (zz[1:-1, -1] - zz[1:-1, -2]) + \
        (9*k_bv[1:-1, -1]*alp_T/(6*(zz[1:-1, -1] - zin[1:-1, -1])) + \
        k_bv[1:-1, -2]/(zin[1:-1, -1] - zin[1:-1, -2]))*\
        (xx[2:-1, -1] - xx[1:-2, -1]))*\
        (xi*ES[1:-1, -1]/(cc_p[1:-1, -1]**2) - \
        lam*(1/cc_p[1:-1, -1] - xi*phi[1:-1, -1]/(cc_p[1:-1, -1]**2))))*dphidH[1:-1, -1]
    
    
    lower[:, -1] = -dt*theta*(sig*(-(1 - alp_T)*dHdz[:, -1]/8 - dHdz[:, -2]/2 + \
        lam*(-(1 - alp_T)*dphidz[:, -1]/8 - dphidz[:, -2]/2)) - \
        (k_bv[:, -2]/(zin[:, -1] - zin[:, -2]) + \
        k_bv[:, -1]*alp_T/(6*(zz[:, -1] - zin[:, -1])))*(xi*ES[:, -2]/(cc_p[:, -2]**2) - \
        lam*(1/cc_p[:, -2] - xi*phi[:, -2]/(cc_p[:, -2]**2))))*\
        (xx[1:, -1] - xx[:-1, -1])*dphidH[:, -2]

                                               
#=============================================================================
#================================== corners ==================================
#=============================================================================

    # bottom-left
    main[0, 0] = -dt*theta*(sig*((dHdx[1, 0]/2 - 9*(1 - alp_L)*dHdx[0, 0]/8 + \
        lam*(dphidx[1, 0]/2 - 9*(1 - alp_L)*dphidx[0, 0]/8))*\
        (zz[0, 1] - zz[0, 0]) + \
        (dHdz[0, 1]/2 - 9*(1 - alp_B)*dHdz[0, 0]/8 + \
        lam*(dphidz[0, 1]/2 - 9*(1 - alp_B)*dphidz[0, 0]/8))*\
        (xx[1, 0] - xx[0, 0])) + \
        ((k_bh[1, 0]/(xin[1, 0] - xin[0, 0]) + \
        9*k_bh[0, 0]*alp_L/(6*(xin[0, 0] - xx[0, 0])))*\
        (zz[0, 1] - zz[0, 0]) + \
        (k_bv[0, 1]/(zin[0, 1] - zin[0, 0]) + \
        9*k_bv[0, 0]*alp_B/(6*(zin[0, 0] - zz[0, 0])))*\
        (xx[1, 0] - xx[0, 0]))*\
        (xi*ES[0, 0]/(cc_p[0, 0]**2) - \
        lam*(1/cc_p[0, 0] - xi*phi[0, 0]/(cc_p[0, 0]**2))))*dphidH[0, 0]

    # bottom-right
    main[-1, 0] = -dt*theta*(sig*((9*(1 - alp_R)*dHdx[-1, 0]/8 - dHdx[-2, 0]/2 + \
        lam*(9*(1 - alp_R)*dphidx[-1, 0]/8 - dphidx[-2, 0]/2))*\
        (zz[-1, 1] - zz[-1, 0]) + \
        (dHdz[-1, 1]/2 - 9*(1 - alp_B)*dHdz[-1, 0]/8 + \
        lam*(dphidz[-1, 1]/2 - 9*(1 - alp_B)*dphidz[-1, 0]/8))*\
        (xx[-1, 0] - xx[-2, 0])) + \
        ((9*k_bh[-1, 0]*alp_R/(6*(xx[-1, 0] - xin[-1, 0])) + \
        k_bh[-2, 0]/(xin[-1, 0] - xin[-2, 0]))*\
        (zz[-1, 1] - zz[-1, 0]) + \
        (k_bv[-1, 1]/(zin[-1, 1] - zin[-1, 0]) + \
        9*k_bv[-1, 0]*alp_B/(6*(zin[-1, 0] - zz[-1, 0])))*\
        (xx[-1, 0] - xx[-2, 0]))*\
        (xi*ES[-1, 0]/(cc_p[-1, 0]**2) - \
        lam*(1/cc_p[-1, 0] - xi*phi[-1, 0]/(cc_p[-1, 0]**2))))*dphidH[-1, 0]

    # top-left
    main[0, -1] = -dt*theta*(sig*((dHdx[1, -1]/2 - 9*(1 - alp_L)*dHdx[0, -1]/8 + \
        lam*(dphidx[1, -1]/2 - 9*(1 - alp_L)*dphidx[0, -1]/8))*\
        (zz[0, -1] - zz[0, -2]) + \
        (9*(1 - alp_T)*dHdz[0, -1]/8 - dHdz[0, -2]/2 + \
        lam*(9*(1 - alp_T)*dphidz[0, -1]/8 - dphidz[0, -2]/2))*\
        (xx[1, -1] - xx[0, -1])) + \
        ((k_bh[1, -1]/(xin[1, -1] - xin[0, -1]) + \
        9*k_bh[0, -1]*alp_L/(6*(xin[0, -1] - xx[0, -1])))*\
        (zz[0, -1] - zz[0, -2]) + \
        (9*k_bv[0, -1]*alp_T/(6*(zz[0, -1] - zin[0, -1])) + \
        k_bv[0, -2]/(zin[0, -1] - zin[0, -2]))*\
        (xx[1, -1] - xx[0, -1]))*\
        (xi*ES[0, -1]/(cc_p[0, -1]**2) - \
        lam*(1/cc_p[0, -1] - xi*phi[0, -1]/(cc_p[0, -1]**2))))*dphidH[0, -1]
        
    # top-right
    main[-1, -1] = -dt*theta*(sig*((9*(1 - alp_R)*dHdx[-1, -1]/8 - dHdx[-2, -1]/2 + \
        lam*(9*(1 - alp_R)*dphidx[-1, -1]/8 - dphidx[-2, -1]/2))*\
        (zz[-1, -1] - zz[-1, -2]) + \
        (9*(1 - alp_T)*dHdz[-1, -1]/8 - dHdz[-1, -2]/2 + \
        lam*(9*(1 - alp_T)*dphidz[-1, -1]/8 - dphidz[-1, -2]/2))*\
        (xx[-1, -1] - xx[-2, -1])) + \
        ((9*k_bh[-1, -1]*alp_R/(6*(xx[-1, -1] - xin[-1, -1])) + \
        k_bh[-2, -1]/(xin[-1, -1] - xin[-2, -1]))*\
        (zz[-1, -1] - zz[-1, -2]) + \
        (9*k_bv[-1, -1]*alp_T/(6*(zz[-1, -1] - zin[-1, -1])) + \
        k_bv[-1, -2]/(zin[-1, -1] - zin[-1, -2]))*\
        (xx[-1, -1] - xx[-2, -1]))*\
        (xi*ES[-1, -1]/(cc_p[-1, -1]**2) - \
        lam*(1/cc_p[-1, -1] - xi*phi[-1, -1]/(cc_p[-1, -1]**2))))*dphidH[-1, -1]


#=============================================================================
#============================== compiling matrix =============================
#=============================================================================    

    main_elem = np.ravel(main, order = 'F')
    left_elem = np.ravel(left, order = 'F')[1:]
    right_elem = np.ravel(right, order = 'F')[:-1]
    lower_elem = np.ravel(lower, order = 'F')
    upper_elem = np.ravel(upper, order = 'F')

    # Computing J
    J = diags(
        diagonals = [main_elem, right_elem, left_elem, upper_elem, lower_elem],
        offsets = [0, 1, -1, Nx, -Nx], 
        shape = (N, N),
        format = 'csr'
        )

    return J