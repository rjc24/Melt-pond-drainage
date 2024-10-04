# Number of cores to use
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import phys_params as ppar
import num_params as npar


def b_theta(method, HH_0, HH, phi_0, phi, xx, xin, zz, zin, dt, 
            TT_Fx, TT_Fz, cc_p0, cc_p, k_bh0, k_bh, k_bv0, k_bv, 
            HH_BC0, HH_BC, phi_BC0, phi_BC, FF_0, FF, FF_phi_0, FF_phi, 
            St, Cr, alp, D_n, G = None, EQN = 1):

    #========================================================================
    # Function which creates the right hand side vector for the heat / 
    # solute conservation equation
    #
    # Inputs:
    #    method - 'BTCS' (backwards Euler) or 'Crank' (Crank-Nicholson)
    #    HH_0, HH - enthalpy / bulk salinity field at times n, n+1
    #    phi_0, phi - solid fraction field at times n, n+1
    #    xx, xin, zz, zin - spatial arrays
    #    dt - time step
    #    TT_Fx, TT_Fz - advective fluxes (temperature / liquid concentration)
    #    cc_p0, cc_p - specific heat capacity / porosity at times n, n+1
    #    k_bh0, k_bh, k_bv0, k_bv - thermal conductivity / solutal diffusivity
    #                               at times n, n+1
    #    HH_BC0, HH_BC - Dirichlet BCs for enthalpy / bulk salinity
    #    phi_BC0, phi_BC - Dirichlet conditions for solid fraction
    #    FF_0, FF - Neumann conditions for enthalpy / bulk salinity 
    #    FF_phi_0, FF_phi - Neumann conditions for solid fraction
    #    St, Cr - Stefan number, concentration ratio
    #    alp - boundary conditions
    #    D_n - numerical diffusivity
    #    G - source term
    #    EQN - heat(1) / solute(2)
    #
    # Outputs:
    #    b - vector
    #
    #========================================================================
    
    # dimensionless quantities
    c_p = ppar.c_p
    
    # numerical diffusivity
    if EQN == 1:
        D_n = 0
    
    
    # Defining number of grid points
    Nx, Nz = np.shape(xin)[0], np.shape(xin)[1]    # number of grid points
    
    # Ensuring supported method has been chosen
    if not method in ('Crank', 'BTCS', 'FTCS'):
        raise SyntaxError('Must choose either Crank, BTCS or FTCS for method.')
        return

        
    # Defining theta 
    if method == 'BTCS':
        theta = 1        
    elif method == 'Crank':
        theta = 0.5
    elif method == 'FTCS':
        theta = 0

#=============================================================================
#============================ boundary conditions ============================
#=============================================================================

    # boundary condition type
    # alp = 1 - Dirichlet, alp = 0 - Neumann
    alp_L, alp_R, alp_B, alp_T = alp[0], alp[1], alp[2], alp[3]
    
    # Dirichlet conditions
    # fixed enthalpies
    HH_L0, HH_R0, HH_B0, HH_T0 = HH_BC0[0], HH_BC0[1], HH_BC0[2], HH_BC0[3]
    HH_L, HH_R, HH_B, HH_T = HH_BC[0], HH_BC[1], HH_BC[2], HH_BC[3]
    # fixed solid fractions
    phi_L0, phi_R0, phi_B0, phi_T0 = phi_BC0[0], phi_BC0[1], phi_BC0[2], phi_BC0[3]
    phi_L, phi_R, phi_B, phi_T = phi_BC[0], phi_BC[1], phi_BC[2], phi_BC[3]
    
    if EQN == 1:
        # s. heat capacity at boundaries
        cc_pL0, cc_pR0, cc_pB0, cc_pT0 = 1 + (c_p - 1)*phi_L0, 1 + (c_p - 1)*phi_R0, \
                                 1 + (c_p - 1)*phi_B0, 1 + (c_p - 1)*phi_T0
        cc_pL, cc_pR, cc_pB, cc_pT = 1 + (c_p - 1)*phi_L, 1 + (c_p - 1)*phi_R, \
                                 1 + (c_p - 1)*phi_B, 1 + (c_p - 1)*phi_T
    elif EQN == 2:
        # liquid-fraction at boundaries
        cc_pL0, cc_pR0, cc_pB0, cc_pT0 = 1 - phi_L0, 1 - phi_R0, \
                                         1 - phi_B0, 1 - phi_T0
        cc_pL, cc_pR, cc_pB, cc_pT = 1 - phi_L, 1 - phi_R, \
                                     1 - phi_B, 1 - phi_T

    # Neumann conditions
    # enthalpy fluxes
    FF_L0, FF_R0, FF_B0, FF_T0 = FF_0[0], FF_0[1], FF_0[2], FF_0[3]
    FF_L, FF_R, FF_B, FF_T = FF[0], FF[1], FF[2], FF[3]
    # solid fraction fluxes
    FF_phi_L0, FF_phi_R0, FF_phi_B0, FF_phi_T0 = \
        FF_phi_0[0], FF_phi_0[1], FF_phi_0[2], FF_phi_0[3]
    FF_phi_L, FF_phi_R, FF_phi_B, FF_phi_T = \
        FF_phi[0], FF_phi[1], FF_phi[2], FF_phi[3]


    # Initializing b
    b, time_p, advec, diffus, diffus_p, s_frac, s_frac_p, n_diffus, n_diffus_p = \
        map(np.zeros, 9*([Nx, Nz], ))
    


#=============================================================================
#================================= main body =================================
#=============================================================================

#================================= time terms ================================

    time_p[:, :] = (xx[1:, :] - xx[:-1, :])*(zz[:, 1:] - zz[:, :-1])*HH_0[:, :]


#============================== advection terms ==============================

    advec[:, :] = dt*((zz[:, 1:] - zz[:, :-1])*(TT_Fx[1:, :] - TT_Fx[:-1, :]) + \
        (xx[1:, :] - xx[:-1, :])*(TT_Fz[:, 1:] - TT_Fz[:, :-1]))


#============================== diffusion terms ==============================
                
    # spatial derivatives

    dHdr_0, dHdr = map(np.zeros, 2*(np.shape(xx), ))
    dHdz_0, dHdz = map(np.zeros, 2*(np.shape(zz), ))
    
    dHdr_0[1:-1, :] = (HH_0[1:, :]/cc_p0[1:, :] - HH_0[:-1, :]/cc_p0[:-1, :])/\
                      (xin[1:, :] - xin[:-1, :])
    dHdr[1:-1, :] = (HH[1:, :]/cc_p[1:, :] - HH[:-1, :]/cc_p[:-1, :])/\
                    (xin[1:, :] - xin[:-1, :])
    dHdz_0[:, 1:-1] = (HH_0[:, 1:]/cc_p0[:, 1:] - HH_0[:, :-1]/cc_p0[:, :-1])/\
                      (zin[:, 1:] - zin[:, :-1])
    dHdz[:, 1:-1] = (HH[:, 1:]/cc_p[:, 1:] - HH[:, :-1]/cc_p[:, :-1])/\
                    (zin[:, 1:] - zin[:, :-1])

    if np.all(FF_L0) == 0:
        dHdr_0[0, :] = alp_L*(-8*HH_L0[:]/cc_pL0[:] + \
            9*HH_0[0, :]/cc_p0[0, :] - HH_0[1, :]/cc_p0[1, :])/\
            (6*(xin[0, :] - xx[0, :])) + (1 - alp_L)*FF_L0[:]
        dHdr[0, :] = -alp_L*8*HH_L[:]/cc_pL[:]/(6*(xin[0, :] - xx[0, :])) + \
            (1 - alp_L)*FF_L[:]

    if np.all(FF_R0) == 0:
        dHdr_0[-1, :] = alp_R*(8*HH_R0[:]/cc_pR0[:] - \
            9*HH_0[-1, :]/cc_p0[-1, :] + HH_0[-2, :]/cc_p0[-2, :])/\
            (6*(xx[-1, :] - xin[-1, :])) + (1 - alp_R)*FF_R0[:]
        dHdr[-1, :] = alp_R*8*HH_R[:]/cc_pR[:]/(6*(xx[-1, :] - xin[-1, :])) + \
            (1 - alp_R)*FF_R[:]

    if np.all(FF_B0) == 0:
        dHdz_0[:, 0] = alp_B*(-8*HH_B0[:]/cc_pB0[:] + \
            9*HH_0[:, 0]/cc_p0[:, 0] - HH_0[:, 1]/cc_p0[:, 1])/\
            (6*(zin[:, 0] - zz[:, 0])) + (1 - alp_B)*FF_B0[:]
        dHdz[:, 0] = -alp_B*8*HH_B[:]/cc_pB[:]/(6*(zin[:, 0] - zz[:, 0])) + \
            (1 - alp_B)*FF_B[:]

    if np.all(FF_T0) == 0:
        dHdz_0[:, -1] = alp_T*(8*HH_T0[:]/cc_pT0[:] - \
            9*HH_0[:, -1]/cc_p0[:, -1] + HH_0[:, -2]/cc_p0[:, -2])/\
            (6*(zz[:, -1] - zin[:, -1])) + (1 - alp_T)*FF_T0[:]
        dHdz[:, -1] = alp_T*8*HH_T[:]/cc_pT[:]/(6*(zz[:, -1] - zin[:, -1])) + \
            (1 - alp_T)*FF_T[:]


    # compiling diffusion terms

    diffus_p[:, :] = (1 - theta)*dt*((xx[1:, :] - xx[:-1, :])*(k_bv0[:, 1:]*dHdz_0[:, 1:] - \
        k_bv0[:, :-1]*dHdz_0[:, :-1]) + (zz[:, 1:] - zz[:, :-1])*\
        (k_bh0[1:, :]*dHdr_0[1:, :] - k_bh0[:-1, :]*dHdr_0[:-1, :]))


    diffus[0, :] = theta*dt*(-(zz[0, 1:] - zz[0, :-1])*k_bh[0, :]*dHdr[0, :])
    diffus[-1, :] = theta*dt*((zz[-1, 1:] - zz[-1, :-1])*k_bh[-1, :]*dHdr[-1, :])
    diffus[:, 0] = theta*dt*(-(xx[1:, 0] - xx[:-1, 0])*\
                                k_bv[:, 0]*dHdz[:, 0])
    diffus[:, -1] = theta*dt*((xx[1:, -1] - xx[:-1, -1])*\
                                k_bv[:, -1]*dHdz[:, -1])

    diffus[0, 0] = theta*dt*(-(zz[0, 1] - zz[0, 0])*\
                          k_bh[0, 0]*dHdr[0, 0] - \
        (xx[1, 0] - xx[0, 0])*k_bv[0, 0]*dHdz[0, 0])
        
    diffus[0, -1] = theta*dt*(-(zz[0, -1] - zz[0, -2])*\
                           k_bh[0, -1]*dHdr[0, -1] + \
        (xx[1, -1] - xx[0, -1])*k_bv[0, -1]*dHdz[0, -1])

    diffus[-1, 0] = theta*dt*((zz[-1, 1] - zz[-1, 0])*\
                           k_bh[-1, 0]*dHdr[-1, 0] - \
        (xx[-1, 0] - xx[-2, 0])*k_bv[-1, 0]*dHdz[-1, 0])
        
    diffus[-1, -1] = theta*dt*((zz[-1, -1] - zz[-1, -2])*\
        k_bh[-1, -1]*dHdr[-1, -1] + \
        (xx[-1, -1] - xx[-2, -1])*k_bv[-1, -1]*dHdz[-1, -1])


#=========================== solid fraction terms ============================

    if EQN == 1:
#        lam = (St + (c_p - 1)*Cr)
        lam = St
    elif EQN == 2:
        lam = -Cr

    # spatial derivatives

    dphidr_0, dphidr = map(np.zeros, 2*(np.shape(xx), ))
    dphidz_0, dphidz = map(np.zeros, 2*(np.shape(zz), ))
    
    dphidr_0[1:-1, :] = (phi_0[1:, :]/cc_p0[1:, :] - phi_0[:-1, :]/cc_p0[:-1, :])/\
                        (xin[1:, :] - xin[:-1, :])
    dphidr[1:-1, :] = (phi[1:, :]/cc_p[1:, :] - phi[:-1, :]/cc_p[:-1, :])/\
                      (xin[1:, :] - xin[:-1, :])
    dphidz_0[:, 1:-1] = (phi_0[:, 1:]/cc_p0[:, 1:] - phi_0[:, :-1]/cc_p0[:, :-1])/\
                        (zin[:, 1:] - zin[:, :-1])
    dphidz[:, 1:-1] = (phi[:, 1:]/cc_p[:, 1:] - phi[:, :-1]/cc_p[:, :-1])/\
                      (zin[:, 1:] - zin[:, :-1])


    if np.all(FF_phi_L0) == 0:
        dphidr_0[0, :] = alp_L*(-8*phi_L0[:]/cc_pL0[:] + \
            9*phi_0[0, :]/cc_p0[0, :] - phi_0[1, :]/cc_p0[1, :])/\
            (6*(xin[0, :] - xx[0, :])) + (1 - alp_L)*FF_phi_L0[:]
        dphidr[0, :] = alp_L*(-8*phi_L[:]/cc_pL[:] + \
            9*phi[0, :]/cc_p[0, :] - phi[1, :]/cc_p[1, :])/\
            (6*(xin[0, :] - xx[0, :])) + (1 - alp_L)*FF_phi_L[:]

    if np.all(FF_phi_R0) == 0:
        dphidr_0[-1, :] = alp_R*(8*phi_R0[:]/cc_pR0[:] - \
            9*phi_0[-1, :]/cc_p0[-1, :] + phi_0[-2, :]/cc_p0[-2, :])/\
            (6*(xx[-1, :] - xin[-1, :])) + (1 - alp_R)*FF_phi_R0[:]
        dphidr[-1, :] = alp_R*(8*phi_R[:]/cc_pR[:] - \
            9*phi[-1, :]/cc_p[-1, :] + phi[-2, :]/cc_p[-2, :])/\
            (6*(xx[-1, :] - xin[-1, :])) + (1 - alp_R)*FF_phi_R[:]

    if np.all(FF_phi_B0) == 0:
        dphidz_0[:, 0] = alp_B*(-8*phi_B0[:]/cc_pB0[:] + \
            9*phi_0[:, 0]/cc_p0[:, 0] - phi_0[:, 1]/cc_p0[:, 1])/\
            (6*(zin[:, 0] - zz[:, 0])) + (1 - alp_B)*FF_phi_B0[:]
        dphidz[:, 0] = alp_B*(-8*phi_B[:]/cc_pB[:] + \
            9*phi[:, 0]/cc_p[:, 0] - phi[:, 1]/cc_p[:, 1])/\
            (6*(zin[:, 0] - zz[:, 0])) + (1 - alp_B)*FF_phi_B[:]

    if np.all(FF_phi_T0) == 0:
        dphidz_0[:, -1] = alp_T*(8*phi_T0[:]/cc_pT0[:] - \
            9*phi_0[:, -1]/cc_p0[:, -1] + phi_0[:, -2]/cc_p0[:, -2])/\
            (6*(zz[:, -1] - zin[:, -1])) + (1 - alp_T)*FF_phi_T0[:]
        dphidz[:, -1] = alp_T*(8*phi_T[:]/cc_pT[:] - \
            9*phi[:, -1]/cc_p[:, -1] + phi[:, -2]/cc_p[:, -2])/\
            (6*(zz[:, -1] - zin[:, -1])) + (1 - alp_T)*FF_phi_T[:]

    # compiling diffusion terms

    s_frac_p[:, :] = lam*dt*(1 - theta)*((xx[1:, :] - xx[:-1, :])*\
        (k_bv0[:, 1:]*dphidz_0[:, 1:] - k_bv0[:, :-1]*dphidz_0[:, :-1]) + \
        (zz[:, 1:] - zz[:, :-1])*\
        (k_bh0[1:, :]*dphidr_0[1:, :] - k_bh0[:-1, :]*dphidr_0[:-1, :]))

    s_frac[:, :] = lam*dt*theta*((xx[1:, :] - xx[:-1, :])*\
        (k_bv[:, 1:]*dphidz[:, 1:] - k_bv[:, :-1]*dphidz[:, :-1]) + \
        (zz[:, 1:] - zz[:, :-1])*\
        (k_bh[1:, :]*dphidr[1:, :] - k_bh[:-1, :]*dphidr[:-1, :]))


#=========================== artificial source ================================
    
    source_a = G

#======================== numerical diffusion terms ===========================

    if EQN == 2:

        dCdr_0, dCdr = map(np.zeros, 2*(np.shape(xx), ))
        dCdz_0, dCdz = map(np.zeros, 2*(np.shape(zz), ))
        
        dCdr_0[1:-1, :] = (HH_0[1:, :] - HH_0[:-1, :])/\
                          (xin[1:, :] - xin[:-1, :])
        dCdr[1:-1, :] = (HH[1:, :] - HH[:-1, :])/\
                        (xin[1:, :] - xin[:-1, :])
        dCdz_0[:, 1:-1] = (HH_0[:, 1:] - HH_0[:, :-1])/\
                          (zin[:, 1:] - zin[:, :-1])
        dCdz[:, 1:-1] = (HH[:, 1:] - HH[:, :-1])/\
                        (zin[:, 1:] - zin[:, :-1])
    
        if np.all(FF_L0) == 0:
            dCdr_0[0, :] = alp_L*(-8*HH_L0[:] + \
                9*HH_0[0, :] - HH_0[1, :])/\
                (6*(xin[0, :] - xx[0, :])) + (1 - alp_L)*FF_L0[:]
            dCdr[0, :] = -alp_L*8*HH_L[:]/(6*(xin[0, :] - xx[0, :])) + \
                (1 - alp_L)*FF_L[:]
    
        if np.all(FF_R0) == 0:
            dCdr_0[-1, :] = alp_R*(8*HH_R0[:] - \
                9*HH_0[-1, :] + HH_0[-2, :])/\
                (6*(xx[-1, :] - xin[-1, :])) + (1 - alp_R)*FF_R0[:]
            dCdr[-1, :] = alp_R*8*HH_R[:]/(6*(xx[-1, :] - xin[-1, :])) + \
                (1 - alp_R)*FF_R[:]
    
        if np.all(FF_B0) == 0:
            dCdz_0[:, 0] = alp_B*(-8*HH_B0[:] + \
                9*HH_0[:, 0] - HH_0[:, 1])/\
                (6*(zin[:, 0] - zz[:, 0])) + (1 - alp_B)*FF_B0[:]
            dCdz[:, 0] = -alp_B*8*HH_B[:]/(6*(zin[:, 0] - zz[:, 0])) + \
                (1 - alp_B)*FF_B[:]
    
        if np.all(FF_T0) == 0:
            dCdz_0[:, -1] = alp_T*(8*HH_T0[:] - \
                9*HH_0[:, -1] + HH_0[:, -2])/\
                (6*(zz[:, -1] - zin[:, -1])) + (1 - alp_T)*FF_T0[:]
            dCdz[:, -1] = alp_T*8*HH_T[:]/(6*(zz[:, -1] - zin[:, -1])) + \
                (1 - alp_T)*FF_T[:]
    
        n_diffus_p[:, :] = (1 - theta)*dt*((xx[1:, :] - xx[:-1, :])*(D_n*dCdz_0[:, 1:] - \
            D_n*dCdz_0[:, :-1]) + (zz[:, 1:] - zz[:, :-1])*\
            (D_n*dCdr_0[1:, :] - D_n*dCdr_0[:-1, :]))

    
        n_diffus[0, :] = theta*dt*(-(zz[0, 1:] - zz[0, :-1])*D_n*dCdr[0, :])
        n_diffus[-1, :] = theta*dt*((zz[-1, 1:] - zz[-1, :-1])*D_n*dCdr[-1, :])
        n_diffus[:, 0] = theta*dt*(-(xx[1:, 0] - xx[:-1, 0])*\
                                    D_n*dCdz[:, 0])
        n_diffus[:, -1] = theta*dt*((xx[1:, -1] - xx[:-1, -1])*\
                                    D_n*dCdz[:, -1])
    
    
        n_diffus[0, 0] = theta*dt*(-(zz[0, 1] - zz[0, 0])*\
                              D_n*dCdr[0, 0] - \
            (xx[1, 0] - xx[0, 0])*D_n*dCdz[0, 0])
            
        n_diffus[0, -1] = theta*dt*(-(zz[0, -1] - zz[0, -2])*\
                               D_n*dCdr[0, -1] + \
            (xx[1, -1] - xx[0, -1])*D_n*dCdz[0, -1])
    
        n_diffus[-1, 0] = theta*dt*((zz[-1, 1] - zz[-1, 0])*\
                               D_n*dCdr[-1, 0] - \
            (xx[-1, 0] - xx[-2, 0])*D_n*dCdz[-1, 0])
            
        n_diffus[-1, -1] = theta*dt*((zz[-1, -1] - zz[-1, -2])*\
            D_n*dCdr[-1, -1] + \
            (xx[-1, -1] - xx[-2, -1])*D_n*dCdz[-1, -1])

#=============================================================================
#============================== compiling vector =============================
#=============================================================================


    b[:, :] = time_p[:, :] + diffus[:, :] + diffus_p[:, :] + \
        s_frac[:, :] + s_frac_p[:, :] + source_a[:, :] + \
        n_diffus[:, :] + n_diffus_p[:, :] - advec[:, :]



    b_vec = np.ravel(b, order = 'F')

    return b_vec
