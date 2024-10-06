# Number of cores to use
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import phys_params as ppar


def b_theta(method, HH_0, HH, phi_0, phi, xx, xin, dt, 
            k_b0, k_b, cc_p0, cc_p, Q_0, Q, HH_BC0, HH_BC, phi_BC0, phi_BC, 
            FF_0, FF, FF_phi_0, FF_phi, St, Cr, alp, D_n, G = None, EQN = 1):

    #========================================================================
    # Function which creates the right hand side vector for the heat / 
    # solute conservation equation
    #
    # Inputs:
    #    method - 'BTCS' (backwards Euler) or 'Crank' (Crank-Nicholson)
    #    HH_0, HH - enthalpy / bulk salinity field at times n, n+1
    #    phi_0, phi - solid fraction field at times n, n+1
    #    xx, xin - spatial arrays
    #    dt - time step
    #    cc_p0, cc_p - specific heat capacity / porosity at times n, n+1
    #    k_bh0, k_bh, k_bv0, k_bv - thermal conductivity / solutal diffusivity
    #                               at times n, n+1
    #    Q_0, Q - heat / solute flux in channel at times n, n+1
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
    Nx = np.size(xin)

    # Checking length of various arrays:
    # Temperature
    if not np.size(HH_0) == Nx:
        raise ValueError('Length of TT_p must equal to length of x')
        return
    
    # Ensuring supported method has been chosen
    if not method in ('Crank', 'BTCS'):
        raise SyntaxError('Must choose either Crank or BTCS for method.')
        return

    c_pl, c_ps = 4200, 2100  # liquid/solid heat capacity
    c_p = c_ps/c_pl
    
    
    # Defining theta 
    if method == 'BTCS':
        theta = 1        
    elif method == 'Crank':
        theta = 0.5

    # boundary condition type
    # alp = 1 - Dirichlet, alp = 0 - Neumann
    alp_L, alp_R = alp[0], alp[1]

    # Dirichlet conditions
    # fixed enthalpies
    HH_L0, HH_R0 = HH_BC0[0], HH_BC0[1]
    HH_L, HH_R = HH_BC[0], HH_BC[1]
    # fixed solid fractions
    phi_L0, phi_R0 = phi_BC0[0], phi_BC0[1]
    phi_L, phi_R = phi_BC[0], phi_BC[1]
    
    if EQN == 1:
        # s. heat capacity at boundaries
        cc_pL0, cc_pR0 = 1 + (c_p - 1)*phi_L0, 1 + (c_p - 1)*phi_R0
        cc_pL, cc_pR = 1 + (c_p - 1)*phi_L, 1 + (c_p - 1)*phi_R

    elif EQN == 2:
        # liquid-fraction at boundaries
        cc_pL0, cc_pR0 = 1 - phi_L0, 1 - phi_R0        
        cc_pL, cc_pR = 1 - phi_L, 1 - phi_R

    # Neumann conditions
    # enthalpy fluxes
    FF_L0, FF_R0 = FF_0[0], FF_0[1]
    FF_L, FF_R = FF[0], FF[1]
    # solid fraction fluxes
    FF_phi_L0, FF_phi_R0 = FF_phi_0[0], FF_phi_0[1]
    FF_phi_L, FF_phi_R = FF_phi[0], FF_phi[1]

    # Initializing b
    b, time_p, advec, diffus, diffus_p, s_frac, s_frac_p, source_n, \
        source_a, n_diffus, n_diffus_p = map(np.zeros, 11*(Nx, ))

#================================= Main body =================================

#================================= time terms ================================

    time_p[:] = (xx[1:]**2 - xx[:-1]**2)/2*HH_0[:]


#============================== advection terms ==============================

    advec[:] = 0

#============================== diffusion terms ==============================

    dHdx_0, dHdx = map(np.zeros, 2*(np.shape(xx), ))
    
    dHdx_0[1:-1] = (HH_0[1:]/cc_p0[1:] - HH_0[:-1]/cc_p0[:-1])/\
                   (xin[1:] - xin[:-1])
    dHdx[1:-1] = (HH[1:]/cc_p[1:] - HH[:-1]/cc_p[:-1])/(xin[1:] - xin[:-1])

    if np.all(FF_L0) == 0:
        dHdx_0[0] = alp_L*(-8*HH_L0/cc_pL0 + \
            9*HH_0[0]/cc_p0[0] - HH_0[1]/cc_p0[1])/\
            (6*(xin[0] - xx[0])) + (1 - alp_L)*FF_L0
        dHdx[0] = -alp_L*8*HH_L/cc_pL/(6*(xin[0] - xx[0])) + \
            (1 - alp_L)*FF_L

    if np.all(FF_R0) == 0:
        dHdx_0[-1] = alp_R*(8*HH_R0/cc_pR0 - \
            9*HH_0[-1]/cc_p0[-1] + HH_0[-2]/cc_p0[-2])/\
            (6*(xx[-1] - xin[-1])) + (1 - alp_R)*FF_R0
        dHdx[-1] = alp_R*8*HH_R/cc_pR/(6*(xx[-1] - xin[-1])) + \
            (1 - alp_L)*FF_R

    diffus_p[:] = (1 - theta)*dt*(k_b0[1:]*xx[1:]*dHdx_0[1:] - \
                                  k_b0[:-1]*xx[:-1]*dHdx_0[:-1])
    diffus[0] = -theta*dt*k_b[0]*xx[0]*dHdx[0]
    diffus[-1] = theta*dt*k_b[-1]*xx[-1]*dHdx[-1]


#=========================== solid fraction terms ============================

    if EQN == 1:
        lam = St
    elif EQN == 2:
        lam = -Cr
        
    dphidx_0, dphidx = map(np.zeros, 2*(np.shape(xx), ))
    
    dphidx_0[1:-1] = (phi_0[1:]/cc_p0[1:] - phi_0[:-1]/cc_p0[:-1])/\
                     (xin[1:] - xin[:-1])
    dphidx[1:-1] = (phi[1:]/cc_p[1:] - phi[:-1]/cc_p[:-1])/\
                   (xin[1:] - xin[:-1])

    if np.all(FF_phi_L0) == 0:
        dphidx_0[0] = alp_L*(-8*phi_L0/cc_pL0 + \
            9*phi_0[0]/cc_p0[0] - phi_0[1]/cc_p0[1])/\
            (6*(xin[0] - xx[0])) + (1 - alp_L)*FF_phi_L0
        dphidx[0] = alp_L*(-8*phi_L/cc_pL + \
            9*phi[0]/cc_p[0] - phi[1]/cc_p[1])/\
            (6*(xin[0] - xx[0])) + (1 - alp_L)*FF_phi_L

    if np.all(FF_phi_R0) == 0:
        dphidx_0[-1] = alp_R*(8*phi_R0/cc_pR0 - \
            9*phi_0[-1]/cc_p0[-1] + phi_0[-2]/cc_p0[-2])/\
            (6*(xx[-1] - xin[-1])) + (1 - alp_R)*FF_phi_R0
        dphidx[-1] = alp_R*(8*phi_R/cc_pR - \
            9*phi[-1]/cc_p[-1] + phi[-2]/cc_p[-2])/\
            (6*(xx[-1] - xin[-1])) + (1 - alp_R)*FF_phi_R

    s_frac_p[:] = lam*dt*(1 - theta)*(k_b0[1:]*xx[1:]*dphidx_0[1:] - \
                                      k_b0[:-1]*xx[:-1]*dphidx_0[:-1])

    s_frac[:] = lam*dt*theta*(k_b[1:]*xx[1:]*dphidx[1:] - \
                              k_b[:-1]*xx[:-1]*dphidx[:-1])


#============================= natural source =================================

    source_n[:] = (theta*Q[:]*(xx[1:]**2 - xx[:-1]**2)/2 + \
            (1 - theta)*Q_0[:]*(xx[1:]**2 - xx[:-1]**2)/2)*dt

#=========================== artificial source ================================
    
    source_a[:] = G
    
    
#======================== artificial diffusion terms ==========================

    if EQN == 2:

        dCdx_0, dCdx = map(np.zeros, 2*(np.shape(xx), ))
        
        dCdx_0[1:-1] = (HH_0[1:] - HH_0[:-1])/(xin[1:] - xin[:-1])
        dCdx[1:-1] = (HH[1:] - HH[:-1])/(xin[1:] - xin[:-1])
    
        if np.all(FF_L0) == 0:
            dCdx_0[0] = alp_L*(-8*HH_L0 + 9*HH_0[0] - HH_0[1])/\
                (6*(xin[0] - xx[0])) + (1 - alp_L)*FF_L0
            dCdx[0] = -alp_L*8*HH_L/(6*(xin[0] - xx[0])) + \
                (1 - alp_L)*FF_L
    
        if np.all(FF_R0) == 0:
            dCdx_0[-1] = alp_R*(8*HH_R0 - 9*HH_0[-1] + HH_0[-2])/\
                (6*(xx[-1] - xin[-1])) + (1 - alp_R)*FF_R0
            dCdx[-1] = alp_R*8*HH_R/(6*(xx[-1] - xin[-1])) + \
                (1 - alp_L)*FF_R

        n_diffus_p[:] = (1 - theta)*dt*(D_n*xx[1:]*dCdx_0[1:] - \
                                        D_n*xx[:-1]*dCdx_0[:-1])
    
        n_diffus[0] = -theta*dt*D_n*xx[0]*dCdx[0]
        n_diffus[-1] = theta*dt*D_n*xx[-1]*dCdx[-1]


#=============================================================================
#============================== compiling vector =============================
#=============================================================================

    # Defining b
    b[:] = time_p[:] + diffus[:] + diffus_p[:] + \
        s_frac[:] + s_frac_p[:] + source_n[:] + source_a[:] + \
        n_diffus[:] + n_diffus_p[:] - advec[:]


    
    return b