# Number of cores to use
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from numpy.linalg import norm
from scipy.sparse.linalg import spsolve
from Matrix_def import A_theta
from Vector_def import b_theta
from elliptic_solve_sparse import p_solve
import phys_params as ppar
from Enthalpy_functions import SFrac_calc, Diff_bounds, Temp_calc, Perm_calc, LConc
from upwind import MUSCL
from newton_functions import J_resid



def NL_solve(method, HH_0, HH_p, CC_0, CC_p, pp_0, pp_p, xx, xin, zz, zin, t1, t2,
           params, BC_D, BC_N, alp, D_n, 
           G_T = None, G_C = None, G_p = None, gamma = 0, omega = 1, 
           tol_a = 1e-8, tol_r = 1e-8, max_iter = 15000):
    
    #========================================================================
    # Function which carries out Picard iteration or Newton method
    #
    # Inputs:
    #    method - 'BTCS' (backwards Euler) or 'Crank' (Crank-Nicholson)
    #    HH_0, HH_p - enthalpy field at times n, n+1
    #    CC_0, CC_p - bulk salinity field at times n, n+1
    #    pp_0, pp_p - pressure field at times n, n+1
    #    xx, xin, zz, zin - spatial arrays
    #    t1, t2 - times n, n+1
    #    params - dimensionless parameters
    #    BC_D, BC_N - Dirichlet, Neumann boundary conditions
    #    alp - boundary condition specifier
    #    D_n - numerical diffusivity
    #    G_T, G_C, G_p - source terms (heat, solute, pressure)
    #    gamma - method selector (Picard = 0, Newton = 1)
    #    omega - relaxation parameter (default value = 1)
    #    tol_a, tol_r - absolute, relative tolerance
    #    max_iter - maximum number of iterations
    #
    # Outputs:
    #    HH - enthalpy
    #    TT - temperature
    #    CC - bulk salinity
    #    phi - solid fraction
    #    pp - pressure
    #    u_b, w_b - horizontal, vertical velocities
    #    condition - residual
    #
    #========================================================================

    
    # Ensuring supported method has been chosen
    if not method in ('Crank', 'BTCS'):
        raise SyntaxError('Must choose either Crank or BTCS for method.')
        return
    
    # Extracting info from inputs
    Nx, Nz = np.shape(xin)
    dt = t2 - t1
    
    # Defining G for no input
    if G_T is None:
        G_T = np.zeros([Nx, Nz])
    if G_C is None:
        G_C = np.zeros([Nx, Nz])
    if G_p is None:
        G_p = np.zeros([Nx, Nz])
    if (np.shape(G_T) != (Nx, Nz)) or (np.shape(G_C) != (Nx, Nz)) or \
        (np.shape(G_p) != (Nx, Nz)):
        raise ValueError('Length of G must equal to length of x')
#        print(G_T)
        return

    # coverting fields into column arrays
    HH_0_col, HH_p_col = np.ravel(HH_0, order = 'F'), \
                         np.ravel(HH_p, order = 'F')
    CC_0_col, CC_p_col = np.ravel(CC_0, order = 'F'), \
                         np.ravel(CC_p, order = 'F')

    # non-dim parameters
    St, Cr, Pe, Da_h = params[:]
    c_p = ppar.c_p

    n = 0  # Counter
    HH_col, HH_del, CC_col, CC_del = map(np.zeros, 4*(np.shape(HH_0_col),))
    TT_0, TT, HH, CC, pp, phi_0, phi, chi_0, chi_p, cc_p0, cc_p = \
                                     map(np.zeros, 11*((Nx, Nz),))
    TT_Fx, SS_Fx, u_b0, u_b = map(np.zeros, 4*((Nx+1, Nz), ))
    TT_Fz, SS_Fz, w_b0, w_b = map(np.zeros, 4*((Nx, Nz+1), ))
    phi_0[:], phi[:] = SFrac_calc(HH_0, CC_0, St, Cr), SFrac_calc(HH_p, CC_p, St, Cr)
    TT_0[:], TT[:] = Temp_calc(HH_0, CC_0, phi_0, St, Cr), \
                     Temp_calc(HH_p, CC_p, phi, St, Cr)
    CC[:] = CC_p

    tol = tol_a   # Tolerance
    
    # unpacking boundary conditions
    alp_T, alp_p = alp[:]
    TT_BC, HH_BC, CC_BC, phi_BC, pp_BC = BC_D[:]
    FF_TT, FF_HH, FF_CC, FF_phi, FF_pp = BC_N[:]

    TT_BC_0 = TT_BC.copy()
    HH_BC_0 = HH_BC.copy()
    CC_BC_0 = CC_BC.copy()
    phi_BC_0 = phi_BC.copy()
    FF_TT_0 = FF_TT.copy()
    FF_HH_0 = FF_HH.copy()
    FF_CC_0 = FF_CC.copy()
    FF_phi_0 = FF_phi.copy()
    
    SS_BC_0, SS_BC = [], []
    FF_SS_0, FF_SS = [], []
    for i in range(len(TT_BC)):
        SS_BC_0.append(LConc(CC_BC_0[i], phi_BC_0[i], Cr))
        SS_BC.append(LConc(CC_BC[i], phi_BC[i], Cr))
    for i in range(len(TT_BC)):
        FF_0_temp, FF_temp = map(np.zeros, 2*(np.size(SS_BC_0[i]), ))
        FF_0_temp[SS_BC_0[i] == TT_BC_0[i]] = FF_TT_0[i][SS_BC_0[i] == TT_BC_0[i]]
        FF_0_temp[SS_BC_0[i] == CC_BC_0[i]] = FF_CC_0[i][SS_BC_0[i] == CC_BC_0[i]]
        FF_temp[SS_BC[i] == TT_BC[i]] = FF_TT[i][SS_BC[i] == TT_BC[i]]
        FF_temp[SS_BC[i] == CC_BC[i]] = FF_CC[i][SS_BC[i] == CC_BC[i]]
        FF_SS_0.append(FF_0_temp)
        FF_SS.append(FF_temp)
        

    # phase-weighted specific heat capacity
    cc_p0[:], cc_p[:] = 1 + (c_p - 1)*phi_0[:], 1 + (c_p - 1)*phi[:]
    # liquid-fraction 
    chi_0[:], chi_p[:] = 1 - phi_0[:], 1 - phi[:]

    # permeability, pressure, velocity
    Pi_0, Pi_bh0, Pi_bv0 = Perm_calc(phi_0, xx, xin, zz, zin, phi_BC_0, FF_phi_0,
                                 alp_T, Da_h)
    Pi, Pi_bh, Pi_bv = Perm_calc(phi, xx, xin, zz, zin, phi_BC, FF_phi,
                                 alp_T, Da_h)
    

    # velocities
    pp_0[:], u_b0[:], w_b0[:] = p_solve(xx, xin, zz, zin, Pi_bh0, Pi_bv0,
                                        pp_BC, FF_pp, alp_p, Pe, G = G_p)

    pp[:], u_b[:], w_b[:] = p_solve(xx, xin, zz, zin, Pi_bh, Pi_bv,
                                    pp_BC, FF_pp, alp_p, Pe, G = G_p)


    # diffusivities/conductivities
    k_bh0, k_bv0 = Diff_bounds(phi_0, xx, xin, zz, zin, EQN = 1, OPTION = 1)
    k_bh, k_bv = Diff_bounds(phi, xx, xin, zz, zin, EQN = 1, OPTION = 1)
    D_bh0, D_bv0 = Diff_bounds(phi_0, xx, xin, zz, zin, EQN = 2, OPTION = 1)
    D_bh, D_bv = Diff_bounds(phi, xx, xin, zz, zin, EQN = 2, OPTION = 1)


    # cell-boundary values
    # temperature
    TT_bh0, TT_bv0 = MUSCL(TT_0, u_b0, w_b0, xx, xin, zz, zin, TT_BC, FF_TT, alp_T)
    TT_bh, TT_bv = MUSCL(TT, u_b, w_b, xx, xin, zz, zin, TT_BC, FF_TT, alp_T)



    # liquid concentration
    SS_0, SS = LConc(CC_0, phi_0, Cr), LConc(CC, phi, Cr)

    SS_bh0, SS_bv0 = MUSCL(SS_0, u_b0, w_b0, xx, xin, zz, zin, SS_BC, FF_SS, alp_T)
    SS_bh, SS_bv = MUSCL(SS, u_b, w_b, xx, xin, zz, zin, SS_BC, FF_SS, alp_T)


    # constructing advective fluxes (semi-implicit)
    TT_Fx[:, :] = (u_b0[:, :] + u_b[:, :])*(TT_bh0[:, :] + TT_bh[:, :])/4
    TT_Fz[:, :] = (w_b0[:, :] + w_b[:, :])*(TT_bv0[:, :] + TT_bv[:, :])/4

    SS_Fx[:, :] = (u_b0[:, :] + u_b[:, :])*(SS_bh0[:, :] + SS_bh[:, :])/4
    SS_Fz[:, :] = (w_b0[:, :] + w_b[:, :])*(SS_bv0[:, :] + SS_bv[:, :])/4
    
    
# =============================================================================
#     # constructing advective fluxes (fully explicit)
#     TT_Fx[:, :] = u_b0[:, :]*TT_bh0[:, :]
#     TT_Fz[:, :] = w_b0[:, :]*TT_bv0[:, :]
# 
#     SS_Fx[:, :] = u_b0[:, :]*SS_bh0[:, :]
#     SS_Fz[:, :] = w_b0[:, :]*SS_bv0[:, :]
# =============================================================================

     # defining matrix and vector for heat / solute equation
    A_T = A_theta(method, xx, xin, zz, zin, dt, cc_p, k_bh, k_bv, alp_T, D_n, EQN = 1)
    b_T0 = b_theta(method, HH_0, HH_p, phi_0, phi, xx, xin, zz, zin, dt, 
                TT_Fx, TT_Fz, cc_p0, cc_p, k_bh0, k_bh, k_bv0, k_bv, 
                HH_BC_0, HH_BC, phi_BC_0, phi_BC, FF_HH_0, FF_HH, FF_phi_0, FF_phi, 
                St, Cr, alp_T, D_n, G = G_T, EQN = 1)


    if gamma == 1:
        J_Tres = J_resid(method, HH_p, CC_p, phi, xx, xin, zz, zin, dt, 
                         k_bh, k_bv, cc_p, HH_BC, phi_BC, FF_HH, FF_phi, 
                         St, Cr, alp_T, D_n, EQN = 1)
    else:
        J_Tres = 0

    A_C = A_theta(method, xx, xin, zz, zin, dt, chi_p, D_bh, D_bv, alp_T, D_n, EQN = 2)
    b_C0 = b_theta(method, CC_0, CC_p, phi_0, phi, xx, xin, zz, zin, dt, 
                SS_Fx, SS_Fz, chi_0, chi_p, D_bh0, D_bh, D_bv0, D_bv, 
                CC_BC_0, CC_BC, phi_BC_0, phi_BC, FF_CC_0, FF_CC, FF_phi_0, FF_phi, 
                St, Cr, alp_T, D_n, G = G_C, EQN = 2)

    if gamma == 1:
        J_Cres = J_resid(method, HH_p, CC_p, phi, xx, xin, zz, zin, dt, 
                         D_bh, D_bv, chi_p, CC_BC, phi_BC, FF_CC, FF_phi, 
                         St, Cr, alp_T, D_n, EQN = 2)
    else:
        J_Cres = 0

    condition_p = 100
    con_count = 0

    while n < max_iter:

        n = n + 1
        
        # solving
        Jac_T = A_T + J_Tres   # Jacobian
        F_T = A_T*HH_p_col - b_T0   # function
        HH_del[:] = spsolve(Jac_T, -F_T)

        Jac_C = A_C + J_Cres   # Jacobian
        F_C = A_C*CC_p_col - b_C0   # function
        CC_del[:] = spsolve(Jac_C, -F_C)

        HH_col[:] = HH_p_col + omega*HH_del
        CC_col[:] = CC_p_col + omega*CC_del
        HH = HH_col.reshape([Nx, Nz], order = 'F')[:]
        CC = CC_col.reshape([Nx, Nz], order = 'F')[:]
        phi[:] = SFrac_calc(HH, CC, St, Cr)
        TT[:] = Temp_calc(HH, CC, phi, St, Cr)

        # phase-weighted specific heat capacity
        cc_p[:] = 1 + (c_p - 1)*phi[:]
        # liquid-fraction
        chi_p[:] = 1 - phi[:]

        # permeability, pressure, velocity
        Pi, Pi_bh, Pi_bv = Perm_calc(phi, xx, xin, zz, zin, phi_BC, FF_phi,
                                     alp_T, Da_h)

        pp[:], u_b[:], w_b[:] = p_solve(xx, xin, zz, zin, Pi_bh, Pi_bv,
                                        pp_BC, FF_pp, alp_p, Pe, G = G_p)

        # diffusivities/conductivities
        k_bh, k_bv = Diff_bounds(phi, xx, xin, zz, zin, EQN = 1, OPTION = 1)
        D_bh, D_bv = Diff_bounds(phi, xx, xin, zz, zin, EQN = 2, OPTION = 1)


        # cell-boundary values
        TT_bh, TT_bv = MUSCL(TT, u_b, w_b, xx, xin, zz, zin, TT_BC, FF_TT, alp_T)

        # liquid concentration
        SS = LConc(CC, phi, Cr)
        SS_bh, SS_bv = MUSCL(SS, u_b, w_b, xx, xin, zz, zin, SS_BC, FF_SS, alp_T)
        
        # updating advective fluxes
        TT_Fx[:, :] = (u_b0[:, :] + u_b[:, :])*(TT_bh0[:, :] + TT_bh[:, :])/4
        TT_Fz[:, :] = (w_b0[:, :] + w_b[:, :])*(TT_bv0[:, :] + TT_bv[:, :])/4

        SS_Fx[:, :] = (u_b0[:, :] + u_b[:, :])*(SS_bh0[:, :] + SS_bh[:, :])/4
        SS_Fz[:, :] = (w_b0[:, :] + w_b[:, :])*(SS_bv0[:, :] + SS_bv[:, :])/4

        # redefining matrix and vector for heat / solute equations
        A_T = A_theta(method, xx, xin, zz, zin, dt, cc_p, k_bh, k_bv, alp_T, D_n, EQN = 1)
        b_T0 = b_theta(method, HH_0, HH, phi_0, phi, xx, xin, zz, zin, dt, 
                    TT_Fx, TT_Fz, cc_p0, cc_p, k_bh0, k_bh, k_bv0, k_bv, 
                    HH_BC_0, HH_BC, phi_BC_0, phi_BC, FF_HH_0, FF_HH, FF_phi_0, FF_phi, 
                    St, Cr, alp_T, D_n, G = G_T, EQN = 1)
        
        if gamma == 1:
            J_Tres = J_resid(method, HH, CC, phi, xx, xin, zz, zin, dt, 
                             k_bh, k_bv, cc_p, HH_BC, phi_BC, FF_HH, FF_phi, 
                             St, Cr, alp_T, D_n, EQN = 1)

        A_C = A_theta(method, xx, xin, zz, zin, dt, chi_p, D_bh, D_bv, alp_T, D_n, EQN = 2)
        b_C0 = b_theta(method, CC_0, CC, phi_0, phi, xx, xin, zz, zin, dt, 
                    SS_Fx, SS_Fz, chi_0, chi_p, D_bh0, D_bh, D_bv0, D_bv, 
                    CC_BC_0, CC_BC, phi_BC_0, phi_BC, FF_CC_0, FF_CC, FF_phi_0, FF_phi, 
                    St, Cr, alp_T, D_n, G = G_C, EQN = 2)
        
        if gamma == 1:
            J_Cres = J_resid(method, HH, CC, phi, xx, xin, zz, zin, dt, 
                             D_bh, D_bv, chi_p, CC_BC, phi_BC, FF_CC, FF_phi, 
                             St, Cr, alp_T, D_n, EQN = 2)
        
        # calculating residual
        condition = max(norm(HH_del), norm(CC_del))
            
        # checking for instability
        if condition > condition_p:
            con_count += 1
            
        if con_count == 5:
            HH[:, :] = np.nan
            TT[:, :] = np.nan
            CC[:, :] = np.nan
            phi[:, :] = np.nan
            pp[:, :] = np.nan
            u_b[:, :] = np.nan
            w_b[:, :] = np.nan
            condition = np.nan
            break
            
        if np.isnan(norm(HH_del)) or np.isnan(norm(CC_del)):
            print('Time step too large - instability')
            break

        # breaking if tolerance met
        if condition < tol:
            break
        
        # updating fields
        HH_p[:] = HH 
        CC_p[:] = CC
        HH_p_col[:] = HH_col
        CC_p_col[:] = CC_col
        condition_p = condition

    if not norm(condition) < tol:
        tol_upper = 10*tol_a + 10*tol_r*norm(HH_0)   # Tolerance
        # allowing for completion if resid slightly larger than tol
        # (remove if desired)
        if norm(condition) < tol_upper:
            return HH, TT, CC, phi, pp, u_b, w_b, condition
#            print('Picard iteration failed to converge')
        return HH, TT, CC, phi, pp, u_b, w_b, condition
    else:
        return HH, TT, CC, phi, pp, u_b, w_b, condition
        
