# Number of cores to use
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from numpy.linalg import norm
from scipy.sparse.linalg import spsolve
from Matrix_def import A_theta
from Vector_def import b_theta
import phys_params as ppar
from Enthalpy_functions import SFrac_calc, Q_Vflux, Diff_bounds, Temp_calc, \
    LConc_scal, SFrac_bounds
from int_basic import int_quad


def NL_solve(method, HH_0, HH_p, CC_0, CC_p, xx, xin, t1, t2, 
             a_0, a, p_vals, params, BC_D, BC_N, alp, D_n, G_T = None, G_C = None, 
             omega = 1, tol_a = 1e-8, tol_r = 1e-8, max_iter = 15000):
    
    #========================================================================
    # Function which carries out Picard iteration
    #
    # Inputs:
    #    method - 'BTCS' (backwards Euler) or 'Crank' (Crank-Nicholson)
    #    HH_0, HH_p - enthalpy field at times n, n+1
    #    CC_0, CC_p - bulk salinity field at times n, n+1
    #    xx, xin - spatial arrays
    #    t1, t2 - times n, n+1
    #    a_0, a - interface position at times t1, t2
    #    p_vals - pond temperature / salinity
    #    params - dimensionless parameters
    #    BC_D, BC_N - Dirichlet, Neumann boundary conditions
    #    alp - boundary condition specifier
    #    D_n - numerical diffusivity
    #    G_T, G_C - source terms (heat, solute)
    #    omega - relaxation parameter (default value = 1)
    #    tol_a, tol_r - absolute, relative tolerance
    #    max_iter - maximum number of iterations
    #
    # Outputs:
    #    HH - enthalpy
    #    TT - temperature
    #    CC - bulk salinity
    #    phi - solid fraction
    #    a - interface position
    #    condition - residual
    #
    #========================================================================

    
    # Ensuring supported method has been chosen
    if not method in ('Crank', 'BTCS'):
        raise SyntaxError('Must choose either Crank or BTCS for method.')
        return
    
    # Generating grid
    Nx = np.size(xin)
    dt = t2 - t1
    
    # Defining G for no input
    if G_T is None:
        G_T = np.zeros(Nx)
    if G_C is None:
        G_C = np.zeros(Nx)
    if (np.size(G_T) != Nx) or (np.size(G_C) != Nx):
        raise ValueError('Length of G must equal to length of x')
        return

    # non-dim parameters
    St, Cr, Re, Pr, lam, a_t = params[:]
    c_p = ppar.c_p
    T_p, C_p = p_vals[:]

    n = 0  # Counter
    HH, HH_del, CC, CC_del, TT_0, TT, phi_0, phi, chi_p0, chi_p, cc_p0, cc_p = \
        map(np.zeros, 12*(np.size(HH_0),))
    phi_0[:], phi[:] = SFrac_calc(HH_0, CC_0, St, Cr), SFrac_calc(HH_p, CC_p, St, Cr)
    TT_0[:], TT[:] = Temp_calc(HH_0, CC_0, phi_0, St, Cr), \
                     Temp_calc(HH_p, CC_p, phi, St, Cr)
    CC[:] = CC_p   # this wasn't here before - should be but remove if things go wrong

    tol = tol_a# + tol_r*norm(HH_0)   # Tolerance

    # unpacking boundary conditions
    TT_BC, HH_BC, CC_BC, phi_BC = BC_D[:]
    FF_TT, FF_HH, FF_CC, FF_phi = BC_N[:]

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
    
    phi_b0 = SFrac_bounds(phi_0, xin, xx)
    phi_b = SFrac_bounds(phi, xin, xx)

    phi_L0, phi_R0 = phi_b0[0], phi_b0[-1]
    phi_L, phi_R = phi_b[0], phi_b[-1]
    
    phi_E0 = [phi_L0, phi_R0]
    phi_E = [phi_L, phi_R]
    
    for i in range(len(TT_BC)):
        SS_BC_0.append(LConc_scal(CC_BC_0[i], phi_BC_0[i], Cr))
        SS_BC.append(LConc_scal(CC_BC[i], phi_BC[i], Cr))
    for i in range(len(TT_BC)):
        if phi_E0[i] > 0:
            FF_SS_0.append(FF_CC_0[i])
        else:
            FF_SS_0.append(FF_TT_0[i])

        if phi_E[i] > 0:
            FF_SS.append(FF_CC[i])
        else:
            FF_SS.append(FF_TT[i])


    # phase-weighted specific heat capacity
    cc_p0[:], cc_p[:] = 1 + (c_p - 1)*phi_0[:], 1 + (c_p - 1)*phi[:]
    # liquid-fraction 
    chi_p0[:], chi_p[:] = 1 - phi_0[:], 1 - phi[:]

    if any(HH_p >= CC_p):
        j = (HH_p >= CC_p).nonzero()[0][-1]    # Boundary index
        a_0, T_f0, T_c0, C_c0 = int_quad(TT_0, CC_0, xx, xin, a_0, OPT = 2)
        a, T_f, T_c, C_c = int_quad(TT, CC, xx, xin, a_0, OPT = 2)
    else:
        j = 0
        T_f0 = 0
        T_c0 = 0
        C_c0 = 0
        T_f = 0
        T_c = 0
        C_c = 0


    Q_T0 = Q_Vflux(a_0, T_f0, TT_0, xx, xin, Re, Cr, T_p, lam, a_t, T_c0)
    Q_T = Q_Vflux(a, T_f0, TT, xx, xin, Re, Cr, T_p, lam, a_t, T_c0)
    Q_C0 = Q_Vflux(a_0, T_f0, CC_0, xx, xin, Re, Cr, C_p, lam, a_t, C_c0)
    Q_C = Q_Vflux(a, T_f0, CC, xx, xin, Re, Cr, C_p, lam, a_t, C_c0)
    
    k_b0 = Diff_bounds(phi_0, xx, xin, EQN = 1, OPTION = 1)
    k_b = Diff_bounds(phi, xx, xin, EQN = 1, OPTION = 1)
    D_b0 = Diff_bounds(phi_0, xx, xin, EQN = 2, OPTION = 1)
    D_b = Diff_bounds(phi, xx, xin, EQN = 2, OPTION = 1)
    

    A_T = A_theta(method, xx, xin, dt, cc_p, k_b, alp, D_n)
    b_T0 = b_theta(method, HH_0, HH_p, phi_0, phi, xx, xin, dt, 
                  k_b0, k_b, cc_p0, cc_p, Q_T0, Q_T, HH_BC_0, HH_BC, 
                  phi_BC_0, phi_BC, FF_HH_0, FF_HH, FF_phi_0, FF_phi, 
                  St, Cr, alp, D_n, G = G_T, EQN = 1)

    A_C = A_theta(method, xx, xin, dt, chi_p, D_b, alp, D_n)
    b_C0 = b_theta(method, CC_0, CC_p, phi_0, phi, xx, xin, dt, 
                  D_b0, D_b, chi_p0, chi_p, Q_C0, Q_C, CC_BC_0, CC_BC, 
                  phi_BC_0, phi_BC, FF_CC_0, FF_CC, FF_phi_0, FF_phi, 
                  St, Cr, alp, D_n, G = G_C, EQN = 2)


    condition_p = 100
    con_count = 0

    while n < max_iter:

        n = n + 1

        Jac_T = A_T   # Jacobian
        F_T = A_T*HH_p - b_T0   # function
        HH_del[:] = spsolve(Jac_T, -F_T)
        
        Jac_C = A_C   # Jacobian
        F_C = A_C*CC_p - b_C0   # function
        CC_del[:] = spsolve(Jac_C, -F_C)

        HH[:] = HH_p + omega*HH_del
        CC[:] = CC_p + omega*CC_del
        phi[:] = SFrac_calc(HH, CC, St, Cr)
        TT[:] = Temp_calc(HH, CC, phi, St, Cr)

        # phase-weighted specific heat capacity
        cc_p[:] = 1 + (c_p - 1)*phi[:]
        # liquid-fraction
        chi_p[:] = 1 - phi[:]

        if any(HH >= CC):
            j = (HH >= CC).nonzero()[0][-1]    # Boundary index
            a, T_f, T_c, C_c = int_quad(HH, CC, xx, xin, a_0, OPT = 2)
            Q_T = Q_Vflux(a, T_f, TT, xx, xin, Re, Cr, T_p, lam, a_t, T_c)
            Q_C = Q_Vflux(a, T_f, CC, xx, xin, Re, Cr, C_p, lam, a_t, C_c)
        else:
            j = 0
            a = 0
            Q = np.zeros(Nx)
        

        k_b = Diff_bounds(phi, xx, xin, EQN = 1, OPTION = 1)
        D_b = Diff_bounds(phi, xx, xin, EQN = 2, OPTION = 1)

        A_T = A_theta(method, xx, xin, dt, cc_p, k_b, alp, D_n)
        b_T0 = b_theta(method, HH_0, HH, phi_0, phi, xx, xin, dt, 
                      k_b0, k_b, cc_p0, cc_p, Q_T0, Q_T, HH_BC_0, HH_BC, 
                      phi_BC_0, phi_BC, FF_HH_0, FF_HH, FF_phi_0, FF_phi, 
                      St, Cr, alp, D_n, G = G_T, EQN = 1)

        A_C = A_theta(method, xx, xin, dt, chi_p, D_b, alp, D_n)
        b_C0 = b_theta(method, CC_0, CC, phi_0, phi, xx, xin, dt, 
                      D_b0, D_b, chi_p0, chi_p, Q_C0, Q_C, CC_BC_0, CC_BC, 
                      phi_BC_0, phi_BC, FF_CC_0, FF_CC, FF_phi_0, FF_phi, 
                      St, Cr, alp, D_n, G = G_C, EQN = 2)

        HH_diff = HH - HH_p
        CC_diff = CC - CC_p

        condition = max(norm(HH_del), norm(CC_del))
        condition_alt = max(norm(HH_diff), norm(CC_diff))
        
            
        if condition > condition_p:
            con_count += 1
            
        if con_count == 5:
            HH[:] = np.nan
            TT[:] = np.nan
            CC[:] = np.nan
            phi[:] = np.nan
            condition = np.nan
            break


        if np.isnan(norm(HH_del)) or np.isnan(norm(CC_del)):
#            print('Time step too large - instability')
            break

        if condition < tol or condition_alt < tol:
            break

        HH_p[:] = HH 
        CC_p[:] = CC
        condition_p = condition


    condition = min(condition, condition_alt)

    if not norm(condition) < tol:
        tol_upper = 10*tol_a + 10*tol_r*norm(HH_0)   # Tolerance
        # allowing for completion if resid slightly larger than tol
        # (remove if desired)
        if norm(condition) < tol_upper:
            return HH, TT, CC, phi, a, condition
        return HH, TT, CC, phi, a, condition
    else:
        return HH, TT, CC, phi, a, condition
        
