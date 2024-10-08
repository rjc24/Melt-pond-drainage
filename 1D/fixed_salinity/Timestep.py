# Number of cores to use
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from NL_solve import NL_solve

def time_step_1(method, HH_0, CC_0, xx, xin, t1, t2, a_0, T_p,
                params, BC_D, BC_N, alp, G_T = None, G_C = None, 
                omega_s = 1, omega_i = 0.5, tol_a = 1e-8, 
                tol_r = 1e-8, tol_g = 1e-8, max_iter = 100):
    
    #========================================================================
    # Function which calculates solution at t = t2 from data at t = t1
    #
    # Inputs:
    #    method - 'BTCS' (backwards Euler) or 'Crank' (Crank-Nicholson)
    #    HH_0 - enthalpy field at t = t1
    #    CC_0 - bulk salinity field at t = t1
    #    xx, xin - spatial arrays
    #    t1, t2 - time steps
    #    a_0 - interface position at t = t1
    #    T_p - pond temperature
    #    params - dimensionless parameters
    #    BC_D, BC_N - Dirichlet, Neumann boundary conditions
    #    alp - boundary condition specifier
    #    G_T, G_C - source terms (heat, solute)
    #    omega_s, omega_i - relaxation parameters (Picard, interface position)
    #    tol_a, tol_r, tol_g - absolute, relative, interface tolerance
    #    max_iter - maximum number of iterations (calls to Picard / Newton)
    #
    # Outputs:
    #    HH - enthalpy at t = t2
    #    TT - temperature at t = t2
    #    CC - bulk salinity at t = t2
    #    phi - solid fraction at t = t2
    #    a - interface position at t = t2
    #    err - residual from Picard iteration
    #========================================================================
    
    HH_p, CC_p = map(np.zeros, 2*(np.size(xin), ))
    # Initial guess for a^(n+1)
    a = a_0
    a_p, a_pp = a_0, a_0   # previous iterates of a
    
    HH_p[:] = HH_0
    CC_p[:] = CC_0
    diff = 1   # maybe change initial diff value
    err = 1
    tol = tol_a# + tol_r*norm(HH_0)   # Tolerance
    count = 0
    NL_max = 100
    while diff > tol_g or err > tol:
        count = count + 1
        HH, TT, CC, phi, a, err = \
            NL_solve(method, HH_0, HH_p, CC_0, CC_p, xx, xin, 
                     t1, t2, a_0, a, T_p, params, BC_D, BC_N, alp, 
                     G_T, G_C, omega_s, tol_a, tol_r, 
                     max_iter = NL_max)
            
        if not any(np.isnan(np.array([a, a_p, a_pp]))):
            a = omega_i*a + (1-omega_i)*a_p
            diff = abs(a - a_p) + abs(a_p - a_pp)
        else:
            diff = 0
        if np.mod(count,250) == 0:
            print(count, diff)
        HH_p[:] = HH
        CC_p[:] = CC
        a_pp = a_p
        a_p = a

        if count == max_iter:
            break

    return HH, TT, CC, phi, a, err


def time_step_A(method, HH_0, CC_0, xx, xin, t1, t2, dt_max,
              a_0, T_p, params, BC_D, BC_N, 
              alp, omega_s = 1, omega_i = 0.5, 
              tol_a = 1e-8, tol_r = 1e-8, tol_g = 1e-8, max_iter = 100):

    #========================================================================
    # Adaptive time-stepper. Calculates solution at time t2 using data from
    # time t1 in one full step and two half steps. Procedure is repeated until
    # desired tolerance is met.
    #
    # Inputs:
    #    method - 'BTCS' (backwards Euler) or 'Crank' (Crank-Nicholson)
    #    HH_0 - enthalpy field at t = t1
    #    CC_0 - bulk salinity field at t = t1
    #    xx, xin - spatial arrays
    #    t1, t2 - time steps
    #    dt_max - maximum time step (initial)
    #    a_0 - interface position at t = t1
    #    T_p - pond temperature
    #    params - dimensionless parameters
    #    BC_D, BC_N - Dirichlet, Neumann boundary conditions
    #    alp - boundary condition specifier
    #    omega_s, omega_i - relaxation parameters (Picard, interface position)
    #    tol_a, tol_r, tol_g - absolute, relative, interface tolerance
    #    max_iter - maximum number of iterations (calls to Picard / Newton)
    #
    # Outputs:
    #    HH_h - enthalpy at t = t2
    #    TT_h - temperature at t = t2
    #    CC_h - bulk salinity at t = t2
    #    phi_h - solid fraction at t = t2
    #    a_h - interface position at t = t2
    #    t2, t3 - time of solution, next time to calculate
    #    err_est, n - estimated error, number of iterations until tolerance met
    #
    #========================================================================

    # unpacking BCs
    TT_BC = BC_D[0]
    CC_BC = BC_D[2]
    phi_BC, FF_phi = BC_D[3], BC_N[3]
    alp_T = alp[0]
    
    HH_0h, CC_0h, dx = map(np.zeros, 3*(np.size(xin), ))
    err_h_arr = np.zeros(2)

    max_t_iter = 15   # maximum no. of attempts
    n = 0   # counter
    tol = tol_a# + tol_r*norm(HH_0)   # Tolerance
    if method == 'BTCS':   # setting theta
        theta = 1
    elif method == 'Crank':
        theta = 0.5
    q = 1/theta   # defining q

    dx[:] = xx[1:] - xx[:-1]

    while n < max_t_iter:
        n = n + 1
        dt = t2 - t1   # full timestep
        th = t1 + 0.5*dt   # half-time
        if any(n == np.array([3, 6])):
            tol = 10*tol   # adjusting tolerance if struggling to converge
        # Setting temporal error tolerance
        if method == 'BTCS':
            tol_t = min(dx)
        elif method == 'Crank':
            tol_t = min(dx**2)

        G_Tf, G_Cf = map(np.zeros, 2*(np.shape(HH_0), ))

        # full step
        HH_f, TT_f, CC_f, phi_f, a_f, err_f = \
            time_step_1(method, HH_0, CC_0, xx, xin, t1, t2, a_0,
                T_p, params, BC_D, BC_N, alp, G_Tf, G_Cf,
                omega_s, omega_i, tol_a, tol_r, tol_g, max_iter)


        if ~np.isnan(HH_f).any():
            # two half steps
            t_a, t_b = t1, th
            a_a = a_0
            HH_0h[:] = HH_0
            CC_0h[:] = CC_0
            for i in range(2):
                
                G_Th, G_Ch = map(np.zeros, 2*(np.shape(HH_0), ))
    
                HH_h, TT_h, CC_h, phi_h, a_h, err_h_arr[i] = \
                    time_step_1(method, HH_0h, CC_0h, xx, xin, t_a, t_b,
                        a_a, T_p, params, BC_D, BC_N, alp, 
                        G_Th, G_Ch, omega_s, omega_i, tol_a, tol_r, tol_g, 
                        max_iter)

                if i == 0:
                    HH_0h[:] = HH_h
                    CC_0h[:] = CC_h
                    t_a, t_b = th, t2
                    a_a = a_h

        else:
            HH_h, TT_h, CC_h, phi_h, a_h, err_h_arr[:] = \
                HH_f, TT_f, CC_f, phi_f, a_f, err_f
                
        err_est_1 = max(abs(HH_f - HH_h))
        err_est_2 = np.max(abs(CC_f - CC_h))
        if np.isnan(a_f) or np.isnan(a_h):
            err_est_3 = 0
        else:
            err_est_3 = abs(a_f - a_h)
        err_est = err_est_1 + err_est_2 + err_est_3


        z = tol_t/err_est
        if np.isnan(err_est):   # unstable step
            t2 = t1 + 0.5*dt
        elif z >= 1 and all(err_h_arr <= tol):   # successful step
            dtp = dt
            dt = min(dtp*min(0.8*z**(1/(q+1)), 2), dt_max)
            break
        elif z < 1 or any(err_h_arr > tol):   # failed step
            t2_old = t2
            t2 = t1 + 0.8*z**(1/q)*dt
            if t2 > t2_old:
                t2 = t1 + 0.8*(t2_old - t1)

            
    if n == max_t_iter:
        print('A. timestep: max. number of iterations reached')

    t3 = t2 + dt

    return HH_h, TT_h, CC_h, phi_h, a_h, t2, t3, err_est, n


