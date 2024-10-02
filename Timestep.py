# Number of cores to use
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from NL_solve import NL_solve

def time_step_1(method, HH_0, CC_0, pp_0, xx, xin, zz, zin, t1, t2, 
                params, BC_D, BC_N, alp, D_n, G_T = None, 
                G_C = None, G_p = None, gamma = 0, omega_s = 1, 
                tol_a = 1e-8, tol_r = 1e-8, max_iter = 100):

    #========================================================================
    # Function which calculates solution at t = t2 from data at t = t1
    #
    # Inputs:
    #    method - 'BTCS' (backwards Euler) or 'Crank' (Crank-Nicholson)
    #    HH_0 - enthalpy field at t = t1
    #    CC_0 - bulk salinity field at t = t1
    #    pp_0 - pressure field at t =t1
    #    xx, xin, zz, zin - spatial arrays
    #    t1, t2 - time steps
    #    params - dimensionless parameters
    #    BC_D, BC_N - Dirichlet, Neumann boundary conditions
    #    alp - boundary condition specifier
    #    D_n - numerical diffusivity
    #    G_T, G_C, G_p - source terms (heat, solute, pressure)
    #    gamma - method selector (Picard = 0, Newton = 1)
    #    omega_s - relaxation parameter (default value = 1)
    #    tol_a, tol_r - absolute, relative tolerance
    #    max_iter - maximum number of iterations (calls to Picard / Newton)
    #
    # Outputs:
    #    HH - enthalpy at t = t2
    #    TT - temperature at t = t2
    #    CC - bulk salinity at t = t2
    #    phi - solid fraction at t = t2
    #    pp - pressure at t = t2
    #    u_b, w_b - horizontal, vertical velocities at t = t2
    #    err - residual from Picard / Newton method
    #========================================================================
    
    HH_p, CC_p, pp_p = map(np.zeros, 3*(np.shape(HH_0), ))

    # data from prev. time step
    HH_p[:] = HH_0
    CC_p[:] = CC_0
    pp_p[:] = pp_0
    err = 1
    tol = tol_a

    count = 0
    NL_max = 100   # max iter for Picard / Newton
    while err > tol:
        count = count + 1
        HH, TT, CC, phi, pp, u_b, w_b, err = \
            NL_solve(method, HH_0, HH_p, CC_0, CC_p, pp_0, pp_p, xx, xin, zz, zin, 
                     t1, t2, params, BC_D, BC_N, alp, D_n, 
                     G_T, G_C, G_p, gamma, omega_s, tol_a, tol_r, 
                     max_iter = NL_max)

        if np.mod(count,250) == 0:
            print(count)
        HH_p[:] = HH
        CC_p[:] = CC
        pp_p[:] = pp


        if count == max_iter:
            if err > tol:
                print('NL_solve not converging')
            break

    return HH, TT, CC, phi, pp, u_b, w_b, err    #  n_temp, a_temp,


def time_step_A(method, HH_0, CC_0, pp_0, xx, xin, xx_p, zz, zin, zz_p, 
              t1, t2, dt_max, params, BC_D, BC_N, 
              alp, D_n, gamma = 0, omega_s = 1, 
              tol_a = 1e-8, tol_r = 1e-8, max_iter = 100):

    #========================================================================
    # Adaptive time-stepper. Calculates solution at time t2 using data from
    # time t1 in one full step and two half steps. Procedure is repeated until
    # desired tolerance is met.
    #
    # Inputs:
    #    method - 'BTCS' (backwards Euler) or 'Crank' (Crank-Nicholson)
    #    HH_0 - enthalpy field at t = t1
    #    CC_0 - bulk salinity field at t = t1
    #    pp_0 - pressure field at t =t1
    #    xx, xin, xx_p, zz, zin, zz_p - spatial arrays
    #    t1, t2 - time steps
    #    dt_max - maximum time step (initial)
    #    params - dimensionless parameters
    #    BC_D, BC_N - Dirichlet, Neumann boundary conditions
    #    alp - boundary condition specifier
    #    D_n - numerical diffusivity
    #    gamma - method selector (Picard = 0, Newton = 1)
    #    omega_s - relaxation parameter (default value = 1)
    #    tol_a, tol_r - absolute, relative tolerance
    #    max_iter - maximum number of iterations (calls to Picard / Newton)
    #
    # Outputs:
    #    HH_h - enthalpy at t = t2
    #    TT_h - temperature at t = t2
    #    CC_h - bulk salinity at t = t2
    #    phi_h - solid fraction at t = t2
    #    pp_h - pressure at t = t2
    #    u_bh, w_bh - horizontal, vertical velocities at t = t2
    #    t2, t3 - time of solution, next time to calculate
    #    err_est, n - estimated error, number of iterations until tolerance met
    #
    #========================================================================
    

    HH_0h, CC_0h, pp_0h = map(np.zeros, 3*(np.shape(HH_0), ))
    err_h_arr = np.zeros(2)

    max_t_iter = 15   # maximum no. of attempts
    n = 0   # counter
    tol = tol_a# + tol_r*norm(HH_0)   # Tolerance
    if method == 'BTCS':   # setting theta
        theta = 1
    elif method == 'Crank':
        theta = 0.5
    q = 1/theta   # defining q


    dx, dz = xx[1, 0] - xx[0, 0], zz[0, 1] - zz[0, 0]

    while n < max_t_iter:
        n = n + 1
        dt = t2 - t1   # full timestep
        th = t1 + 0.5*dt   # half-time
        print('Adaptive time step : ', n)
        if any(n == np.array([3, 6])):
            tol = 10*tol   # adjusting tolerance if struggling to converge
        # Setting temporal error tolerance
        if method == 'BTCS':
            tol_t = 2*np.fmax(dz, dx)
        elif method == 'Crank':
            tol_t = 2*np.fmax(dz**2, dx**2)


        # zero source terms
        G_Tf, G_Cf, G_pf = map(np.zeros, 3*(np.shape(HH_0), ))

        # full step
        HH_f, TT_f, CC_f, phi_f, pp_f, u_bf, w_bf, err_f = \
            time_step_1(method, HH_0, CC_0, pp_0, xx, xin, zz, zin, t1, t2, 
                        params, BC_D, BC_N, alp, D_n, G_Tf, G_Cf, 
                        G_pf, gamma, omega_s, tol_a, tol_r, max_iter)


        if ~np.isnan(HH_f).any():
            # two half steps
            t_a, t_b = t1, th
            HH_0h[:] = HH_0
            CC_0h[:] = CC_0
            pp_0h[:] = pp_0
            for i in range(2):
            
                G_Th, G_Ch, G_ph = map(np.zeros, 3*(np.shape(HH_0), ))
    
                HH_h, TT_h, CC_h, phi_h, pp_h, u_bh, w_bh, err_h_arr[i] = \
                    time_step_1(method, HH_0h, CC_0h, pp_0h, xx, xin, zz, zin, 
                            t_a, t_b, params, 
                            BC_D, BC_N, alp, D_n, G_Th, G_Ch, G_ph, gamma, 
                            omega_s, tol_a, tol_r, max_iter)
    
                if i == 0:
                    HH_0h[:] = HH_h
                    CC_0h[:] = CC_h
                    pp_0h[:] = pp_h
                    t_a, t_b = th, t2

        else:
            HH_h, TT_h, CC_h, phi_h, pp_h, u_bh, w_bh, err_h_arr[:] = \
                HH_f, TT_f, CC_f, phi_f, pp_f, u_bf, w_bf, err_f

        # estimating error
        err_est_1 = np.max(abs(HH_f - HH_h))
        err_est_2 = np.max(abs(CC_f - CC_h))
        
        err_est = err_est_1 + err_est_2
        
        
        z = tol_t/err_est
        if np.isnan(err_est):   # unstable step
            t2 = t1 + 0.5*dt
            print('step size unstable')
        elif z >= 1 and all(err_h_arr <= tol):   # successful step
            dtp = dt
            dt = min(dtp*min(0.8*z**(1/(q+1)), 2), dt_max)
            print('step size', dt, 'estimated error = ', err_est)
            break
        elif z < 1 or any(err_h_arr > tol):   # failed step
            t2_old = t2
            t2 = t1 + 0.8*z**(1/q)*dt
            if t2 > t2_old:
                t2 = t1 + 0.8*(t2_old - t1)
            print('failed step, estimated error = ', err_est)


    if n == max_t_iter:
        print('A. timestep: max. number of iterations reached')

    t3 = t2 + dt

    return HH_h, TT_h, CC_h, phi_h, pp_h, u_bh, w_bh, t2, t3, err_est, n


