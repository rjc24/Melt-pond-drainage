# Number of cores to use
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from t_int import data_int
from Timestep import time_step_A
#from timeit import default_timer as timer


def data_call(run, method, HH_0, TT_0, CC_0, xx, xin, tt, tt_shots, 
              dt_max, a_0, T_p, params, BC_D, BC_N, alp,
              omega_s = 0.9, omega_i = 0.9, 
              tol_a = 1e-8, tol_r = 1e-8, tol_g = 1e-6, max_iter = 500,
              shots = True):
    
    #========================================================================
    # Function which calculates time-evolving solution at t = tt_shots
    # for given initial data / prescribed parameters
    #
    # Inputs:
    #    run - location for saving
    #    method - 'BTCS' (backwards Euler) or 'Crank' (Crank-Nicholson)
    #    HH_0 - enthalpy field at t = 0
    #    TT_0 - temperature field at t = 0
    #    CC_0 - bulk salinity field at t = 0
    #    xx, xin - spatial arrays
    #    tt - time array
    #    tt_shots - times to save data snapshots
    #    dt_max - maximum time step (initial)
    #    a_0 - initial interface position
    #    T_p - pond temperature
    #    params - dimensionless parameters
    #    BC_D, BC_N - Dirichlet, Neumann boundary conditions
    #    alp - boundary condition specifier
    #    omega_s, omega_i - relaxation parameters (Picard, interface position)
    #    tol_a, tol_r, tol_g - absolute, relative, interface tolerance
    #    max_iter - maximum number of iterations
    #    shots - save snapshots? True or False 
    #            (saves at every time step if False)
    #
    # Outputs:
    #    HH_all - enthalpy
    #    TT_all - temperature
    #    CC_all - bulk salinity
    #    pp_all - solid fraction
    #    a_n_all, a_p_all - interfaces
    #    tt_shots - time snapshots
    #
    #========================================================================

    path = 'data/'+run   # path for saving data
    Nx = np.size(xin)   # grid size
    t_max = tt_shots[-1]   # final time
    
    # arrays
    HH_all, TT_all, CC_all = map(np.zeros, 3*(Nx, ))
    HH_p, TT_p, CC_p = map(np.zeros, 3*(Nx, ))
    a_all = np.zeros(1)


    HH_all[:] = HH_0
    TT_all[:] = TT_0
    CC_all[:] = CC_0
    a_all[:] = a_0

    HH_p[:] = HH_0
    TT_p[:] = TT_0
    CC_p[:] = CC_0

    # Defining max. timestep
    # unpacking boundary conditions
    TT_BC, HH_BC, CC_BC, phi_BC = BC_D[:]
    FF_TT, FF_HH, FF_CC, FF_phi = BC_N[:]

    St, Cr, Pe = params[:]

    a = np.zeros(1)
    a[0] = a_0
    
    end = 'None'   # ending criterion

    i = -1   # starting loop
    t_count = 1
    while tt[-1] < t_max:

        i = i + 1
    
#        print('iteration', i)
        a_0 = a[i]
        t_1 = tt[i]
        t_2 = tt[i+1]
        
        HH, TT, CC, phi, a_n, t_1, t_2, err_est, n = \
            time_step_A('Crank', HH_p, CC_p, xx, xin, t_1, t_2, 
                  dt_max, a_0, T_p, params, BC_D, BC_N,
                  alp, omega_s = omega_s, omega_i = omega_i, 
                  tol_a = tol_a, tol_r = tol_r, tol_g = tol_g, 
                  max_iter = max_iter)

        tt[i+1] = t_1
        tt = np.concatenate([tt, [t_2]])
        a = np.concatenate([a, [a_n]])
    
        if np.mod(i,250) == 0:
            print('time:' , tt[i], ' of ', t_max, ' (', 100*tt[i]/t_max, ' percent complete)')
    
        # ending if timestep becomes too small
        if tt[-1] - tt[-2] < 1e-100:
            t_max = tt[-1]
            end = 'End : timestep went to zero'

        # ending if convergence fails
        if n == max_iter:
            t_max = tt[-1]
            end = 'End : failed to converge'    

        # ending if interface gets close to origin
        if np.size(HH[HH > CC]) == 3 and np.size(HH_p[HH_p > CC_p]) > 3:
            t_max = tt[-1] + 1e-8*t_max
            end = 'End : interface reached origin'

        # ending if interface leaves domain via right boundary
        if all(HH > CC) and not all(HH_p > CC_p):
            t_max = tt[-1] + 1e-4*t_max   # for n.n.
            end = 'End : interface reached right of domain'
    
    
        # ending if interface pos. is undefined
        if i > 10:
            if not all(np.isnan(a[-10:-1])):
                if all(np.isnan(a[-9:])):
                    t_max = tt[-1] + 1e-4*t_max
                    end = 'End : interface became undefined'
    
        if shots == True:
            if t_count < np.size(tt_shots)-1:
                if tt[-1] > tt_shots[t_count] and tt[-2] < tt_shots[t_count]:
                    HH_i, TT_i, CC_i, a_i = \
                        data_int(tt_shots[t_count], tt[-2], tt[-1], 
                                 xx, HH_p, HH, TT_p, TT, CC_p, CC, 
                                 a[-2], a[-1])
                    HH_all = np.vstack([HH_all, HH_i.copy()])
                    TT_all = np.vstack([TT_all, TT_i.copy()])
                    CC_all = np.vstack([CC_all, CC_i.copy()])
    
                    if not np.isnan(a_i):
                        a_all = np.vstack([a_all, [a_i.copy()]])
                    else:
                        a_all = np.vstack([a_all, [np.nan]])
                    t_count += 1
        elif shots == False:
            HH_all = np.vstack([HH_all, HH.copy()])
            TT_all = np.vstack([TT_all, TT.copy()])
            CC_all = np.vstack([CC_all, CC.copy()])
            
        # saving data in case of crash
        if np.mod(t_count, int(np.size(tt_shots)/10)) == 0:
            
            np.savetxt(path+'HH(t).csv', HH_all, delimiter = ',')
            np.savetxt(path+'TT(t).csv', TT_all, delimiter = ',')
            np.savetxt(path+'CC(t).csv', CC_all, delimiter = ',')
            np.savetxt(path+'a(t).csv', a_all, delimiter = ',')
            np.savetxt(path+'tt.csv', tt_shots, delimiter = ',')


        HH_p = HH[:]
        TT_p = TT[:]
        CC_p = CC[:]
    
    if end == 'None':
        end = 'End : t_max reached'
    print(end)

    # saving fields at t_max
    if tt[-1] > t_max and tt[-2] < t_max:
        HH_i, TT_i, CC_i, a_i = \
            data_int(tt_shots[t_count], tt[-2], tt[-1], xx, 
                     HH_p, HH, TT_p, TT, CC_p, CC, a[-2], a[-1])
        HH_all = np.vstack([HH_all, HH_i.copy()])
        TT_all = np.vstack([TT_all, TT_i.copy()])
        CC_all = np.vstack([CC_all, CC_i.copy()])

        if not np.isnan(a_i):
            a_all = np.vstack([a_all, [a_i.copy()]])
        else:
            a_all = np.vstack([a_all, [np.nan]])
            
    # unpacking values for saving
    # non-dim parameters
    St, Cr, Pe = params[:]

    # unpacking boundary conditions
    TT_BC, HH_BC, CC_BC, phi_BC = BC_D[:]
    FF_TT, FF_HH, FF_CC, FF_phi = BC_N[:]


    path_cont = 'arrays/cont/'+run

    # scalars
    a_file = open(path_cont+'a.asc', 'w+')
    t_file = open(path_cont+'t.asc', 'w+')
    dt_file = open(path_cont+'dt_max.asc', 'w+')
    St_file = open(path_cont+'St.asc', 'w+')
    Cr_file = open(path_cont+'Cr.asc', 'w+')    
    Pe_file = open(path_cont+'Pe.asc', 'w+')    
    T_p_file = open(path_cont+'T_p.asc', 'w+')
    Nx_file = open(path_cont+'Nx.asc', 'w+')
    
    # arrays
    xx_file = open(path_cont+'xx.csv', 'w+')
    xin_file = open(path_cont+'xin.csv', 'w+')
    TT_file = open(path_cont+'TT_0.csv', 'w+')
    HH_file = open(path_cont+'HH_0.csv', 'w+')
    CC_file = open(path_cont+'CC_0.csv', 'w+')
    phi_file = open(path_cont+'phi_0.csv', 'w+')
   
    # boundaries
    # settings
    alp_file = open(path_cont+'alp.csv', 'w+')

    # Dirichlet
    T_L_file = open('arrays/T_L.asc', 'w+')
    T_R_file = open('arrays/T_R.asc', 'w+')
    H_L_file = open('arrays/H_L.asc', 'w+')
    H_R_file = open('arrays/H_R.asc', 'w+')
    C_L_file = open('arrays/C_L.asc', 'w+')
    C_R_file = open('arrays/C_R.asc', 'w+')
    phi_L_file = open('arrays/phi_L.asc', 'w+')
    phi_R_file = open('arrays/phi_R.asc', 'w+')
    
    # Neumann
    FT_L_file = open('arrays/FT_L.asc', 'w+')
    FT_R_file = open('arrays/FT_R.asc', 'w+')
    FH_L_file = open('arrays/FH_L.asc', 'w+')
    FH_R_file = open('arrays/FH_R.asc', 'w+')
    FC_L_file = open('arrays/FC_L.asc', 'w+')
    FC_R_file = open('arrays/FC_R.asc', 'w+')
    Fphi_L_file = open('arrays/Fphi_L.asc', 'w+')
    Fphi_R_file = open('arrays/Fphi_R.asc', 'w+')
    
    
    # saving
    # fields
    np.savetxt(xx_file, xx, delimiter = ',')
    np.savetxt(xin_file, xin, delimiter = ',')
    np.savetxt(TT_file, TT, delimiter = ',')
    np.savetxt(HH_file, HH, delimiter = ',')
    np.savetxt(CC_file, CC, delimiter = ',')
    np.savetxt(phi_file, phi, delimiter = ',')
    
    # scalars    
    print(a[-2], file = a_file)
    print(tt[-2], file = t_file)
    print(dt_max, file = dt_file)
    print(St, file = St_file)
    print(Cr, file = Cr_file)
    print(Pe, file = Pe_file)
    print(T_p, file = T_p_file)
    print(Nx, file = Nx_file)

    # boundaries
    # settings
    np.savetxt(alp_file, alp, delimiter = ',')
    
    # Dirichlet
    print(TT_BC[0], file = T_L_file)
    print(TT_BC[1], file = T_R_file)
    print(HH_BC[0], file = H_L_file)
    print(HH_BC[1], file = H_R_file)
    print(CC_BC[0], file = C_L_file)
    print(CC_BC[1], file = C_R_file)
    print(phi_BC[0], file = phi_L_file)
    print(phi_BC[1], file = phi_R_file)

    # Neumann
    print(FF_TT[0], file = FT_L_file)
    print(FF_TT[1], file = FT_R_file)
    print(FF_HH[0], file = FH_L_file)
    print(FF_HH[1], file = FH_R_file)
    print(FF_CC[0], file = FC_L_file)
    print(FF_CC[1], file = FC_R_file)
    print(FF_phi[0], file = Fphi_L_file)
    print(FF_phi[1], file = Fphi_R_file)
    
    # closing
    # scalars
    t_file.close()
    dt_file.close()
    St_file.close()
    Cr_file.close()
    Pe_file.close()
    T_p_file.close()
    Nx_file.close()
    
    # fields
    xx_file.close()
    xin_file.close()
    TT_file.close()
    HH_file.close()
    CC_file.close()
    phi_file.close()
    
    # boundaries
    # settings
    alp_file.close()

    # Dirichlet
    T_L_file.close()
    T_R_file.close()
    H_L_file.close()
    H_R_file.close()
    C_L_file.close()
    C_R_file.close()
    phi_L_file.close()
    phi_R_file.close()
    
    # Neumann
    FT_L_file.close()
    FT_R_file.close()
    FH_L_file.close()
    FH_R_file.close()
    FC_L_file.close()
    FC_R_file.close()
    Fphi_L_file.close()
    Fphi_R_file.close()

    
    return HH_all, TT_all, CC_all, a_all, tt_shots

