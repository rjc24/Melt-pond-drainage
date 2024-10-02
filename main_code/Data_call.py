# Number of cores to use
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from t_int import data_int
from Timestep import time_step_A
from Enthalpy_functions import SFrac_calc, Perm_calc
from elliptic_solve_sparse import velocities
from interface import int_loc
import num_params as npar
#from timeit import default_timer as timer


def data_call(run, method, HH_0, TT_0, CC_0, pp_0, xx, xin, xx_p, 
              zz, zin, zz_p, tt, tt_shots, 
              dt_max, a_0, AR, params, BC_D, BC_N, alp, D_n, 
              gamma = 0, omega_s = 0.9, 
              tol_a = 1e-8, tol_r = 1e-8, max_iter = 500,
              shots = True, cont = 0):

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
    #    pp_0 - pressure field at t = 0
    #    xx, xin, xx_p, zz, zin, zz_p - spatial arrays
    #    tt - time array
    #    tt_shots - times to save data snapshots
    #    dt_max - maximum time step (initial)
    #    a_0 - initial interface position
    #    AR - domain aspect ratio
    #    params - dimensionless parameters
    #    BC_D, BC_N - Dirichlet, Neumann boundary conditions
    #    alp - boundary condition specifier
    #    D_n - numerical diffusivity
    #    G_T, G_C, G_p - source terms (heat, solute, pressure)
    #    gamma - method selector (Picard = 0, Newton = 1)
    #    omega_s - relaxation parameter (default value = 1)
    #    tol_a, tol_r - absolute, relative tolerance
    #    max_iter - maximum number of iterations
    #    shots - save snapshots? True or False 
    #            (saves at every time step if False)
    #    cont - continuing from previous run? 0 (no) or 1 (yes)
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
    path_cont = 'arrays/cont/'+run   # path for continuing run
    
    Nx, Nz = np.shape(xin)   # grid size
    N = np.array([Nx, Nz])
    t_max = tt_shots[-1]   # final time
    
    # unpacking boundary conditions
    alp_T, alp_p = alp[:]
    TT_BC, HH_BC, CC_BC, phi_BC, pp_BC = BC_D[:]
    FF_TT, FF_HH, FF_CC, FF_phi, FF_pp = BC_N[:]
    
    # unpacking parameters
    St, Cr, Pe, Da_h = params[:]
    
    
    # opening files for saving
    # scalars
    St_file = open(path_cont+'St.asc', 'w+')
    Cr_file = open(path_cont+'Cr.asc', 'w+')
    Pe_file = open(path_cont+'Pe.asc', 'w+')
    Da_h_file = open(path_cont+'Da_h.asc', 'w+')
    AR_file = open(path_cont+'AR.asc', 'w+')
    D_n_file = open(path_cont+'D_n.asc', 'w+')
    
    # arrays
    xx_file = open(path_cont+'xx.csv', 'w+')
    xin_file = open(path_cont+'xin.csv', 'w+')
    zz_file = open(path_cont+'zz.csv', 'w+')
    zin_file = open(path_cont+'zin.csv', 'w+')
    N_file = open(path_cont+'N.csv', 'w+')
    
    # boundaries
    # settings
    alp_T_file = open(path_cont+'alp_T.csv', 'w+')
    alp_p_file = open(path_cont+'alp_p.csv', 'w+')

    # Dirichlet
    T_L_file = open(path_cont+'T_L.csv', 'w+')
    T_R_file = open(path_cont+'T_R.csv', 'w+')
    T_B_file = open(path_cont+'T_B.csv', 'w+')
    T_T_file = open(path_cont+'T_T.csv', 'w+')
    H_L_file = open(path_cont+'H_L.csv', 'w+')
    H_R_file = open(path_cont+'H_R.csv', 'w+')
    H_B_file = open(path_cont+'H_B.csv', 'w+')
    H_T_file = open(path_cont+'H_T.csv', 'w+')
    C_L_file = open(path_cont+'C_L.csv', 'w+')
    C_R_file = open(path_cont+'C_R.csv', 'w+')
    C_B_file = open(path_cont+'C_B.csv', 'w+')
    C_T_file = open(path_cont+'C_T.csv', 'w+')
    phi_L_file = open(path_cont+'phi_L.csv', 'w+')
    phi_R_file = open(path_cont+'phi_R.csv', 'w+')
    phi_B_file = open(path_cont+'phi_B.csv', 'w+')
    phi_T_file = open(path_cont+'phi_T.csv', 'w+')
    p_L_file = open(path_cont+'p_L.csv', 'w+')
    p_R_file = open(path_cont+'p_R.csv', 'w+')
    p_B_file = open(path_cont+'p_B.csv', 'w+')
    p_T_file = open(path_cont+'p_T.csv', 'w+')

    # Neumann
    FT_L_file = open(path_cont+'FT_L.csv', 'w+')
    FT_R_file = open(path_cont+'FT_R.csv', 'w+')
    FT_B_file = open(path_cont+'FT_B.csv', 'w+')
    FT_T_file = open(path_cont+'FT_T.csv', 'w+')
    FH_L_file = open(path_cont+'FH_L.csv', 'w+')
    FH_R_file = open(path_cont+'FH_R.csv', 'w+')
    FH_B_file = open(path_cont+'FH_B.csv', 'w+')
    FH_T_file = open(path_cont+'FH_T.csv', 'w+')
    FC_L_file = open(path_cont+'FC_L.csv', 'w+')
    FC_R_file = open(path_cont+'FC_R.csv', 'w+')
    FC_B_file = open(path_cont+'FC_B.csv', 'w+')
    FC_T_file = open(path_cont+'FC_T.csv', 'w+')
    Fphi_L_file = open(path_cont+'Fphi_L.csv', 'w+')
    Fphi_R_file = open(path_cont+'Fphi_R.csv', 'w+')
    Fphi_B_file = open(path_cont+'Fphi_B.csv', 'w+')
    Fphi_T_file = open(path_cont+'Fphi_T.csv', 'w+')
    Fp_L_file = open(path_cont+'Fp_L.csv', 'w+')
    Fp_R_file = open(path_cont+'Fp_R.csv', 'w+')
    Fp_B_file = open(path_cont+'Fp_B.csv', 'w+')
    Fp_T_file = open(path_cont+'Fp_T.csv', 'w+')
    
    # saving
    # scalars
    print(St, file = St_file)
    print(Cr, file = Cr_file)
    print(Pe, file = Pe_file)
    print(Da_h, file = Da_h_file)
    print(AR, file = AR_file)
    print(D_n, file = D_n_file)
    
    # arrays
    np.savetxt(xx_file, xx, delimiter = ',')
    np.savetxt(xin_file, xin, delimiter = ',')
    np.savetxt(zz_file, zz, delimiter = ',')
    np.savetxt(zin_file, zin, delimiter = ',')
    np.savetxt(N_file, N, delimiter = ',')
    
    # boundaries
    # settings
    np.savetxt(alp_T_file, alp_T, delimiter = ',')
    np.savetxt(alp_p_file, alp_p, delimiter = ',')
    
    # Dirichlet
    np.savetxt(T_L_file, TT_BC[0], delimiter = ',')
    np.savetxt(T_R_file, TT_BC[1], delimiter = ',')
    np.savetxt(T_B_file, TT_BC[2], delimiter = ',')
    np.savetxt(T_T_file, TT_BC[3], delimiter = ',')
    np.savetxt(H_L_file, HH_BC[0], delimiter = ',')
    np.savetxt(H_R_file, HH_BC[1], delimiter = ',')
    np.savetxt(H_B_file, HH_BC[2], delimiter = ',')
    np.savetxt(H_T_file, HH_BC[3], delimiter = ',')
    np.savetxt(C_L_file, CC_BC[0], delimiter = ',')
    np.savetxt(C_R_file, CC_BC[1], delimiter = ',')
    np.savetxt(C_B_file, CC_BC[2], delimiter = ',')
    np.savetxt(C_T_file, CC_BC[3], delimiter = ',')
    np.savetxt(phi_L_file, phi_BC[0], delimiter = ',')
    np.savetxt(phi_R_file, phi_BC[1], delimiter = ',')
    np.savetxt(phi_B_file, phi_BC[2], delimiter = ',')
    np.savetxt(phi_T_file, phi_BC[3], delimiter = ',')
    np.savetxt(p_L_file, pp_BC[0], delimiter = ',')
    np.savetxt(p_R_file, pp_BC[1], delimiter = ',')
    np.savetxt(p_B_file, pp_BC[2], delimiter = ',')
    np.savetxt(p_T_file, pp_BC[3], delimiter = ',')

    # Neumann
    np.savetxt(FT_L_file, FF_TT[0], delimiter = ',')
    np.savetxt(FT_R_file, FF_TT[1], delimiter = ',')
    np.savetxt(FT_B_file, FF_TT[2], delimiter = ',')
    np.savetxt(FT_T_file, FF_TT[3], delimiter = ',')
    np.savetxt(FH_L_file, FF_HH[0], delimiter = ',')
    np.savetxt(FH_R_file, FF_HH[1], delimiter = ',')
    np.savetxt(FH_B_file, FF_HH[2], delimiter = ',')
    np.savetxt(FH_T_file, FF_HH[3], delimiter = ',')
    np.savetxt(FC_L_file, FF_CC[0], delimiter = ',')
    np.savetxt(FC_R_file, FF_CC[1], delimiter = ',')
    np.savetxt(FC_B_file, FF_CC[2], delimiter = ',')
    np.savetxt(FC_T_file, FF_CC[3], delimiter = ',')
    np.savetxt(Fphi_L_file, FF_phi[0], delimiter = ',')
    np.savetxt(Fphi_R_file, FF_phi[1], delimiter = ',')
    np.savetxt(Fphi_B_file, FF_phi[2], delimiter = ',')
    np.savetxt(Fphi_T_file, FF_phi[3], delimiter = ',')
    np.savetxt(Fp_L_file, FF_pp[0], delimiter = ',')
    np.savetxt(Fp_R_file, FF_pp[1], delimiter = ',')
    np.savetxt(Fp_B_file, FF_pp[2], delimiter = ',')
    np.savetxt(Fp_T_file, FF_pp[3], delimiter = ',')
    
    
    # closing
    # scalars
    St_file.close()
    Cr_file.close()
    Pe_file.close()
    Da_h_file.close()
    AR_file.close()
    D_n_file.close()
    
    # arrays
    xx_file.close()
    xin_file.close()
    zz_file.close()
    zin_file.close()
    N_file.close()
    
    # boundaries
    # settings
    alp_T_file.close()
    alp_p_file.close()

    # Dirichlet
    T_L_file.close()
    T_R_file.close()
    T_B_file.close()
    T_T_file.close()
    H_L_file.close()
    H_R_file.close()
    H_B_file.close()
    H_T_file.close()
    C_L_file.close()
    C_R_file.close()
    C_B_file.close()
    C_T_file.close()
    phi_L_file.close()
    phi_R_file.close()
    phi_B_file.close()
    phi_T_file.close()
    p_L_file.close()
    p_R_file.close()
    p_B_file.close()
    p_T_file.close()
    
    # Neumann
    FT_L_file.close()
    FT_R_file.close()
    FT_B_file.close()
    FT_T_file.close()
    FH_L_file.close()
    FH_R_file.close()
    FH_B_file.close()
    FH_T_file.close()
    FC_L_file.close()
    FC_R_file.close()
    FC_B_file.close()
    FC_T_file.close()
    Fphi_L_file.close()
    Fphi_R_file.close()
    Fphi_B_file.close()
    Fphi_T_file.close()
    Fp_L_file.close()
    Fp_R_file.close()
    Fp_B_file.close()
    Fp_T_file.close()


    # new run
    if cont == 0:
        HH_all, TT_all, CC_all, pp_all = \
            map(np.zeros, 4*(np.shape(HH_0), ))
        a_n_all, a_p_all = map(np.zeros, 2*([2, Nz], ))
    
        a_n0, a_p0 = a_0[:]   # unpacking initial interface positions
    
        HH_all[:] = HH_0
        TT_all[:] = TT_0
        CC_all[:] = CC_0
        pp_all[:] = pp_0
        a_n_all[:] = a_n0
        a_p_all[:] = a_p0

    # continued run
    elif cont == 1:
        HH_all = np.load(path+'HH(t).npy')
        TT_all = np.load(path+'TT(t).npy')
        CC_all = np.load(path+'CC(t).npy')
        pp_all = np.load(path+'pp(t).npy')
        a_n_all = np.load(path+'a_n(t).npy')
        a_p_all = np.load(path+'a_p(t).npy')
        Nt_old = np.shape(HH_all)[2]

    HH_p, TT_p, CC_p, pp_p = map(np.zeros, 4*(np.shape(HH_0), ))
    HH_p[:], TT_p[:], CC_p[:], pp_p[:] = HH_0, TT_0, CC_0, pp_0

    # setting max. timestep
    phi_0 = SFrac_calc(HH_0, CC_0, St, Cr)
    Pi, Pi_bh, Pi_bv = Perm_calc(phi_0, xx, xin, zz, zin, phi_BC, FF_phi, alp_T, Da_h)
    u_b0, w_b0 = velocities(xx, xin, zz, zin, pp_0, Pi_bh, Pi_bv, pp_BC, FF_pp, alp_p,
                            Pe)

    xi = 0.5
    dx, dz = xx[1, 0] - xx[0, 0], zz[0, 1] - zz[0, 0]
    dt_max = xi*min(abs(dx/np.max(u_b0)), abs(dz/np.max(w_b0)))
#    dt_max = xi/(np.max(abs(u_b0))/dx + np.max(abs(w_b0))/dz)

    end = 'None'   # ending criterion

    if cont == 0:
        i = -1   # starting loop
        t_count = 1
    elif cont == 1:
        i = Nt_old - 2
        t_count = Nt_old
    while tt[-1] < t_max:
    #for i in range(167):
        i = i + 1

        print('iteration', i)

        t_1 = tt[i]
        t_2 = tt[i+1]

        # one time step
        HH, TT, CC, phi, pp, u_b, w_b, t_1, t_2, err_est, n = \
            time_step_A('Crank', HH_p, CC_p, pp_p, xx, xin, xx_p, 
                  zz, zin, zz_p, t_1, t_2, 
                  dt_max, params, BC_D, BC_N, alp, D_n, 
                  gamma = gamma, omega_s = omega_s,
                  tol_a = tol_a, tol_r = tol_r, max_iter = max_iter)

        tt[i+1] = t_1
        tt = np.concatenate([tt, [t_2]])

        if np.mod(i,25) == 0:
            print('time:' , tt[i], ' of ', t_max, ' (', 100*tt[i]/t_max, ' percent complete)')
    
        # ending if all ice melts
        if np.all(HH > CC) and ~np.all(HH_p > CC_p):
            t_max = tt_shots[t_count+1]
            end = 'End : all ice melted'

        # ending if timestep becomes too small
        if tt[-1] - tt[-2] < 1e-100:
            t_max = tt[-1]
            end = 'End : timestep went to zero'

        # ending if convergence fails
        if n == max_iter:
            t_max = tt[-1]
            end = 'End : failed to converge'

        # saving snapshot
        if shots == True:
            if t_count < np.size(tt_shots)-1:
                while tt[-1] > tt_shots[t_count] and tt[-2] < tt_shots[t_count]:
                    HH_i, TT_i, CC_i, pp_i = \
                        data_int(tt_shots[t_count], tt[-2], tt[-1], 
                                 HH_p, HH, TT_p, TT, CC_p, CC, pp_p, pp)
                    a_in, a_ip = int_loc(HH_i, CC_i, xx, zz, xx_p, zz_p)
    
                    HH_all = np.dstack([HH_all, HH_i.copy()])
                    TT_all = np.dstack([TT_all, TT_i.copy()])
                    CC_all = np.dstack([CC_all, CC_i.copy()])
                    pp_all = np.dstack([pp_all, pp_i.copy()])
                    a_n_all = np.dstack([a_n_all, a_in.copy()])
                    a_p_all = np.dstack([a_p_all, a_ip.copy()])
    #                a_all = np.vstack([a_all, a_i.copy()])
                    print(t_count)
                    t_count += 1
                    if t_count == np.size(tt_shots) - 1:
                        break
                    print('Saving, t_count = ', t_count, ', time = ', tt_shots[t_count])
        elif shots == False:
            a_n, a_p = int_loc(HH, CC, xx, zz, xx_p, zz_p)
            HH_all = np.dstack([HH_all, HH.copy()])
            TT_all = np.dstack([TT_all, TT.copy()])
            CC_all = np.dstack([CC_all, CC.copy()])
            pp_all = np.dstack([pp_all, pp.copy()])
            a_n_all = np.dstack([a_n_all, a_n.copy()])
            a_p_all = np.dstack([a_p_all, a_p.copy()])

        # saving data in case of crash
        if np.mod(t_count, int(np.size(tt_shots)/20)) == 0:
            
            np.save(path+'HH(t).npy', HH_all)
            np.save(path+'TT(t).npy', TT_all)
            np.save(path+'CC(t).npy', CC_all)
            np.save(path+'pp(t).npy', pp_all)
            np.save(path+'a_n(t).npy', a_n_all)
            np.save(path+'a_p(t).npy', a_p_all)
            np.save(path+'tt.npy', tt_shots)
            
        # redefining dt_max
        dt_max = xi*min(abs(dx/np.max(u_b)), abs(dz/np.max(w_b)))

        # updating arrays
        HH_p = HH[:]
        TT_p = TT[:]
        CC_p = CC[:]
        pp_p[:] = pp[:]

    # ending if max time reached    
    if end == 'None':
        end = 'End : t_max reached'
    print(end)

    # saving fields at t_max
    if tt[-1] > t_max and tt[-2] < t_max:
        HH_i, TT_i, CC_i, pp_i = \
            data_int(t_max, tt[-2], tt[-1], 
                     HH_p, HH, TT_p, TT, CC_p, CC, pp_p, pp)
        a_in, a_ip = int_loc(HH_i, CC_i, xx, zz, xx_p, zz_p)

        HH_all = np.dstack([HH_all, HH_i.copy()])
        TT_all = np.dstack([TT_all, TT_i.copy()])
        CC_all = np.dstack([CC_all, CC_i.copy()])
        pp_all = np.dstack([pp_all, pp_i.copy()])
        a_n_all = np.dstack([a_n_all, a_in.copy()])
        a_p_all = np.dstack([a_p_all, a_ip.copy()])


    # unpacking boundary conditions
    alp_T, alp_p = alp[:]
    TT_BC, HH_BC, CC_BC, phi_BC, pp_BC = BC_D[:]
    FF_TT, FF_HH, FF_CC, FF_phi, FF_pp = BC_N[:]
    
    # final interface positions
    a_n, a_p = int_loc(HH, CC, xx, zz, xx_p, zz_p)

    # scalars
    t_file = open(path_cont+'t.asc', 'w+')
    dt_file = open(path_cont+'dt_max.asc', 'w+')

    # arrays
    TT_file = open(path_cont+'TT_0.csv', 'w+')
    HH_file = open(path_cont+'HH_0.csv', 'w+')
    CC_file = open(path_cont+'CC_0.csv', 'w+')
    phi_file = open(path_cont+'phi_0.csv', 'w+')
    pp_file = open(path_cont+'pp_0.csv', 'w+')
    a_n_file = open(path_cont+'a_n.csv', 'w+')
    a_p_file = open(path_cont+'a_p.csv', 'w+')
    
    
    # saving
    # scalars
    print(tt[-2], file = t_file)
    print(dt_max, file = dt_file)
    
    # fields
    np.savetxt(TT_file, TT, delimiter = ',')
    np.savetxt(HH_file, HH, delimiter = ',')
    np.savetxt(CC_file, CC, delimiter = ',')
    np.savetxt(phi_file, phi, delimiter = ',')
    np.savetxt(pp_file, pp, delimiter = ',')
    np.savetxt(a_n_file, a_n[-1], delimiter = ',')
    np.savetxt(a_p_file, a_p[-1], delimiter = ',')
    
    # closing
    # scalars
    t_file.close()
    dt_file.close()
    
    # fields
    TT_file.close()
    HH_file.close()
    CC_file.close()
    phi_file.close()
    pp_file.close()
    a_n_file.close()
    a_p_file.close()

    return HH_all, TT_all, CC_all, pp_all, a_n_all, a_p_all, tt_shots

