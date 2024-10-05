# Number of cores to use
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from scipy.linalg import solve
from scipy.optimize import newton
from Enthalpy_functions import Ent_calc, SFrac_calc, Perm_calc
from elliptic_solve_sparse import p_solve#, velocities
import phys_params as ppar
import num_params as npar
from vertical_profile import F_z, dF_z, dF_zdz
from interface import int_loc

# Example initial conditions (fixed salinity)

def init(run_enter = None, St = None, Cr = None, Pe = None, Da_h = None, 
         T_p = None, C_p = None, AR = None, Nz = None, j = None, BC_T = None, 
         BC_p = None, t_run = None):
    
    # IC file
    IC = 1
    
    # Setting file location
    if run_enter == None:
        run_enter = input('Directory for storing data : ')
    run = run_enter+'/'
        
    # Physical parameters (dimensional, changeable)
    C_f = ppar.C_f   # Liquidus slope, bulk salinity
    H_f_d = ppar.H_f   # Freezing enthalpy
    T_L_d_sc, T_R_d_sc = ppar.T_L, ppar.T_R   # Channel (pond) temp
    C_T_d_sc, C_B_d_sc = ppar.C_T, ppar.C_B
    C_l_R = ppar.C_l_R
    del_T = H_f_d - T_R_d_sc  #  temperature diff
    del_C = C_f - C_l_R   # concentration diff
    C_O = (C_B_d_sc - C_f)/del_C
    
    if St == None:
        St = float(input('Enter Stefan number (default = L/(c_p*del_T)) : ') or ppar.St)
    if Cr == None:
        Cr = float(input('Enter Concentration ratio (default = -C_f/del_C) : ') or ppar.Cr)
    if Pe == None:
        Pe = float(input('Enter max Peclet number (default = {}) : '.format(ppar.Pe_m)) or ppar.Pe_m)
    if Da_h == None:
        Da_h = float(input('Enter Hele-Shaw Darcy number (default = {}) : '.format(ppar.Da_h)) or ppar.Da_h)
    if T_p == None:
        T_p = float(input('Enter pond temperature (default = {}) : '.format(ppar.T_p)) or ppar.T_p)
    if AR == None:
        AR = float(input('Enter domain aspect ratio (default 0.1) : ') or 0.1)
    
    # dimensionless parameters (changeable)
    T_L_sc, T_R_sc = (T_L_d_sc - H_f_d)/del_T, (T_R_d_sc - H_f_d)/del_T
    T_T_sc, T_B_sc = del_T*T_p + H_f_d, -2.0   # temperatures at top/bottom of domain (dimesional) 0.1
    ice_t = 0.9   # top of ice
    ice_b = 0.1   # bottom of ice
    T_0 = (T_L_d_sc - H_f_d)/del_T
    
    # numerical parameters
    if Nz == None:
        Nz = int(input('Enter vertical mesh size (number of z cells) : '))
    if np.mod(Nz, 2) == 1:
        Nz = Nz + 1

    # defining Nx s.t. cell aspect ratio is ~1
    Nx = int(2*AR*Nz//1)
    if np.mod(Nx, 2) == 1:
        Nx = Nx + 1
    N = np.array([Nx, Nz])   # total number of cells

    # cell faces
    x_f = np.linspace(-AR, AR, Nx+1)
    z_f = np.linspace(0, 1, Nz+1)
    # cell centres
    x_c, z_c = (x_f[:-1] + x_f[1:])/2, (z_f[:-1] + z_f[1:])/2
    
    # grid
    xx_v, zz_v = np.meshgrid(x_f, z_f, indexing = 'ij')
    xx, zz_p = np.meshgrid(x_f, z_c, indexing = 'ij')   # radial faces
    xx_p, zz = np.meshgrid(x_c, z_f, indexing = 'ij')   # vertical faces
    xin, zin = np.meshgrid(x_c, z_c, indexing = 'ij')   # centres
        
    Nz_t = ((z_c >= ice_t).nonzero()[0][0])   # first cell centre in pond
    Nz_b = ((z_c >= ice_b).nonzero()[0][0])   # first cell centre in ice
    Nz_m = Nz_t - Nz_b
    
    
    TT_d, TT_0, CC_0, HH_0, phi_0, pp_i, pp_0 = map(np.zeros, 7*([Nx, Nz], ))   # variable arrays
    T_L, T_R, H_L, H_R, C_L, C_R, phi_L, phi_R, p_L, p_R = map(np.zeros, 10*(Nz, ))   # boundary arrays
    T_T, T_B, H_T, H_B, C_T, C_B, phi_T, phi_B, p_T, p_B = map(np.zeros, 10*(Nx, ))
    T_Tice, T_Bice = map(np.zeros, 2*(Nx, ))
    FT_L, FT_R, FH_L, FH_R, FC_L, FC_R, Fphi_L, Fphi_R, Fp_L, Fp_R = \
                                                           map(np.zeros, 10*(Nz, ))
    FT_T, FT_B, FH_T, FH_B, FC_T, FC_B, Fphi_T, Fphi_B, Fp_T, Fp_B = \
                                                           map(np.zeros, 10*(Nx, ))
    T_L_d, T_R_d = map(np.zeros, 2*(Nz, ))
    T_T_d, T_B_d = map(np.zeros, 2*(Nx, ))
    a_0 = np.zeros(Nz)
    
    
    # reading int. position
    a_0_all = np.linspace(0.1*AR, 0.7*AR, 7)
    if j == None:
        j = int(input('Enter interface index : '))
    a_0[Nz_b:Nz_t] = a_0_all[j]
    a_0[:Nz_b] = np.nan
    a_0[Nz_t:] = np.nan
    
    TT_init = np.linspace(-1, 0, Nz_m)
            
    # temp. profile at right boundary (ice)
    for i in range(Nz_b, Nz_t):
        T_R[i] = newton(F_z, TT_init[i - Nz_b], dF_z, \
                        args = (z_c[i], ice_b, ice_t, -1, 0, Cr))
            
    
    T_B_d[:] = T_B_sc
    T_T_d[:] = T_T_sc
    
    
    # temperature at bottom of ice
    A_Br = np.array([[AR**3 - a_0[Nz_b]**3, AR**2 - a_0[Nz_b]**2, AR - a_0[Nz_b]], 
                     [3*AR**2, 2*AR, 1], [3*a_0[Nz_b]**2, 2*a_0[Nz_b], 1]])
    b_Br = np.array([T_R_sc, 0, -2*T_0/a_0[Nz_b]])
    c_Br = solve(A_Br, b_Br)
    
    A_Bl = np.array([[-AR**3 + a_0[Nz_b]**3, AR**2 - a_0[Nz_b]**2, -AR + a_0[Nz_b]], 
                     [3*AR**2, -2*AR, 1], [3*a_0[Nz_b]**2, -2*a_0[Nz_b], 1]])
    b_Bl = np.array([T_R_sc, 0, 2*T_0/a_0[Nz_b]])
    c_Bl = solve(A_Bl, b_Bl)

    np.putmask(T_Bice, np.logical_and(xin[:, 0] < a_0[Nz_b], xin[:, 0] > -a_0[Nz_b]), \
               -T_0/a_0[Nz_b]**2*xin[:, 0]**2 + T_0)
    np.putmask(T_Bice, xin[:, 0] >= a_0[Nz_b], \
               c_Br[0]*(xin[:, 0]**3 - a_0[Nz_b]**3) + \
               c_Br[1]*(xin[:,  0]**2 - a_0[Nz_b]**2) + \
               c_Br[2]*(xin[:, 0] - a_0[Nz_b]))
    np.putmask(T_Bice, xin[:, 0] <= -a_0[Nz_b], \
               c_Bl[0]*(xin[xin[:, 0] <= a_0[Nz_b], 0]**3 + a_0[Nz_b]**3) + \
               c_Bl[1]*(xin[xin[:, 0] <= a_0[Nz_b],  0]**2 - a_0[Nz_b]**2) + \
               c_Bl[2]*(xin[xin[:, 0] <= a_0[Nz_b], 0] + a_0[Nz_b]))
    

    # temperature at top of ice
    np.putmask(T_Tice, np.logical_and(xin[:, 0] < a_0[Nz_t-1], xin[:, 0] > 0), \
               T_0/a_0[Nz_t-1]**2*(2/a_0[Nz_t-1]*xin[:, -1]**3 - 3*xin[:, -1]**2) + T_0)
    np.putmask(T_Tice, np.logical_and(xin[:, 0] > -a_0[Nz_t-1], xin[:, 0] <= 0), \
               -T_0/a_0[Nz_t-1]**2*(2/a_0[Nz_t-1]*xin[:, -1]**3 + 3*xin[:, -1]**2) + T_0)
    np.putmask(T_Tice, np.logical_or(xin[:, 0] >= a_0[Nz_t-1], xin[:, 0] <= -a_0[Nz_t-1]), \
               CC_0[:, Nz_t])
    
    for i in range(Nx):
        if T_Tice[i] > 0:
            TT_0[i, Nz_b:Nz_t] = (T_Tice[i] - T_Bice[i])*zin[i, Nz_b:Nz_t] + T_Bice[i]
        if T_Tice[i] <= 0:
            for j in range(Nz_b, Nz_t):
                TT_0[i, j] = newton(F_z, TT_init[j - Nz_b], dF_z, \
                                  args = (z_c[j], ice_b, ice_t, T_Bice[i], T_Tice[i], Cr))
    
    # derivatives at top/bottom of ice (nondimensional)
    dTdz_T, dTdz_B = map(np.zeros, 2*(Nx, ))
    np.putmask(dTdz_T[:], np.logical_and(xin[:, 0] < a_0[Nz_t-1], xin[:, 0] > -a_0[Nz_t-1]), \
               (T_Tice[:] - T_Bice[:]))
    np.putmask(dTdz_T[:], np.logical_or(xin[:, 0] <= -a_0[Nz_t-1], xin[:, 0] >= a_0[Nz_t-1]), \
               dF_zdz(T_Tice[:], ice_t, ice_b, ice_t, T_Bice[:], T_Tice[:], Cr))
    
    
    np.putmask(dTdz_B[:], np.logical_and(xin[:, 0] < a_0[Nz_b], xin[:, 0] > -a_0[Nz_b]), \
               (T_Tice[:] - T_Bice[:]))
    np.putmask(dTdz_B[:], np.logical_or(xin[:, 0] <= -a_0[Nz_b], xin[:, 0] >= a_0[Nz_b]), \
               dF_zdz(T_Bice[:], ice_b, ice_b, ice_t, T_Bice[:], T_Tice[:], Cr))
        
    # derivatives at top/bottom of ice at boundaries
    dTdz_T_B = dF_zdz(0, ice_t, ice_b, ice_t, -1, 0, Cr)
    dTdz_B_B = dF_zdz(-1, ice_b, ice_b, ice_t, -1, 0, Cr)
    
    for i in range(Nx):
    # temperature in pond
        #=================== quadratic ======================
        A_T = np.array([[z_f[-1]**2, z_f[-1], 1],
                        [ice_t**2, ice_t, 1],
                        [2*ice_t, 1, 0]])
        b_T = np.array([(T_T_sc - H_f_d)/del_T, T_Tice[i], dTdz_T[i]])
        c_T = solve(A_T, b_T)
    
        # temp. in pond
        TT_0[i, Nz_t:] = (c_T[0]*z_c[Nz_t:]**2 + c_T[1]*z_c[Nz_t:] + c_T[2])
    
    # temperature in ocean
        #=================== quadratic ======================
        A_B = np.array([[ice_b**2, ice_b, 1], 
                        [z_f[0]**2, z_f[0], 1],
                        [2*ice_b, 1, 0]])
        b_B = np.array([T_Bice[i], (T_B_sc - H_f_d)/del_T, dTdz_B[i]])
        c_B = solve(A_B, b_B)
        
        # temp. in ocean
        TT_0[i, :Nz_b] = (c_B[0]*z_c[:Nz_b]**2 + c_B[1]*z_c[:Nz_b] + c_B[2])
        
    # boundary temperatures in pond/ocean
    # temperature in pond
    A_TB = np.array([[z_f[-1]**2, z_f[-1], 1],
                    [ice_t**2, ice_t, 1],
                    [2*ice_t, 1, 0]])
    b_TB = np.array([(T_T_sc - H_f_d)/del_T, 0, dTdz_T_B])
    c_TB = solve(A_TB, b_TB)
    
    # temp. in pond
    T_R[Nz_t:] = (c_TB[0]*z_c[Nz_t:]**2 + c_TB[1]*z_c[Nz_t:] + c_TB[2])
    
    # temperature in ocean
    #=================== quadratic ======================
    A_BB = np.array([[ice_b**2, ice_b, 1], 
                    [z_f[0]**2, z_f[0], 1],
                    [2*ice_b, 1, 0]])
    b_BB = np.array([-1, (T_B_sc - H_f_d)/del_T, dTdz_B_B])
    c_BB = solve(A_BB, b_BB)
    
    # temp. in ocean
    T_R[:Nz_b] = (c_BB[0]*z_c[:Nz_b]**2 + c_BB[1]*z_c[:Nz_b] + c_BB[2])
    T_L[:] = T_R
    
    #TT_0[:, :Nz_b] = (T_B_sc - H_f_d)/del_T
    z_ref = 0.15
    
    # defining fields
    CC_0[:, Nz_b:] = 0   # salinity field
    CC_0[:, :Nz_b] = C_O
    HH_0[:, :] = Ent_calc(TT_0, CC_0, St, Cr)   # enthalpy
    phi_0[:, :] = SFrac_calc(HH_0, CC_0, St, Cr)   # solid fraction
    
    # interface
    a_0n, a_0p = int_loc(HH_0, CC_0, xx, zz, xx_p, zz_p)
    np.putmask(a_0n[0], ~np.isnan(a_0n[0]), -a_0)
    np.putmask(a_0p[0], ~np.isnan(a_0p[0]), a_0)
    
    # boundary temperatures
    T_L[:] = T_L
    T_R[:] = T_R
    T_B[:] = (T_B_d - H_f_d)/del_T
    T_T[:] = (T_T_d - H_f_d)/del_T
    # boundary enthalpies
    H_L[:] = Ent_calc(T_L, CC_0[0, :], St, Cr)
    H_R[:] = Ent_calc(T_R, CC_0[-1, :], St, Cr)
    H_B[:] = Ent_calc(T_B, CC_0[:, 0], St, Cr)
    H_T[:] = Ent_calc(T_T, CC_0[:, -1], St, Cr)
    # boundary salinities
    C_L[:] = CC_0[0, :]
    C_R[:] = CC_0[-1, :]
    C_B[:] = CC_0[:, 0]
    C_T[:] = CC_0[:, -1]
    # boundary solid fractions
    phi_L[:] = SFrac_calc(H_L, CC_0[0, :], St, Cr)
    phi_R[:] = SFrac_calc(H_R, CC_0[-1, :], St, Cr)
    phi_B[:] = SFrac_calc(H_B, CC_0[:, 0], St, Cr)
    phi_T[:] = SFrac_calc(H_T, CC_0[:, -1], St, Cr)
    # boundary pressure
    p_L[:] = 0
    p_R[:] = 0
    p_B[:] = 0
    p_T[:] = 1
    
    # packing boundary conditions
    # Dirichlet
    HH_BC = [H_L, H_R, H_B, H_T]
    TT_BC = [T_L, T_R, T_B, T_T]
    CC_BC = [C_L, C_R, C_B, C_T]
    phi_BC = [phi_L, phi_R, phi_B, phi_T]
    pp_BC = [p_L, p_R, p_B, p_T]
    
    # Neumann
    FF = [FH_L, FH_R, FH_B, FH_T]
    FF_TT = [FT_L, FT_R, FT_B, FT_T]
    FF_CC = [FC_L, FC_R, FC_B, FC_T]
    FF_phi = [Fphi_L, Fphi_R, Fphi_B, Fphi_T]
    FF_pp = [Fp_L, Fp_R, Fp_B, Fp_T]
    
    
    # reading boundary conditions
    if BC_T == None and BC_p == None:
        BC_T = int(input('Enter BC types for T, H, C, phi (L, R, B, T) - '+\
                      '0 = (N, N, N, D), 1 = (D, D, N, D) (default 0) : ') or 0)
        BC_p = int(input('Enter BC types for p (L, R, B, T) - '+\
                      '0 = (N, N, D, D), 1 = (N, N, D, N) (default 0) : ') or 0)
        
    if BC_T == 0:
        alp_T = [0, 0, 0, 1]
    elif BC_T == 1:
        alp_T = [1, 1, 0, 1]

        
    if BC_p == 0:
        alp_p = [0, 0, 1, 1]
    elif BC_p == 1:
        alp_p = [0, 0, 1, 0]
    
    alp = [alp_T, alp_p]
    
    # calculating pressure
    G = np.zeros([Nx, Nz])
    # permeability
    Pi, Pi_bh, Pi_bv = Perm_calc(phi_0, xx, xin, zz, zin, phi_BC, FF_phi, alp_T, Da_h)
    # velocities
    pp_0, u_b, w_b = p_solve(xx, xin, zz, zin, Pi_bh, Pi_bv, pp_BC, FF_pp, alp_p, 
                             Pe, G = G)
    
    # Numerical parameters
    xi = 0.3
    dx, dz = xx[1, 0] - xx[0, 0], zz[0, 1] - zz[0, 0]
    dt_max = xi/(np.max(abs(u_b))/dx + np.max(abs(w_b))/dz)
    dt_max = 1e-4*dt_max
    dt_0 = dt_max   # time step
    D_n = npar.omeg*dx**2   # numerical diffusivity
    
    # Initialising timestep routine
    t_init = 0
    if t_run == None:
        t_run = float(input('Enter simulation runtime (default 0.05) : ') or 0.05)
    t_max = t_init + t_run      #100*dt_0
    tt = np.array([t_init, t_init + dt_0])
    N_shots = max(int(5000*t_run), 10)
    tt_shots = np.linspace(t_init, t_max, N_shots+1)    
    
    # opening files for saving
    # fields
    xx_file = open('arrays/xx.csv', 'w+')
    xin_file = open('arrays/xin.csv', 'w+')
    xx_p_file = open('arrays/xx_p.csv', 'w+')
    zz_file = open('arrays/zz.csv', 'w+')
    zin_file = open('arrays/zin.csv', 'w+')
    zz_p_file = open('arrays/zz_p.csv', 'w+')
    TT_file = open('arrays/TT_0.csv', 'w+')
    HH_file = open('arrays/HH_0.csv', 'w+')
    CC_file = open('arrays/CC_0.csv', 'w+')
    phi_file = open('arrays/phi_0.csv', 'w+')
    pp_file = open('arrays/pp_0.csv', 'w+')
    a_n_file = open('arrays/a_n.csv', 'w+')
    a_p_file = open('arrays/a_p.csv', 'w+')
    tt_file = open('arrays/tt.csv', 'w+')
    tt_shots_file = open('arrays/tt_shots.csv', 'w+')
    N_file = open('arrays/N.csv', 'w+')
    
    # boundaries
    # settings
    alp_T_file = open('arrays/alp_T.csv', 'w+')
    alp_p_file = open('arrays/alp_p.csv', 'w+')
    
    # Dirichlet
    T_L_file = open('arrays/T_L.csv', 'w+')
    T_R_file = open('arrays/T_R.csv', 'w+')
    T_B_file = open('arrays/T_B.csv', 'w+')
    T_T_file = open('arrays/T_T.csv', 'w+')
    H_L_file = open('arrays/H_L.csv', 'w+')
    H_R_file = open('arrays/H_R.csv', 'w+')
    H_B_file = open('arrays/H_B.csv', 'w+')
    H_T_file = open('arrays/H_T.csv', 'w+')
    C_L_file = open('arrays/C_L.csv', 'w+')
    C_R_file = open('arrays/C_R.csv', 'w+')
    C_B_file = open('arrays/C_B.csv', 'w+')
    C_T_file = open('arrays/C_T.csv', 'w+')
    phi_L_file = open('arrays/phi_L.csv', 'w+')
    phi_R_file = open('arrays/phi_R.csv', 'w+')
    phi_B_file = open('arrays/phi_B.csv', 'w+')
    phi_T_file = open('arrays/phi_T.csv', 'w+')
    p_L_file = open('arrays/p_L.csv', 'w+')
    p_R_file = open('arrays/p_R.csv', 'w+')
    p_B_file = open('arrays/p_B.csv', 'w+')
    p_T_file = open('arrays/p_T.csv', 'w+')
    
    # Neumann
    FT_L_file = open('arrays/FT_L.csv', 'w+')
    FT_R_file = open('arrays/FT_R.csv', 'w+')
    FT_B_file = open('arrays/FT_B.csv', 'w+')
    FT_T_file = open('arrays/FT_T.csv', 'w+')
    FH_L_file = open('arrays/FH_L.csv', 'w+')
    FH_R_file = open('arrays/FH_R.csv', 'w+')
    FH_B_file = open('arrays/FH_B.csv', 'w+')
    FH_T_file = open('arrays/FH_T.csv', 'w+')
    FC_L_file = open('arrays/FC_L.csv', 'w+')
    FC_R_file = open('arrays/FC_R.csv', 'w+')
    FC_B_file = open('arrays/FC_B.csv', 'w+')
    FC_T_file = open('arrays/FC_T.csv', 'w+')
    Fphi_L_file = open('arrays/Fphi_L.csv', 'w+')
    Fphi_R_file = open('arrays/Fphi_R.csv', 'w+')
    Fphi_B_file = open('arrays/Fphi_B.csv', 'w+')
    Fphi_T_file = open('arrays/Fphi_T.csv', 'w+')
    Fp_L_file = open('arrays/Fp_L.csv', 'w+')
    Fp_R_file = open('arrays/Fp_R.csv', 'w+')
    Fp_B_file = open('arrays/Fp_B.csv', 'w+')
    Fp_T_file = open('arrays/Fp_T.csv', 'w+')
    
    # scalars
    IC_file = open('arrays/IC.asc', 'w+')
    dt_file = open('arrays/dt_max.asc', 'w+')
    St_file = open('arrays/St.asc', 'w+')
    Cr_file = open('arrays/Cr.asc', 'w+')
    Pe_file = open('arrays/Pe.asc', 'w+')
    Da_h_file = open('arrays/Da_h.asc', 'w+')
    AR_file = open('arrays/AR.asc', 'w+')
    D_n_file = open('arrays/D_n.asc', 'w+')
    
    # strings
    run_file = open('arrays/run.asc', 'w+')
    
    # saving
    # fields
    np.savetxt(xx_file, xx, delimiter = ',')
    np.savetxt(xin_file, xin, delimiter = ',')
    np.savetxt(xx_p_file, xx_p, delimiter = ',')
    np.savetxt(zz_file, zz, delimiter = ',')
    np.savetxt(zin_file, zin, delimiter = ',')
    np.savetxt(zz_p_file, zz_p, delimiter = ',')
    np.savetxt(TT_file, TT_0, delimiter = ',')
    np.savetxt(HH_file, HH_0, delimiter = ',')
    np.savetxt(CC_file, CC_0, delimiter = ',')
    np.savetxt(phi_file, phi_0, delimiter = ',')
    np.savetxt(pp_file, pp_0, delimiter = ',')
    np.savetxt(a_n_file, a_0n, delimiter = ',')
    np.savetxt(a_p_file, a_0p, delimiter = ',')
    np.savetxt(tt_file, tt, delimiter = ',')
    np.savetxt(tt_shots_file, tt_shots, delimiter = ',')
    np.savetxt(N_file, N, delimiter = ',')
    
    # boundary conditions
    # settings
    np.savetxt(alp_T_file, alp_T, delimiter = ',')
    np.savetxt(alp_p_file, alp_p, delimiter = ',')
    
    # Dirichlet
    np.savetxt(T_L_file, T_L, delimiter = ',')
    np.savetxt(T_R_file, T_R, delimiter = ',')
    np.savetxt(T_B_file, T_B, delimiter = ',')
    np.savetxt(T_T_file, T_T, delimiter = ',')
    np.savetxt(H_L_file, H_L, delimiter = ',')
    np.savetxt(H_R_file, H_R, delimiter = ',')
    np.savetxt(H_B_file, H_B, delimiter = ',')
    np.savetxt(H_T_file, H_T, delimiter = ',')
    np.savetxt(C_L_file, C_L, delimiter = ',')
    np.savetxt(C_R_file, C_R, delimiter = ',')
    np.savetxt(C_B_file, C_B, delimiter = ',')
    np.savetxt(C_T_file, C_T, delimiter = ',')
    np.savetxt(phi_L_file, phi_L, delimiter = ',')
    np.savetxt(phi_R_file, phi_R, delimiter = ',')
    np.savetxt(phi_B_file, phi_B, delimiter = ',')
    np.savetxt(phi_T_file, phi_T, delimiter = ',')
    np.savetxt(p_L_file, p_L, delimiter = ',')
    np.savetxt(p_R_file, p_R, delimiter = ',')
    np.savetxt(p_B_file, p_B, delimiter = ',')
    np.savetxt(p_T_file, p_T, delimiter = ',')
    
    # Neumann
    np.savetxt(FT_L_file, FT_L, delimiter = ',')
    np.savetxt(FT_R_file, FT_R, delimiter = ',')
    np.savetxt(FT_B_file, FT_B, delimiter = ',')
    np.savetxt(FT_T_file, FT_T, delimiter = ',')
    np.savetxt(FH_L_file, FH_L, delimiter = ',')
    np.savetxt(FH_R_file, FH_R, delimiter = ',')
    np.savetxt(FH_B_file, FH_B, delimiter = ',')
    np.savetxt(FH_T_file, FH_T, delimiter = ',')
    np.savetxt(FC_L_file, FC_L, delimiter = ',')
    np.savetxt(FC_R_file, FC_R, delimiter = ',')
    np.savetxt(FC_B_file, FC_B, delimiter = ',')
    np.savetxt(FC_T_file, FC_T, delimiter = ',')
    np.savetxt(Fphi_L_file, Fphi_L, delimiter = ',')
    np.savetxt(Fphi_R_file, Fphi_R, delimiter = ',')
    np.savetxt(Fphi_B_file, Fphi_B, delimiter = ',')
    np.savetxt(Fphi_T_file, Fphi_T, delimiter = ',')
    np.savetxt(Fp_L_file, Fp_L, delimiter = ',')
    np.savetxt(Fp_R_file, Fp_R, delimiter = ',')
    np.savetxt(Fp_B_file, Fp_B, delimiter = ',')
    np.savetxt(Fp_T_file, Fp_T, delimiter = ',')
    
    # scalars
    print(IC, file = IC_file)
    print(dt_max, file = dt_file)
    print(St, file = St_file)
    print(Cr, file = Cr_file)
    print(Pe, file = Pe_file)
    print(Da_h, file = Da_h_file)
    print(AR, file = AR_file)
    print(D_n, file = D_n_file)
    
    # strings
    print(run, file = run_file)
    
    # closing
    # fields
    xx_file.close()
    xin_file.close()
    xx_p_file.close()
    zz_file.close()
    zin_file.close()
    zz_p_file.close()
    TT_file.close()
    HH_file.close()
    CC_file.close()
    phi_file.close()
    pp_file.close()
    a_n_file.close()
    a_p_file.close()
    tt_file.close()
    tt_shots_file.close()
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
    
    # scalars
    IC_file.close()
    dt_file.close()
    St_file.close()
    Cr_file.close()
    Pe_file.close()
    Da_h_file.close()
    AR_file.close()
    D_n_file.close()
    
    # strings
    run_file.close()

    return