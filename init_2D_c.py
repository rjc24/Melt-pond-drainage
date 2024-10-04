# Number of cores to use
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from Enthalpy_functions import SFrac_calc, Perm_calc
from elliptic_solve_sparse import p_solve
from interface import int_loc

# initial conditions for continuing previous run

def init(run_enter = None, t_run = None):
    
    # IC file
    IC = 1
    
    # Setting file location
    if run_enter == None:
        run_enter = input('Directory for storing data : ')
    run = run_enter+'/'
    
    path = 'data/'+run
    path_cont = 'arrays/cont/'+run
    
        
    # Extracting data from previous run
    # fields
    xx_file = open(path_cont+'xx.csv', 'r')
    xin_file = open(path_cont+'xin.csv', 'r')
    zz_file = open(path_cont+'zz.csv', 'r')
    zin_file = open(path_cont+'zin.csv', 'r')
    N_file = open(path_cont+'N.csv', 'r')
    
    # boundaries
    # settings
    alp_T_file = open(path_cont+'alp_T.csv', 'r')
    alp_p_file = open(path_cont+'alp_p.csv', 'r')
    
    # Dirichlet
    T_L_file = open(path_cont+'T_L.csv', 'r')
    T_R_file = open(path_cont+'T_R.csv', 'r')
    T_B_file = open(path_cont+'T_B.csv', 'r')
    T_T_file = open(path_cont+'T_T.csv', 'r')
    H_L_file = open(path_cont+'H_L.csv', 'r')
    H_R_file = open(path_cont+'H_R.csv', 'r')
    H_B_file = open(path_cont+'H_B.csv', 'r')
    H_T_file = open(path_cont+'H_T.csv', 'r')
    C_L_file = open(path_cont+'C_L.csv', 'r')
    C_R_file = open(path_cont+'C_R.csv', 'r')
    C_B_file = open(path_cont+'C_B.csv', 'r')
    C_T_file = open(path_cont+'C_T.csv', 'r')
    phi_L_file = open(path_cont+'phi_L.csv', 'r')
    phi_R_file = open(path_cont+'phi_R.csv', 'r')
    phi_B_file = open(path_cont+'phi_B.csv', 'r')
    phi_T_file = open(path_cont+'phi_T.csv', 'r')
    p_L_file = open(path_cont+'p_L.csv', 'r')
    p_R_file = open(path_cont+'p_R.csv', 'r')
    p_B_file = open(path_cont+'p_B.csv', 'r')
    p_T_file = open(path_cont+'p_T.csv', 'r')
    
    # Neumann
    FT_L_file = open(path_cont+'FT_L.csv', 'r')
    FT_R_file = open(path_cont+'FT_R.csv', 'r')
    FT_B_file = open(path_cont+'FT_B.csv', 'r')
    FT_T_file = open(path_cont+'FT_T.csv', 'r')
    FH_L_file = open(path_cont+'FH_L.csv', 'r')
    FH_R_file = open(path_cont+'FH_R.csv', 'r')
    FH_B_file = open(path_cont+'FH_B.csv', 'r')
    FH_T_file = open(path_cont+'FH_T.csv', 'r')
    FC_L_file = open(path_cont+'FC_L.csv', 'r')
    FC_R_file = open(path_cont+'FC_R.csv', 'r')
    FC_B_file = open(path_cont+'FC_B.csv', 'r')
    FC_T_file = open(path_cont+'FC_T.csv', 'r')
    Fphi_L_file = open(path_cont+'Fphi_L.csv', 'r')
    Fphi_R_file = open(path_cont+'Fphi_R.csv', 'r')
    Fphi_B_file = open(path_cont+'Fphi_B.csv', 'r')
    Fphi_T_file = open(path_cont+'Fphi_T.csv', 'r')
    Fp_L_file = open(path_cont+'Fp_L.csv', 'r')
    Fp_R_file = open(path_cont+'Fp_R.csv', 'r')
    Fp_B_file = open(path_cont+'Fp_B.csv', 'r')
    Fp_T_file = open(path_cont+'Fp_T.csv', 'r')

    # scalars
    St_file = open(path_cont+'St.asc', 'r')
    Cr_file = open(path_cont+'Cr.asc', 'r')
    Pe_file = open(path_cont+'Pe.asc', 'r')
    Da_h_file = open(path_cont+'Da_h.asc', 'r')
    AR_file = open(path_cont+'AR.asc', 'r')
    D_n_file = open(path_cont+'D_n.asc', 'r')
    
    
    # creating data
    # fields
    xx = np.loadtxt(xx_file, delimiter = ',')
    xin = np.loadtxt(xin_file, delimiter = ',')
    zz = np.loadtxt(zz_file, delimiter = ',')
    zin = np.loadtxt(zin_file, delimiter = ',')
    N = np.loadtxt(N_file, delimiter = ',')
    
    # boundaries
    # settings
    alp_T = np.loadtxt(alp_T_file, delimiter = ',')
    alp_p = np.loadtxt(alp_p_file, delimiter = ',')
    
    # Dirichlet
    T_L = np.loadtxt(T_L_file, delimiter = ',')
    T_R = np.loadtxt(T_R_file, delimiter = ',')
    T_B = np.loadtxt(T_B_file, delimiter = ',')
    T_T = np.loadtxt(T_T_file, delimiter = ',')
    H_L = np.loadtxt(H_L_file, delimiter = ',')
    H_R = np.loadtxt(H_R_file, delimiter = ',')
    H_B = np.loadtxt(H_B_file, delimiter = ',')
    H_T = np.loadtxt(H_T_file, delimiter = ',')
    C_L = np.loadtxt(C_L_file, delimiter = ',')
    C_R = np.loadtxt(C_R_file, delimiter = ',')
    C_B = np.loadtxt(C_B_file, delimiter = ',')
    C_T = np.loadtxt(C_T_file, delimiter = ',')
    phi_L = np.loadtxt(phi_L_file, delimiter = ',')
    phi_R = np.loadtxt(phi_R_file, delimiter = ',')
    phi_B = np.loadtxt(phi_B_file, delimiter = ',')
    phi_T = np.loadtxt(phi_T_file, delimiter = ',')
    p_L = np.loadtxt(p_L_file, delimiter = ',')
    p_R = np.loadtxt(p_R_file, delimiter = ',')
    p_B = np.loadtxt(p_B_file, delimiter = ',')
    p_T = np.loadtxt(p_T_file, delimiter = ',')
    
    # Neumann
    FT_L = np.loadtxt(FT_L_file, delimiter = ',')
    FT_R = np.loadtxt(FT_R_file, delimiter = ',')
    FT_B = np.loadtxt(FT_B_file, delimiter = ',')
    FT_T = np.loadtxt(FT_T_file, delimiter = ',')
    FH_L = np.loadtxt(FH_L_file, delimiter = ',')
    FH_R = np.loadtxt(FH_R_file, delimiter = ',')
    FH_B = np.loadtxt(FH_B_file, delimiter = ',') 
    FH_T = np.loadtxt(FH_T_file, delimiter = ',')
    FC_L = np.loadtxt(FC_L_file, delimiter = ',')
    FC_R = np.loadtxt(FC_R_file, delimiter = ',')
    FC_B = np.loadtxt(FC_B_file, delimiter = ',') 
    FC_T = np.loadtxt(FC_T_file, delimiter = ',')
    Fphi_L = np.loadtxt(Fphi_L_file, delimiter = ',')
    Fphi_R = np.loadtxt(Fphi_R_file, delimiter = ',')
    Fphi_B = np.loadtxt(Fphi_B_file, delimiter = ',')
    Fphi_T = np.loadtxt(Fphi_T_file, delimiter = ',')
    Fp_L = np.loadtxt(Fp_L_file, delimiter = ',')
    Fp_R = np.loadtxt(Fp_R_file, delimiter = ',')
    Fp_B = np.loadtxt(Fp_B_file, delimiter = ',')
    Fp_T = np.loadtxt(Fp_T_file, delimiter = ',')
    
    # scalars
    St = float(np.loadtxt(St_file))
    Cr = float(np.loadtxt(Cr_file))
    Pe = float(np.loadtxt(Pe_file))
    Da_h = float(np.loadtxt(Da_h_file))
    Nx = int(N[0])
    Nz = int(N[1])
    AR = float(np.loadtxt(AR_file))
    D_n = float(np.loadtxt(D_n_file))
    
    # closing
    # fields
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
    
    # scalars
    St_file.close()
    Cr_file.close()
    Pe_file.close()
    Da_h_file.close()
    AR_file.close()
    D_n_file.close()
    
    
    # main fields from previous runs
    HH_all = np.load(path+'HH(t).npy')
    TT_all = np.load(path+'TT(t).npy')
    CC_all = np.load(path+'CC(t).npy')
    pp_all = np.load(path+'pp(t).npy')
    tt_all = np.load(path+'tt.npy')
    Nt = np.shape(HH_all)[2]

    x_f, z_f = xx[:, 0], zz[0, :]   # cell face locations
    x_c, z_c = xin[:, 0], zin[0, :]   # cell centre locations
    
    zz_p = np.meshgrid(x_f, z_c, indexing = 'ij')[1]   # radial faces
    xx_p = np.meshgrid(x_c, z_f, indexing = 'ij')[0]   # vertical faces
    
    # initial data for continued run
    HH_0 = HH_all[:, :, -1]
    TT_0 = TT_all[:, :, -1]
    CC_0 = CC_all[:, :, -1]
    pp_0 = pp_all[:, :, -1]
    phi_0 = SFrac_calc(HH_0, CC_0, St, Cr)
    
    # packing boundary conditions
    # Dirichlet
    phi_BC = [phi_L, phi_R, phi_B, phi_T]
    pp_BC = [p_L, p_R, p_B, p_T]
    
    # Neumann
    FF_phi = [Fphi_L, Fphi_R, Fphi_B, Fphi_T]
    FF_pp = [Fp_L, Fp_R, Fp_B, Fp_T]

    # calculating velocities
    G = np.zeros([Nx, Nz])
    # permeability
    Pi, Pi_bh, Pi_bv = Perm_calc(phi_0, xx, xin, zz, zin, phi_BC, FF_phi, alp_T, Da_h)
    # velocities
    pp_0, u_b, w_b = p_solve(xx, xin, zz, zin, Pi_bh, Pi_bv, pp_BC, FF_pp, alp_p, 
                             Pe, G = G)
    
    # interface locations
    a_0n, a_0p = int_loc(HH_0, CC_0, xx, zz, xx_p, zz_p)
    
    
    # Numerical parameters
    xi = 0.3
    dx, dz = xx[1, 0] - xx[0, 0], zz[0, 1] - zz[0, 0]
    dt_max = xi/(np.max(abs(u_b))/dx + np.max(abs(w_b))/dz)
    dt_max = 1e-4*dt_max
    dt_0 = dt_max   # time step
    
    # Initialising timestep routine
    t_init = tt_all[-1]
    if t_run == None:
        t_run = float(input('Enter simulation runtime (default 0.05) : ') or 0.05)
    t_max = t_init + t_run      #100*dt_0
    tt = np.concatenate([tt_all, [t_init + dt_0]])
    N_shots = int(Nt/(tt_all[-1] - tt_all[0])*t_run)
    tt_shots_n = np.linspace(t_init, t_max, N_shots+1)[1:]
    tt_shots = np.concatenate([tt_all, tt_shots_n])
    
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