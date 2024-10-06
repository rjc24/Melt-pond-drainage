# Number of cores to use
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


import numpy as np
from scipy.linalg import solve
import Enthalpy_functions as ef
import phys_params as ppar
import num_params as npar

# example initial conditions

def init(run_enter = None, St = None, Cr = None, Re = None, T_p = None, 
         C_p = None, lam = None, j = None, Nx = None, t_run = None, BC = None):

    # Setting file location
    if run_enter == None:
        run_enter = input('Directory for storing data : ')
    run = run_enter+'/'    

    # Dimensionless parameters
    T_L, T_R = 0.1, -1   # Channel (pond) temp
    if St == None:
        St = float(input('Enter Stefan number (default = L/(c_p*del_T)) : ') or ppar.St)
    if Cr == None:
        Cr = float(input('Enter concentration ratio (default -H_f/del_T) : ') or ppar.Cr)
    if Re == None:
        Re = float(input('Enter Reynolds number (default 10) : ') or 10)
    if T_p == None:
        T_p = float(input('Enter pond temperature (default 0.1) : ') or 0.1)
    if C_p == None:
        C_p = float(input('Enter pond salinity (default Cr) : ') or Cr)
    if lam == None:
        lam = float(input('Enter length ratio (default 0.5) : ') or 0.5)
    Pr = ppar.Pr


    a_0_all = np.linspace(0, 1, 22)
    if j == None:
        j = int(input('Enter interface index : '))
    a_0 = a_0_all[j]
        
    # calculating lam->turb transitional radius
    a_t = ef.a_star(0.5, Re, lam)

    # numerical parameters
    if Nx == None:
        Nx = int(input('Enter mesh size (number of cells) : '))
    
    HH_0, TT_0, CC_0, phi_0, dx = \
        map(np.zeros, 5*(Nx,))   # cell centres, initial H, T & phi
    xx = np.linspace(0, 1, Nx+1)
    xin = (xx[:-1] + xx[1:])/2
    
    # Temperature profile
    A_Br = np.array([[1 - a_0**3, 1 - a_0**2, 1 - a_0], 
                     [3, 2, 1], [3*a_0**2, 2*a_0, 1]])
    b_Br = np.array([T_R, 0, -2*T_L/a_0])
    c_Br = solve(A_Br, b_Br)
    
    np.putmask(TT_0, xin[:] < a_0, -T_L/a_0**2*xin[:]**2 + T_L)
    np.putmask(TT_0, xin[:] >= a_0, \
               c_Br[0]*(xin[:]**3 - a_0**3) + c_Br[1]*(xin[:]**2 - a_0**2) + \
               c_Br[2]*(xin[:] - a_0))
    
        
    HH_0[:] = ef.Ent_calc(TT_0, CC_0, St, Cr)   # enthalpy
    phi_0[:] = ef.SFrac_calc(HH_0, CC_0, St, Cr)   # solid fraction
    
    # Numerical parameters
    alpha = 0.1   # numerical factor (Need < 0.5)
    dx[:] = xx[1:] - xx[:-1]   # spatial step
    dt_0 = alpha*np.min(dx)**2.5   # time step
    dt_max = alpha*np.min(dx)   # max. time step
    D_n = npar.omeg*np.max(dx)**2
    
    # Initialising timestep routine
    t_init = 0
    if t_run == None:
        t_run = float(input('Enter simulation runtime (default 1.0) : ') or 1.0)
    t_max = t_init + t_run      #100*dt_0
    tt = np.array([t_init, t_init + dt_0])
    N_shots = max(int(500*t_run), 10)
    tt_shots = np.linspace(t_init, t_max, N_shots)
    a = np.zeros(1)   # interface location
    a[0] = a_0
    
    # Boundary conditions
    # salinities
    C_L = CC_0[0]
    C_R = CC_0[-1]
    # enthalpies
    H_L = ef.Enthalpy_calc(T_L, C_L, St, Cr)
    H_R = ef.Enthalpy_calc(T_R, C_R, St, Cr)
    # boundary solid fractions
    phi_L = ef.SFrac_calc_scal(H_L, C_L, St, Cr)
    phi_R = ef.SFrac_calc_scal(H_R, C_R, St, Cr)
    
    FH_L, FH_R = 0, 0
    FT_L, FT_R = 0, 0
    FC_L, FC_R = 0, 0
    Fphi_L, Fphi_R = 0, 0
    
    # packing boundary conditions
    # Dirichlet
    HH_BC = [H_L, H_R]
    TT_BC = [T_L, T_R]
    CC_BC = [C_L, C_R]
    phi_BC = [phi_L, phi_R]
    
    # Neumann
    FF = [FH_L, FH_R]
    FF_TT = [FT_L, FT_R]
    FF_CC = [FC_L, FC_R]
    FF_phi = [Fphi_L, Fphi_R]

    # reading boundary conditions
    if BC == None:
        BC = int(input('Enter BC types for T (L, R) - '+\
                      '0 - (N, N), 1 - (N, D) (default 0) : ') or 0)
        
    if BC == 0:
        alp = [0, 0]
    elif BC == 1:
        alp = [0, 1]
    
    
    # strings
    run_file = open('arrays/run.asc', 'w+')
    
    # fields
    xx_file = open('arrays/xx.csv', 'w+')
    xin_file = open('arrays/xin.csv', 'w+')
    TT_file = open('arrays/TT_0.csv', 'w+')
    HH_file = open('arrays/HH_0.csv', 'w+')
    CC_file = open('arrays/CC_0.csv', 'w+')
    phi_file = open('arrays/phi_0.csv', 'w+')
    tt_file = open('arrays/tt.csv', 'w+')
    tt_shots_file = open('arrays/tt_shots.csv', 'w+')
    
    # scalars
    a_file =  open('arrays/a_0.asc', 'w+')
    a_t_file = open('arrays/a_t.asc', 'w+')
    St_file = open('arrays/St.asc', 'w+')
    Cr_file = open('arrays/Cr.asc', 'w+')
    Re_file = open('arrays/Re.asc', 'w+')
    Pr_file = open('arrays/Pr.asc', 'w+')
    T_p_file = open('arrays/T_p.asc', 'w+')
    C_p_file = open('arrays/C_p.asc', 'w+')
    lam_file = open('arrays/lam.asc', 'w+')
    dt_file = open('arrays/dt_max.asc', 'w+')
    Nx_file = open('arrays/Nx.asc', 'w+')
    D_n_file = open('arrays/D_n.asc', 'w+')
    
    # boundaries
    # settings
    alp_file = open('arrays/alp.csv', 'w+')
    
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
    # strings
    print(run, file = run_file)
    
    # fields
    np.savetxt(xx_file, xx, delimiter = ',')
    np.savetxt(xin_file, xin, delimiter = ',')
    np.savetxt(TT_file, TT_0, delimiter = ',')
    np.savetxt(HH_file, HH_0, delimiter = ',')
    np.savetxt(CC_file, CC_0, delimiter = ',')
    np.savetxt(phi_file, phi_0, delimiter = ',')
    np.savetxt(tt_file, tt, delimiter = ',')
    np.savetxt(tt_shots_file, tt_shots, delimiter = ',')
    
    # scalars
    print(a_0, file = a_file)
    print(a_t, file = a_t_file)
    print(St, file = St_file)
    print(Cr, file = Cr_file)
    print(Re, file = Re_file)
    print(Pr, file = Pr_file)
    print(T_p, file = T_p_file)
    print(C_p, file = C_p_file)
    print(lam, file = lam_file)
    print(dt_max, file = dt_file)
    print(Nx, file = Nx_file)
    print(D_n, file = D_n_file)
    
    # boundary conditions
    # settings
    np.savetxt(alp_file, alp, delimiter = ',')
    
    # Dirichlet
    print(T_L, file = T_L_file)
    print(T_R, file = T_R_file)
    print(H_L, file = H_L_file)
    print(H_R, file = H_R_file)
    print(C_L, file = C_L_file)
    print(C_R, file = C_R_file)
    print(phi_L, file = phi_L_file)
    print(phi_R, file = phi_R_file)
    
    # Neumann
    print(FT_L, file = FT_L_file)
    print(FT_R, file = FT_R_file)
    print(FH_L, file = FH_L_file)
    print(FH_R, file = FH_R_file)
    print(FC_L, file = FC_L_file)
    print(FC_R, file = FC_R_file)
    print(Fphi_L, file = Fphi_L_file)
    print(Fphi_R, file = Fphi_R_file)
    
    # closing
    # fields
    run_file.close()
    
    # scalars
    a_file.close()
    a_t_file.close()
    dt_file.close()
    St_file.close()
    Cr_file.close()
    Re_file.close()
    Pr_file.close()
    T_p_file.close()
    C_p_file.close()
    lam_file.close()
    Nx_file.close()
    D_n_file.close()
    
    # fields
    xx_file.close()
    xin_file.close()
    TT_file.close()
    HH_file.close()
    CC_file.close()
    phi_file.close()
    tt_file.close()
    tt_shots_file.close()
    
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
    
    return
