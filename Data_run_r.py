# Number of cores to use
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from importlib import import_module
import phys_params as ppar
from Data_call import data_call
from re import findall
from timeit import default_timer as timer

#========================================================================
# Reads initial condition files, dimensionless params, sets off solution 
# procedure. Used with init_2D_r for rotated channel.
#========================================================================

# Setting file location
run_name = input('Directory for storing data : ')

# allowing for conitnued run
cont = float(input('Continue previous simulation from stopping point'+\
                   ' - 1 (Y) or 0 (N) ? (default 0) : ') or 0)

# reading in initial conditions
if cont == 1:
    inits = 'init_2D_c'
else:
    inits = input('Enter name of file containing initial conditions : ')
    St = float(input('Enter Stefan number (default = L/(c_p*del_T)) : ') or ppar.St)
    Cr = float(input('Enter Concentration ratio (default = -C_f/del_C) : ') or ppar.Cr)
    Pe = float(input('Enter Peclet number (default = 100) : ') or 100)
    Da_h = float(input('Enter Hele-Shaw Darcy number (default = {0:.2f}) : '.format(ppar.Da_h)) or ppar.Da_h)
    T_p = float(input('Enter pond temperature (default = {}) : '.format(ppar.T_p)) or ppar.T_p)
    C_p = float(input('Enter pond salinity (default = {}) : '.format(-0.3)) or -0.3)
    AR = float(input('Enter domain aspect ratio (default 0.2) : ') or 0.2)
    Nz = int(input('Enter vertical mesh size (number of z cells) : '))
    a_0_int = int(input('Enter interface index (default 1) : ') or 1)
    p_an_i = int(input('Enter rotation index (default 0) : ') or 0)
    BC_T = int(input('Enter BC types for T, H, C, phi (L, R, B, T) - '+\
                  '0 = (N, N, N, D), 1 = (D, D, N, D) (default 0) : ') or 0)
    BC_p = int(input('Enter BC types for p (L, R, B, T) - '+\
                  '0 = (N, N, D, D), 1 = (N, N, D, N) (default 0) : ') or 0)

t_run = float(input('Enter simulation runtime (default 0.05) : ') or 0.05)

ic = import_module(inits)   #importing initial conditions

p_an_v = np.arange(0, 7*np.pi/32, np.pi/32)
i_pos = np.array([0, 0.5])
a_len = 1


if cont == 0:
    ic.init(run_enter = run_name, St = St, Cr = Cr, Pe = Pe, Da_h = Da_h, 
             T_p = T_p, C_p = C_p, AR = AR, Nz = Nz, j = a_0_int, 
             a_len = a_len, p_an = p_an_v[p_an_i], i_pos = i_pos, 
             BC_T = BC_T, BC_p = BC_p, t_run = t_run)
else:
    ic.init(run_enter = run_name, t_run = t_run)
    
# strings
run_file = open('arrays/run.asc')

# scalars
dt_file = open('arrays/dt_max.asc', 'r')
St_file = open('arrays/St.asc', 'r')
Cr_file = open('arrays/Cr.asc', 'r')
Pe_file = open('arrays/Pe.asc', 'r')
Da_h_file = open('arrays/Da_h.asc', 'r')
AR_file = open('arrays/AR.asc', 'r')

# arrays
xx_file = open('arrays/xx.csv', 'r')
xin_file = open('arrays/xin.csv', 'r')
xx_p_file = open('arrays/xx_p.csv', 'r')
zz_file = open('arrays/zz.csv', 'r')
zin_file = open('arrays/zin.csv', 'r')
zz_p_file = open('arrays/zz_p.csv', 'r')
TT_file = open('arrays/TT_0.csv', 'r')
HH_file = open('arrays/HH_0.csv', 'r')
CC_file = open('arrays/CC_0.csv', 'r')
phi_file = open('arrays/phi_0.csv', 'r')
pp_file = open('arrays/pp_0.csv', 'r')
a_n_file = open('arrays/a_n.csv', 'r')
a_p_file = open('arrays/a_p.csv', 'r')
tt_file = open('arrays/tt.csv', 'r')
tt_shots_file = open('arrays/tt_shots.csv', 'r')
N_file = open('arrays/N.csv', 'r')
D_n_file = open('arrays/D_n.asc', 'r')

# boundaries
# settings
alp_T_file = open('arrays/alp_T.csv', 'r')
alp_p_file = open('arrays/alp_p.csv', 'r')

# Dirichlet
T_L_file = open('arrays/T_L.csv', 'r')
T_R_file = open('arrays/T_R.csv', 'r')
T_B_file = open('arrays/T_B.csv', 'r')
T_T_file = open('arrays/T_T.csv', 'r')
H_L_file = open('arrays/H_L.csv', 'r')
H_R_file = open('arrays/H_R.csv', 'r')
H_B_file = open('arrays/H_B.csv', 'r')
H_T_file = open('arrays/H_T.csv', 'r')
C_L_file = open('arrays/C_L.csv', 'r')
C_R_file = open('arrays/C_R.csv', 'r')
C_B_file = open('arrays/C_B.csv', 'r')
C_T_file = open('arrays/C_T.csv', 'r')
phi_L_file = open('arrays/phi_L.csv', 'r')
phi_R_file = open('arrays/phi_R.csv', 'r')
phi_B_file = open('arrays/phi_B.csv', 'r')
phi_T_file = open('arrays/phi_T.csv', 'r')
p_L_file = open('arrays/p_L.csv', 'r')
p_R_file = open('arrays/p_R.csv', 'r')
p_B_file = open('arrays/p_B.csv', 'r')
p_T_file = open('arrays/p_T.csv', 'r')

# Neumann
FT_L_file = open('arrays/FT_L.csv', 'r')
FT_R_file = open('arrays/FT_R.csv', 'r')
FT_B_file = open('arrays/FT_B.csv', 'r')
FT_T_file = open('arrays/FT_T.csv', 'r')
FH_L_file = open('arrays/FH_L.csv', 'r')
FH_R_file = open('arrays/FH_R.csv', 'r')
FH_B_file = open('arrays/FH_B.csv', 'r')
FH_T_file = open('arrays/FH_T.csv', 'r')
FC_L_file = open('arrays/FC_L.csv', 'r')
FC_R_file = open('arrays/FC_R.csv', 'r')
FC_B_file = open('arrays/FC_B.csv', 'r')
FC_T_file = open('arrays/FC_T.csv', 'r')
Fphi_L_file = open('arrays/Fphi_L.csv', 'r')
Fphi_R_file = open('arrays/Fphi_R.csv', 'r')
Fphi_B_file = open('arrays/Fphi_B.csv', 'r')
Fphi_T_file = open('arrays/Fphi_T.csv', 'r')
Fp_L_file = open('arrays/Fp_L.csv', 'r')
Fp_R_file = open('arrays/Fp_R.csv', 'r')
Fp_B_file = open('arrays/Fp_B.csv', 'r')
Fp_T_file = open('arrays/Fp_T.csv', 'r')


# loading
# strings
run = run_file.readline().replace('\n','')

# fields
xx = np.loadtxt(xx_file, delimiter = ',')
xin = np.loadtxt(xin_file, delimiter = ',')
xx_p = np.loadtxt(xx_p_file, delimiter = ',')
zz = np.loadtxt(zz_file, delimiter = ',')
zin = np.loadtxt(zin_file, delimiter = ',')
zz_p = np.loadtxt(zz_p_file, delimiter = ',')
TT_0 = np.loadtxt(TT_file, delimiter = ',')
HH_0 = np.loadtxt(HH_file, delimiter = ',')
CC_0 = np.loadtxt(CC_file, delimiter = ',')
phi_0 = np.loadtxt(phi_file, delimiter = ',')
pp_0 = np.loadtxt(pp_file, delimiter = ',')
a_n0 = np.loadtxt(a_n_file, delimiter = ',')
a_p0 = np.loadtxt(a_p_file, delimiter = ',')
tt_0 = np.loadtxt(tt_file, delimiter = ',')
tt_shots = np.loadtxt(tt_shots_file, delimiter = ',')
N = np.loadtxt(N_file, delimiter = ',')

# scalars
dt_max = float(np.loadtxt(dt_file))
St = float(np.loadtxt(St_file))
Cr = float(np.loadtxt(Cr_file))
Pe = float(np.loadtxt(Pe_file))
Da_h = float(np.loadtxt(Da_h_file))
Nx = int(N[0])
Nz = int(N[1])
AR = float(np.loadtxt(AR_file))
D_n = float(np.loadtxt(D_n_file))

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

# closing
# strings
run_file.close()

# scalars
dt_file.close()
St_file.close()
Cr_file.close()
Pe_file.close()
Da_h_file.close()
AR_file.close()
D_n_file.close()

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


start = timer()

# packing parameters, boundary conditions
params = [St, Cr, Pe, Da_h]

alp = [alp_T, alp_p]

TT_BC = [T_L, T_R, T_B, T_T]
HH_BC = [H_L, H_R, H_B, H_T]
CC_BC = [C_L, C_R, C_B, C_T]
phi_BC = [phi_L, phi_R, phi_B, phi_T]
pp_BC = [p_L, p_R, p_B, p_T]

FF_TT = [FT_L, FT_R, FT_B, FT_T]
FF_HH = [FH_L, FH_R, FH_B, FH_T]
FF_CC = [FC_L, FC_R, FC_B, FC_T]
FF_phi = [Fphi_L, Fphi_R, Fphi_B, Fphi_T]
FF_pp = [Fp_L, Fp_R, Fp_B, Fp_T]

BC_D = [TT_BC, HH_BC, CC_BC, phi_BC, pp_BC]
BC_N = [FF_TT, FF_HH, FF_CC, FF_phi, FF_pp]

# run number
run_n = int(findall('\d+', run)[0])

print(run_n)

# time things
t_max = tt_shots[-1]

# interface
a_0 = [a_n0, a_p0]

shots = True   # save snapshots of solution (True) or every timestep (False)


run = 'run_'+str(run_n)+'/'


HH_all, TT_all, CC_all, pp_all, a_n_all, a_p_all, tt_all = \
    data_call(run, 'Crank', HH_0, TT_0, CC_0, pp_0, xx, xin, xx_p, 
              zz, zin, zz_p, tt_0, tt_shots, 
              dt_max, a_0, AR, params, BC_D, BC_N, alp, D_n, 
              gamma = 0, omega_s = 1,
              tol_a = 1e-6, tol_r = 1e-8, max_iter = 10,
              shots = shots, cont = cont)



path = 'data/'+run

np.save(path+'HH(t).npy', HH_all)
np.save(path+'TT(t).npy', TT_all)
np.save(path+'CC(t).npy', CC_all)
np.save(path+'pp(t).npy', pp_all)
np.save(path+'a_n(t).npy', a_n_all)
np.save(path+'a_p(t).npy', a_p_all)
np.save(path+'tt.npy', tt_all)

end = timer()

print('time taken : ', end - start)
