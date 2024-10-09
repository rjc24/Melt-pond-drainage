# Number of cores to use
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from importlib import import_module
import phys_params as ppar
import num_params as npar
from Data_call import data_call
from re import findall
#from timeit import default_timer as timer

#========================================================================
# Reads initial condition files, dimensionless params, sets off solution 
# procedure
#========================================================================

# Setting file location
run_name = input('Directory for storing data : ')

#start = timer()

# allowing for conitnued run
cont_str = input('Continue previous simulation from stopping point - Y or N ? (default N) ') or 'N'
if cont_str in ['N', 'n', 'No', 'no']:
    cont = False
elif cont_str in ['Y', 'y', 'Yes', 'yes']:
    cont = True
else:
    raise SyntaxError('invalid input')

# reading in initial conditions
if cont == True:
    inits = 'init_c'
else:
    inits = input('Enter name of file containing initial conditions : ')
#    inits = 'init_1'
ic = import_module(inits)

St = float(input('Enter Stefan number (default = L/(c_p*del_T)) : ') or ppar.St)
Cr = float(input('Enter concentration ratio (default 0.05) : ') or 0.05)
Q_T = float(input('Enter channel heating term (default 25) : ') or 25)
j = int(input('Enter interface index : '))
Nx = int(input('Enter mesh size (number of cells) : '))
t_run = float(input('Enter simulation runtime (default 1.0) : ') or 1.0)
BC = int(input('Enter BC types for T (L, R) - '+\
              '0 - (N, N), 1 - (N, D) (default 0) : ') or 0)


ic.init(run_enter = run_name, St = St, Cr = Cr, Q_T = Q_T,
        j = j, Nx = Nx, t_run = t_run, BC = BC)

# strings
run_file = open('arrays/run.asc')

# scalars
a_file = open('arrays/a_0.asc', 'r')
dt_file = open('arrays/dt_max.asc', 'r')
St_file = open('arrays/St.asc', 'r')
Cr_file = open('arrays/Cr.asc', 'r')
Q_T_file = open('arrays/Q_T.asc', 'r')
Nx_file = open('arrays/Nx.asc', 'r')

# arrays
xx_file = open('arrays/xx.csv', 'r')
xin_file = open('arrays/xin.csv', 'r')
TT_file = open('arrays/TT_0.csv', 'r')
HH_file = open('arrays/HH_0.csv', 'r')
CC_file = open('arrays/CC_0.csv', 'r')
phi_file = open('arrays/phi_0.csv', 'r')
tt_file = open('arrays/tt.csv', 'r')
tt_shots_file = open('arrays/tt_shots.csv', 'r')

# boundaries
# settings
alp_file = open('arrays/alp.csv', 'r')

# Dirichlet
T_L_file = open('arrays/T_L.asc', 'r')
T_R_file = open('arrays/T_R.asc', 'r')
H_L_file = open('arrays/H_L.asc', 'r')
H_R_file = open('arrays/H_R.asc', 'r')
C_L_file = open('arrays/C_L.asc', 'r')
C_R_file = open('arrays/C_R.asc', 'r')
phi_L_file = open('arrays/phi_L.asc', 'r')
phi_R_file = open('arrays/phi_R.asc', 'r')

# Neumann
FT_L_file = open('arrays/FT_L.asc', 'r')
FT_R_file = open('arrays/FT_R.asc', 'r')
FH_L_file = open('arrays/FH_L.asc', 'r')
FH_R_file = open('arrays/FH_R.asc', 'r')
FC_L_file = open('arrays/FC_L.asc', 'r')
FC_R_file = open('arrays/FC_R.asc', 'r')
Fphi_L_file = open('arrays/Fphi_L.asc', 'r')
Fphi_R_file = open('arrays/Fphi_R.asc', 'r')


# loading
# strings
run = run_file.readline().replace('\n','')

# fields
xx = np.loadtxt(xx_file, delimiter = ',')
xin = np.loadtxt(xin_file, delimiter = ',')
TT_0 = np.loadtxt(TT_file, delimiter = ',')
HH_0 = np.loadtxt(HH_file, delimiter = ',')
CC_0 = np.loadtxt(CC_file, delimiter = ',')
phi_0 = np.loadtxt(phi_file, delimiter = ',')
tt_0 = np.loadtxt(tt_file, delimiter = ',')
tt_shots = np.loadtxt(tt_shots_file, delimiter = ',')

# scalars
a_0 = float(np.loadtxt(a_file))
St = float(np.loadtxt(St_file))
Cr = float(np.loadtxt(Cr_file))
Q_T = float(np.loadtxt(Q_T_file))
dt_max = float(np.loadtxt(dt_file))
Nx = int(np.loadtxt(Nx_file))

# boundaries
# settings
alp = np.loadtxt(alp_file, delimiter = ',')

# Dirichlet
T_L = float(np.loadtxt(T_L_file))
T_R = float(np.loadtxt(T_R_file))
H_L = float(np.loadtxt(H_L_file))
H_R = float(np.loadtxt(H_R_file))
C_L = float(np.loadtxt(C_L_file))
C_R = float(np.loadtxt(C_R_file))
phi_L = float(np.loadtxt(phi_L_file))
phi_R = float(np.loadtxt(phi_R_file))

# Neumann
FT_L = float(np.loadtxt(FT_L_file))
FT_R = float(np.loadtxt(FT_R_file))
FH_L = float(np.loadtxt(FH_L_file))
FH_R = float(np.loadtxt(FH_R_file))
FC_L = float(np.loadtxt(FC_L_file))
FC_R = float(np.loadtxt(FC_R_file))
Fphi_L = float(np.loadtxt(Fphi_L_file))
Fphi_R = float(np.loadtxt(Fphi_R_file))

# closing
# strings
run_file.close()

# scalars
a_file.close()
dt_file.close()
St_file.close()
Cr_file.close()
Q_T_file.close()
Nx_file.close()

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

params = [St, Cr]

TT_BC = [T_L, T_R]
HH_BC = [H_L, H_R]
CC_BC = [C_L, C_R]
phi_BC = [phi_L, phi_R]

FF_TT = [FT_L, FT_R]
FF_HH = [FH_L, FH_R]
FF_CC = [FC_L, FC_R]
FF_phi = [Fphi_L, Fphi_R]

BC_D = [TT_BC, HH_BC, CC_BC, phi_BC]
BC_N = [FF_TT, FF_HH, FF_CC, FF_phi]

# run number
run_n = int(findall('\d+', run)[0])

# numerical parameters
tol_g = npar.tol_g

shots = True   # save snapshots of solution (True) or every timestep (False)

HH_all, TT_all, CC_all, a_all, tt_all = \
    data_call(run, 'Crank', HH_0, TT_0, CC_0, xx, xin, tt_0, tt_shots, 
              dt_max, a_0, Q_T, params, BC_D, BC_N, alp, 
              omega_s = 1, omega_i = 1, 
              tol_a = 1e-8, tol_r = 1e-8, tol_g = 1e-6, max_iter = 500,
              shots = shots)

print('Saving to ', run)
path = 'data/'+run
np.savetxt(path+'HH(t).csv', HH_all, delimiter = ',')
np.savetxt(path+'TT(t).csv', TT_all, delimiter = ',')
np.savetxt(path+'CC(t).csv', CC_all, delimiter = ',')
np.savetxt(path+'a(t).csv', a_all, delimiter = ',')
np.savetxt(path+'tt.csv', tt_all, delimiter = ',')

#end = timer()

#print('time taken : ', end - start)
