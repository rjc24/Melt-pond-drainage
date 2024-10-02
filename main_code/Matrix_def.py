# Number of cores to use
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from scipy.sparse import diags
import phys_params as ppar
import num_params as npar

def A_theta(method, xx, xin, zz, zin, dt, cc_p, k_bh, k_bv, alp, D_n, EQN = 1):
    
    #========================================================================
    # Function which creates the left hand side matrix for the heat / 
    # solute conservation equation
    #
    # Inputs:
    #    method - 'BTCS' (backwards Euler) or 'Crank' (Crank-Nicholson)
    #    xx, xin, zz, zin - spatial arrays
    #    dt - time step
    #    k_bh, k_bv - thermal conductivity / solutal diffusivity
    #    alp - boundary conditions
    #    EQN - heat(1) / solute(2)
    #
    # Outputs:
    #    A - pentadiagonal matrix
    #
    #========================================================================
    
    # dimensionless quantities
    c_p = ppar.c_p
    
    # numerical diffusivity
    if EQN == 1:
        D_n = 0


    # Ensuxing supported method has been chosen
    if not method in ('Crank', 'BTCS', 'FTCS'):
        raise SyntaxError('Must choose either Crank, BTCS or FTCS for method.')
        return

    Nx, Nz = np.shape(xin)[0], np.shape(xin)[1]    # number of grid points
    N = Nz*Nx
    
    # Diagonal entries of matrix A
    main = np.zeros([Nx, Nz])
    left, right = np.zeros([Nx, Nz]), np.zeros([Nx, Nz])
    lower, upper = np.zeros([Nx, Nz-1]), np.zeros([Nx, Nz-1])
    
    # Defining theta 
    if method == 'BTCS':
        theta = 1        
    elif method == 'Crank':
        theta = 0.5
    elif method == 'FTCS':
        theta = 0

    # alp = 1 - Dirichlet, alp = 0 - Neumann
    alp_L, alp_R, alp_B, alp_T = alp[0], alp[1], alp[2], alp[3]

#=============================================================================
#=================================== main body ===============================
#=============================================================================

    main[1:-1, 1:-1] = (xx[2:-1, 1:-1] - xx[1:-2, 1:-1])*\
        (zz[1:-1, 2:-1] - zz[1:-1, 1:-2]) + \
        (((k_bh[2:-1, 1:-1]/(xin[2:, 1:-1] - xin[1:-1, 1:-1]) + \
        k_bh[1:-2, 1:-1]/(xin[1:-1, 1:-1] - xin[:-2, 1:-1]))*\
        (zz[1:-1, 2:-1] - zz[1:-1, 1:-2]) + \
        (k_bv[1:-1, 2:-1]/(zin[1:-1, 2:] - zin[1:-1, 1:-1]) + \
        k_bv[1:-1, 1:-2]/(zin[1:-1, 1:-1] - zin[1:-1, :-2]))*\
        (xx[2:-1, 1:-1] - xx[1:-2, 1:-1]))/cc_p[1:-1, 1:-1] + \
        (D_n/(xin[2:, 1:-1] - xin[1:-1, 1:-1]) + \
        D_n/(xin[1:-1, 1:-1] - xin[:-2, 1:-1]))*\
        (zz[1:-1, 2:-1] - zz[1:-1, 1:-2]) + \
        (D_n/(zin[1:-1, 2:] - zin[1:-1, 1:-1]) + \
        D_n/(zin[1:-1, 1:-1] - zin[1:-1, :-2]))*\
        (xx[2:-1, 1:-1] - xx[1:-2, 1:-1]))*dt*theta

    left[1:-1, :] = -(k_bh[1:-2, :]/cc_p[:-2, :] + D_n)*(zz[1:-1, 1:] - zz[1:-1, :-1])/\
        (xin[1:-1, :] - xin[:-2, :])*dt*theta
    right[1:-1, :] = -(k_bh[2:-1, :]/cc_p[2:, :] + D_n)*(zz[1:-1, 1:] - zz[1:-1, :-1])/\
        (xin[2:, :] - xin[1:-1, :])*dt*theta

    lower[:, :-1] = -(k_bv[:, 1:-2]/cc_p[:, :-2] + D_n)*(xx[1:, 1:-1] - xx[:-1, 1:-1])/\
        (zin[:, 1:-1] - zin[:, :-2])*dt*theta
    upper[:, 1:] = -(k_bv[:, 2:-1]/cc_p[:, 2:] + D_n)*(xx[1:, 1:-1] - xx[:-1, 1:-1])/\
        (zin[:, 2:] - zin[:, 1:-1])*dt*theta


#=============================================================================
#================================== boundaries ===============================
#=============================================================================

    # left
    main[0, 1:-1] = (xx[1, 1:-1] - xx[0, 1:-1])*\
        (zz[0, 2:-1] - zz[0, 1:-2]) + \
        (((k_bh[1, 1:-1]/(xin[1, 1:-1] - xin[0, 1:-1]) + \
        9*k_bh[0, 1:-1]*alp_L/(6*(xin[0, 1:-1] - xx[0, 1:-1])))*\
        (zz[0, 2:-1] - zz[0, 1:-2]) + \
        (k_bv[0, 2:-1]/(zin[0, 2:] - zin[0, 1:-1]) + \
        k_bv[0, 1:-2]/(zin[0, 1:-1] - zin[0, :-2]))*\
        (xx[1, 1:-1] - xx[0, 1:-1]))/cc_p[0, 1:-1] + \
        (D_n/(xin[1, 1:-1] - xin[0, 1:-1]) + \
        9*D_n*alp_L/(6*(xin[0, 1:-1] - xx[0, 1:-1])))*\
        (zz[0, 2:-1] - zz[0, 1:-2]) + \
        (D_n/(zin[0, 2:] - zin[0, 1:-1]) + \
        D_n/(zin[0, 1:-1] - zin[0, :-2]))*\
        (xx[1, 1:-1] - xx[0, 1:-1]))*dt*theta

    right[0, :] = -((k_bh[1, :]/(xin[1, :] - xin[0, :]) + \
        k_bh[0, :]*alp_L/(6*(xin[0, :] - xx[0, :])))*\
        (zz[0, 1:] - zz[0, :-1])/cc_p[1, :] + \
        (D_n/(xin[1, :] - xin[0, :]) + \
        D_n*alp_L/(6*(xin[0, :] - xx[0, :])))*\
        (zz[0, 1:] - zz[0, :-1]))*dt*theta
            
    # right
    main[-1, 1:-1] = (xx[-1, 1:-1] - xx[-2, 1:-1])*\
        (zz[-1, 2:-1] - zz[-1, 1:-2]) + \
        (((9*k_bh[-1, 1:-1]*alp_R/(6*(xx[-1, 1:-1] - xin[-1, 1:-1])) + \
        k_bh[-2, 1:-1]/(xin[-1, 1:-1] - xin[-2, 1:-1]))* \
        (zz[-1, 2:-1] - zz[-1, 1:-2]) + \
        (k_bv[-1, 2:-1]/(zin[-1, 2:] - zin[-1, 1:-1]) + \
        k_bv[-1, 1:-2]/(zin[-1, 1:-1] - zin[-1, :-2]))*\
        (xx[-1, 1:-1] - xx[-2, 1:-1]))/cc_p[-1, 1:-1] + \
        (9*D_n*alp_R/(6*(xx[-1, 1:-1] - xin[-1, 1:-1])) + \
        D_n/(xin[-1, 1:-1] - xin[-2, 1:-1]))* \
        (zz[-1, 2:-1] - zz[-1, 1:-2]) + \
        (D_n/(zin[-1, 2:] - zin[-1, 1:-1]) + \
        D_n/(zin[-1, 1:-1] - zin[-1, :-2]))*\
        (xx[-1, 1:-1] - xx[-2, 1:-1]))*dt*theta
        
    left[-1, :] = -((k_bh[-2, :]/(xin[-1, :] - xin[-2, :]) + \
        k_bh[-1, :]*alp_R/(6*(xx[-1, :] - xin[-1, :])))*\
        (zz[-1, 1:] - zz[-1, :-1])/cc_p[-2, :] + \
        (D_n/(xin[-1, :] - xin[-2, :]) + \
        D_n*alp_R/(6*(xx[-1, :] - xin[-1, :])))*\
        (zz[-1, 1:] - zz[-1, :-1]))*dt*theta

    # bottom 
    main[1:-1, 0] = (xx[2:-1, 0] - xx[1:-2, 0])*\
        (zz[1:-1, 1] - zz[1:-1, 0]) + \
        (((k_bh[2:-1, 0]/(xin[2:, 0] - xin[1:-1, 0]) + \
        k_bh[1:-2, 0]/(xin[1:-1, 0] - xin[:-2, 0]))*\
        (zz[1:-1, 1] - zz[1:-1, 0]) + \
        (k_bv[1:-1, 1]/(zin[1:-1, 1] - zin[1:-1, 0]) + \
        9*k_bv[1:-1, 0]*alp_B/(6*(zin[1:-1, 0] - zz[1:-1, 0])))*\
        (xx[2:-1, 0] - xx[1:-2, 0]))/cc_p[1:-1, 0] + \
        (D_n/(xin[2:, 0] - xin[1:-1, 0]) + \
        D_n/(xin[1:-1, 0] - xin[:-2, 0]))*\
        (zz[1:-1, 1] - zz[1:-1, 0]) + \
        (D_n/(zin[1:-1, 1] - zin[1:-1, 0]) + \
        9*D_n*alp_B/(6*(zin[1:-1, 0] - zz[1:-1, 0])))*\
        (xx[2:-1, 0] - xx[1:-2, 0]))*dt*theta

    upper[:, 0] = -((k_bv[:, 1]/(zin[:, 1] - zin[:, 0]) + \
        k_bv[:, 0]*alp_B/(6*(zin[:, 0] - zz[:, 0])))*\
        (xx[1:, 0] - xx[:-1, 0])/cc_p[:, 1] + \
        (D_n/(zin[:, 1] - zin[:, 0]) + \
        D_n*alp_B/(6*(zin[:, 0] - zz[:, 0])))*\
        (xx[1:, 0] - xx[:-1, 0]))*dt*theta

    # top
    main[1:-1, -1] = (xx[2:-1, -1] - xx[1:-2, -1])*\
        (zz[1:-1, -1] - zz[1:-1, -2]) + \
        (((k_bh[2:-1, -1]/(xin[2:, -1] - xin[1:-1, -1]) + \
        k_bh[1:-2, -1]/(xin[1:-1, -1] - xin[:-2, -1]))*\
        (zz[1:-1, -1] - zz[1:-1, -2]) + \
        (9*k_bv[1:-1, -1]*alp_T/(6*(zz[1:-1, -1] - zin[1:-1, -1])) + \
        k_bv[1:-1, -2]/(zin[1:-1, -1] - zin[1:-1, -2]))*\
        (xx[2:-1, -1] - xx[1:-2, -1]))/cc_p[1:-1, -1] + \
        (D_n/(xin[2:, -1] - xin[1:-1, -1]) + \
        D_n/(xin[1:-1, -1] - xin[:-2, -1]))*\
        (zz[1:-1, -1] - zz[1:-1, -2]) + \
        (9*D_n*alp_T/(6*(zz[1:-1, -1] - zin[1:-1, -1])) + \
        D_n/(zin[1:-1, -1] - zin[1:-1, -2]))*\
        (xx[2:-1, -1] - xx[1:-2, -1]))*dt*theta

    lower[:, -1] = -((k_bv[:, -2]/(zin[:, -1] - zin[:, -2]) + \
        k_bv[:, -1]*alp_T/(6*(zz[:, -1] - zin[:, -1])))*\
        (xx[1:, -1] - xx[:-1, -1])/cc_p[:, -2] + \
        (D_n/(zin[:, -1] - zin[:, -2]) + \
        D_n*alp_T/(6*(zz[:, -1] - zin[:, -1])))*\
        (xx[1:, -1] - xx[:-1, -1]))*dt*theta


#=============================================================================
#================================== corners ==================================
#=============================================================================

    # bottom-left
    main[0, 0] = (xx[1, 0] - xx[0, 0])*\
        (zz[0, 1] - zz[0, 0]) + \
        (((k_bh[1, 0]/(xin[1, 0] - xin[0, 0]) + \
        9*k_bh[0, 0]*alp_L/(6*(xin[0, 0] - xx[0, 0])))*\
        (zz[0, 1] - zz[0, 0]) + \
        (k_bv[0, 1]/(zin[0, 1] - zin[0, 0]) + \
        9*k_bv[0, 0]*alp_B/(6*(zin[0, 0] - zz[0, 0])))*\
        (xx[1, 0] - xx[0, 0]))/cc_p[0, 0] + \
        (D_n/(xin[1, 0] - xin[0, 0]) + \
        9*D_n*alp_L/(6*(xin[0, 0] - xx[0, 0])))*\
        (zz[0, 1] - zz[0, 0]) + \
        (D_n/(zin[0, 1] - zin[0, 0]) + \
        9*D_n*alp_B/(6*(zin[0, 0] - zz[0, 0])))*\
        (xx[1, 0] - xx[0, 0]))*dt*theta

    # bottom-right
    main[-1, 0] = (xx[-1, 0] - xx[-2, 0])*\
        (zz[-1, 1] - zz[-1, 0]) + \
        (((9*k_bh[-1, 0]*alp_R/(6*(xx[-1, 0] - xin[-1, 0])) + \
        k_bh[-2, 0]/(xin[-1, 0] - xin[-2, 0]))*\
        (zz[-1, 1] - zz[-1, 0]) + \
        (k_bv[-1, 1]/(zin[-1, 1] - zin[-1, 0]) + \
        9*k_bv[-1, 0]*alp_B/(6*(zin[-1, 0] - zz[-1, 0])))*\
        (xx[-1, 0] - xx[-2, 0]))/cc_p[-1, 0] + \
        (9*D_n*alp_R/(6*(xx[-1, 0] - xin[-1, 0])) + \
        D_n/(xin[-1, 0] - xin[-2, 0]))*\
        (zz[-1, 1] - zz[-1, 0]) + \
        (D_n/(zin[-1, 1] - zin[-1, 0]) + \
        9*D_n*alp_B/(6*(zin[-1, 0] - zz[-1, 0])))*\
        (xx[-1, 0] - xx[-2, 0]))*dt*theta
            
    # top-left
    main[0, -1] = (xx[1, -1] - xx[0, -1])*\
        (zz[0, -1] - zz[0, -2]) + \
        (((k_bh[1, -1]/(xin[1, -1] - xin[0, -1]) + \
        9*k_bh[0, -1]*alp_L/(6*(xin[0, -1] - xx[0, -1])))*\
        (zz[0, -1] - zz[0, -2]) + \
        (9*k_bv[0, -1]*alp_T/(6*(zz[0, -1] - zin[0, -1])) + \
        k_bv[0, -2]/(zin[0, -1] - zin[0, -2]))*\
        (xx[1, -1] - xx[0, -1]))/cc_p[0, -1] + \
        (D_n/(xin[1, -1] - xin[0, -1]) + \
        9*D_n*alp_L/(6*(xin[0, -1] - xx[0, -1])))*\
        (zz[0, -1] - zz[0, -2]) + \
        (9*D_n*alp_T/(6*(zz[0, -1] - zin[0, -1])) + \
        D_n/(zin[0, -1] - zin[0, -2]))*\
        (xx[1, -1] - xx[0, -1]))*dt*theta

    # top-right
    main[-1, -1] = (xx[-1, -1] - xx[-2, -1])*\
        (zz[-1, -1] - zz[-1, -2]) + \
        (((9*k_bh[-1, -1]*alp_R/(6*(xx[-1, -1] - xin[-1, -1])) + \
        k_bh[-2, -1]/(xin[-1, -1] - xin[-2, -1]))*\
        (zz[-1, -1] - zz[-1, -2]) + \
        (9*k_bv[-1, -1]*alp_T/(6*(zz[-1, -1] - zin[-1, -1])) + \
        k_bv[-1, -2]/(zin[-1, -1] - zin[-1, -2]))*\
        (xx[-1, -1] - xx[-2, -1]))/cc_p[-1, -1] + \
        (9*D_n*alp_R/(6*(xx[-1, -1] - xin[-1, -1])) + \
        D_n/(xin[-1, -1] - xin[-2, -1]))*\
        (zz[-1, -1] - zz[-1, -2]) + \
        (9*D_n*alp_T/(6*(zz[-1, -1] - zin[-1, -1])) + \
        D_n/(zin[-1, -1] - zin[-1, -2]))*\
        (xx[-1, -1] - xx[-2, -1]))*dt*theta


#=============================================================================
#============================== compiling matrix =============================
#=============================================================================    

    main_elem = np.ravel(main, order = 'F')
    left_elem = np.ravel(left, order = 'F')[1:]
    right_elem = np.ravel(right, order = 'F')[:-1]
    lower_elem = np.ravel(lower, order = 'F')
    upper_elem = np.ravel(upper, order = 'F')

    # Computing A
    A = diags(
        diagonals = [main_elem, right_elem, left_elem, upper_elem, lower_elem],
        offsets = [0, 1, -1, Nx, -Nx], 
        shape = (N, N),
        format = 'csr'
        )

    return A
