# Number of cores to use
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from scipy.sparse import diags

def A_theta(method, xx, xin, dt, cc_p, k_b, alp):
    
    #========================================================================
    # Function which creates the left hand side matrix for the heat / 
    # solute conservation equation
    #
    # Inputs:
    #    method - 'BTCS' (backwards Euler) or 'Crank' (Crank-Nicholson)
    #    xx, xin - spatial arrays
    #    dt - time step
    #    cc_p - specific heat capacity / porosity
    #    k_b - thermal conductivity / solutal diffusivity
    #    alp - boundary conditions
    #
    # Outputs:
    #    A - tridiagonal matrix
    #
    #========================================================================

    # Ensuring supported method has been chosen
    if not method in ('Crank', 'BTCS'):
        raise SyntaxError('Must choose either Crank or BTCS for method.')
        return

    Nx = np.size(xin)    # number of grid points
    
    # Diagonal entries of matrix A
    main = np.zeros(Nx)
    lower = np.zeros(Nx-1)
    upper = np.zeros(Nx-1)
    
    # Defining theta 
    if method == 'BTCS':
        theta = 1        
    elif method == 'Crank':
        theta = 0.5
        
    # alp = 1 - Dirichlet, alp = 0 - Neumann
    alp_L, alp_R = alp[0], alp[1]


#================================= Main body =================================

    # matrix
    main[1:-1] = (xx[2:-1]**2 - xx[1:-2]**2)/2 + \
        (k_b[2:-1]*xx[2:-1]/(xin[2:] - xin[1:-1]) + \
        k_b[1:-2]*xx[1:-2]/(xin[1:-1] - xin[:-2]))/cc_p[1:-1]*dt*theta

    lower[:-1] = -k_b[1:-2]*xx[1:-2]/cc_p[:-2]/(xin[1:-1] - xin[:-2])*dt*theta
    upper[1:] = -k_b[2:-1]*xx[2:-1]/cc_p[2:]/(xin[2:] - xin[1:-1])*dt*theta


#=============================== Left boundary ===============================

    main[0] = (xx[1]**2 - xx[0]**2)/2 + \
        (k_b[1]*xx[1]/(xin[1] - xin[0]) + \
         9*k_b[0]*xx[0]*alp_L/(6*(xin[0] - xx[0])))/cc_p[0]*dt*theta
    upper[0] = -(k_b[1]*xx[1]/(xin[1] - xin[0]) + \
        k_b[0]*xx[0]*alp_L/(6*(xin[0] - xx[0])))/cc_p[1]*dt*theta

#=============================== Right boundary ===============================


    main[-1] = (xx[-1]**2 - xx[-2]**2)/2 + \
        (9*k_b[-1]*xx[-1]*alp_R/(6*(xx[-1] - xin[-1])) + \
        k_b[-2]*xx[-2]/(xin[-1] - xin[-2]))/cc_p[-1]*dt*theta
        
    lower[-1] = -(k_b[-2]*xx[-2]/(xin[-1] - xin[-2]) + \
        k_b[-1]*xx[-1]*alp_R/(6*(xx[-1] - xin[-1])))/cc_p[-2]*dt*theta


    # Computing A
    A = diags(
        diagonals = [main, upper, lower],
        offsets = [0, 1, -1], 
        shape = (Nx, Nx),
        format = 'csr'
        )

    return A