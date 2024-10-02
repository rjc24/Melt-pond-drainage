# Number of cores to use
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def p_solve(xx, xin, zz, zin, Pi_bh, Pi_bv, pp_BC, FF, alp, Pe, G = None):

    #========================================================================
    # Solves pressure-Poisson equation to calculate pressure field and 
    # velocity components
    #
    # Inputs:
    #    xx, xin, zz, zin - spatial arrays
    #    Pi_bh, Pi_bv - permeability at cell faces
    #    pp_BC - Dirichlet BCs for pressure
    #    FF - Neumann BCs for pressure
    #    alp - boundary conditions
    #    Pe - Peclet number
    #    G - source term
    #
    # Outputs:
    #    pp - pressure field
    #    u_b, w_b - horizontal, vertical velocities
    #
    #========================================================================


    Nx, Nz = np.shape(xin)[0], np.shape(xin)[1]    # number of grid points
    N = Nz*Nx

    pp = np.zeros([Nx, Nz])
    pp_p = np.zeros(N)
#    pp_bh, pp_bv = np.zeros([Nx+1, Nz]), np.zeros([Nx, Nz+1])
    u_b, w_b = np.zeros([Nx+1, Nz]), np.zeros([Nx, Nz+1])
    
    # Diagonal entries of matrix A
    main = np.zeros([Nx, Nz])
    left, right = np.zeros([Nx, Nz]), np.zeros([Nx, Nz])
    lower, upper = np.zeros([Nx, Nz-1]), np.zeros([Nx, Nz-1])

    # vector b
    b = np.zeros([Nx, Nz])

    # pressure derivatives
    dpdr = np.zeros(np.shape(xx))
    dpdz = np.zeros(np.shape(zz))

    # alp = 1 - Dirichlet, alp = 0 - Neumann
    alp_L, alp_R, alp_B, alp_T = alp[0], alp[1], alp[2], alp[3]

    # boundary values
    # Dirichlet
    pp_L, pp_R, pp_B, pp_T = pp_BC[0], pp_BC[1], pp_BC[2], pp_BC[3]
    # Neumann
    FF_L, FF_R, FF_B, FF_T = FF[0], FF[1], FF[2], FF[3]
    

#=============================================================================
#=============================================================================
#==================================== MATRIX =================================
#=============================================================================
#=============================================================================

#=============================================================================
#=================================== main body ===============================
#=============================================================================
        

    # alt formulation
    main[1:-1, 1:-1] = (Pi_bh[2:-1, 1:-1]/(xin[2:, 1:-1] - xin[1:-1, 1:-1]) + \
        Pi_bh[1:-2, 1:-1]/(xin[1:-1, 1:-1] - xin[:-2, 1:-1]))/\
        (xx[2:-1, 1:-1] - xx[1:-2, 1:-1]) + \
        (Pi_bv[1:-1, 2:-1]/(zin[1:-1, 2:] - zin[1:-1, 1:-1]) + \
        Pi_bv[1:-1, 1:-2]/(zin[1:-1, 1:-1] - zin[1:-1, :-2]))/\
        (zz[1:-1, 2:-1] - zz[1:-1, 1:-2])
        

    left[1:-1, :] = -Pi_bh[1:-2, :]/\
        ((xin[1:-1, :] - xin[:-2, :])*(xx[2:-1, :] - xx[1:-2, :]))
    right[1:-1, :] = -Pi_bh[2:-1, :]/\
        ((xin[2:, :] - xin[1:-1, :])*(xx[2:-1, :] - xx[1:-2, :]))

    lower[:, :-1] = -Pi_bv[:, 1:-2]/\
        ((zin[:, 1:-1] - zin[:, :-2])*(zz[:, 2:-1] - zz[:, 1:-2]))
    upper[:, 1:] = -Pi_bv[:, 2:-1]/\
        ((zin[:, 2:] - zin[:, 1:-1])*(zz[:, 2:-1] - zz[:, 1:-2]))


#=============================================================================
#================================== boundaries ===============================
#=============================================================================

    # left
    main[0, 1:-1] = (Pi_bh[1, 1:-1]/(xin[1, 1:-1] - xin[0, 1:-1]) + \
        9*Pi_bh[0, 1:-1]*alp_L/(6*(xin[0, 1:-1] - xx[0, 1:-1])))/\
        (xx[1, 1:-1] - xx[0, 1:-1]) + \
        (Pi_bv[0, 2:-1]/(zin[0, 2:] - zin[0, 1:-1]) + \
        Pi_bv[0, 1:-2]/(zin[0, 1:-1] - zin[0, :-2]))/\
        (zz[0, 2:-1] - zz[0, 1:-2])
        
    right[0, :] = -(Pi_bh[1, :]/(xin[1, :] - xin[0, :]) + \
        Pi_bh[0, :]*alp_L/(6*(xin[0, :] - xx[0, :])))/\
        (xx[1, :] - xx[0, :])

    # right
    main[-1, 1:-1] = (9*Pi_bh[-1, 1:-1]*alp_R/(6*(xx[-1, 1:-1] - xin[-1, 1:-1])) + \
        Pi_bh[-2, 1:-1]/(xin[-1, 1:-1] - xin[-2, 1:-1]))/\
        (xx[-1, 1:-1] - xx[-2, 1:-1]) + \
        (Pi_bv[-1, 2:-1]/(zin[-1, 2:] - zin[-1, 1:-1]) + \
        Pi_bv[-1, 1:-2]/(zin[-1, 1:-1] - zin[-1, :-2]))/\
        (zz[-1, 2:-1] - zz[-1, 1:-2])
        
    left[-1, :] = -(Pi_bh[-2, :]/(xin[-1, :] - xin[-2, :]) + \
        Pi_bh[-1, :]*alp_R/(6*(xx[-1, :] - xin[-1, :])))/\
        (xx[-1, :] - xx[-2, :])

    # bottom 
    main[1:-1, 0] = (Pi_bh[2:-1, 0]/(xin[2:, 0] - xin[1:-1, 0]) + \
        Pi_bh[1:-2, 0]/(xin[1:-1, 0] - xin[:-2, 0]))/\
        (xx[2:-1, 0] - xx[1:-2, 0]) + \
        (Pi_bv[1:-1, 1]/(zin[1:-1, 1] - zin[1:-1, 0]) + \
        9*Pi_bv[1:-1, 0]*alp_B/(6*(zin[1:-1, 0] - zz[1:-1, 0])))/\
        (zz[1:-1, 1] - zz[1:-1, 0])
        
    upper[:, 0] = -(Pi_bv[:, 1]/(zin[:, 1] - zin[:, 0]) + \
        Pi_bv[:, 0]*alp_B/(6*(zin[:, 0] - zz[:, 0])))/\
        (zz[:, 1] - zz[:, 0])

    # top
    main[1:-1, -1] = (Pi_bh[2:-1, -1]/(xin[2:, -1] - xin[1:-1, -1]) + \
        Pi_bh[1:-2, -1]/(xin[1:-1, -1] - xin[:-2, -1]))/\
        (xx[2:-1, -1] - xx[1:-2, -1]) + \
        (9*Pi_bv[1:-1, -1]*alp_T/(6*(zz[1:-1, -1] - zin[1:-1, -1])) + \
        Pi_bv[1:-1, -2]/(zin[1:-1, -1] - zin[1:-1, -2]))/\
        (zz[1:-1, -1] - zz[1:-1, -2])
        
    lower[:, -1] = -(Pi_bv[:, -2]/(zin[:, -1] - zin[:, -2]) + \
        Pi_bv[:, -1]*alp_T/(6*(zz[:, -1] - zin[:, -1])))/\
        (zz[:, -1] - zz[:, -2])
        

#=============================================================================
#================================== corners ==================================
#=============================================================================

    # bottom-left
    main[0, 0] = (Pi_bh[1, 0]/(xin[1, 0] - xin[0, 0]) + \
        9*Pi_bh[0, 0]*alp_L/(6*(xin[0, 0] - xx[0, 0])))/\
        (xx[1, 0] - xx[0, 0]) + \
        (Pi_bv[0, 1]/(zin[0, 1] - zin[0, 0]) + \
        9*Pi_bv[0, 0]*alp_B/(6*(zin[0, 0] - zz[0, 0])))/\
        (zz[0, 1] - zz[0, 0])

    # bottom-right
    main[-1, 0] = (9*Pi_bh[-1, 0]*alp_R/(6*(xx[-1, 0] - xin[-1, 0])) + \
        Pi_bh[-2, 0]/(xin[-1, 0] - xin[-2, 0]))/\
        (xx[-1, 0] - xx[-2, 0]) + \
        (Pi_bv[-1, 1]/(zin[-1, 1] - zin[-1, 0]) + \
        9*Pi_bv[-1, 0]*alp_B/(6*(zin[-1, 0] - zz[-1, 0])))/\
        (zz[-1, 1] - zz[-1, 0])

    # top-left
    main[0, -1] = (Pi_bh[1, -1]/(xin[1, -1] - xin[0, -1]) + \
        9*Pi_bh[0, -1]*alp_L/(6*(xin[0, -1] - xx[0, -1])))/\
        (xx[1, -1] - xx[0, -1]) + \
        (9*Pi_bv[0, -1]*alp_T/(6*(zz[0, -1] - zin[0, -1])) + \
        Pi_bv[0, -2]/(zin[0, -1] - zin[0, -2]))/\
        (zz[0, -1] - zz[0, -2])

    # top-right
    main[-1, -1] = (9*Pi_bh[-1, -1]*alp_R/(6*(xx[-1, -1] - xin[-1, -1])) + \
        Pi_bh[-2, -1]/(xin[-1, -1] - xin[-2, -1]))/\
        (xx[-1, -1] - xx[-2, -1]) + \
        (9*Pi_bv[-1, -1]*alp_T/(6*(zz[-1, -1] - zin[-1, -1])) + \
        Pi_bv[-1, -2]/(zin[-1, -1] - zin[-1, -2]))/\
        (zz[-1, -1] - zz[-1, -2])

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
    

#=============================================================================
#=============================================================================
#==================================== VECTOR =================================
#=============================================================================
#=============================================================================


#=============================================================================
#=================================== main body ===============================
#=============================================================================

    b[:, :] = -G[:, :]   #  accounting for source term
    
    
#=============================================================================
#================================== boundaries ===============================
#=============================================================================

    b[0, 1:-1] += Pi_bh[0, 1:-1]*(alp_L*8*pp_L[1:-1]/(6*(xin[0, 1:-1] - xx[0, 1:-1])) - \
              (1 - alp_L)*FF_L[1:-1])/(xx[1, 1:-1] - xx[0, 1:-1])
    b[-1, 1:-1] += Pi_bh[-1, 1:-1]*(alp_R*8*pp_R[1:-1]/(6*(xx[-1, 1:-1] - xin[-1, 1:-1])) + \
               (1 - alp_R)*FF_R[1:-1])/(xx[-1, 1:-1] - xx[-2, 1:-1])
    b[1:-1, 0] += Pi_bv[1:-1, 0]*(alp_B*8*pp_B[1:-1]/(6*(zin[1:-1, 0] - zz[1:-1, 0])) - \
              (1 - alp_B)*FF_B[1:-1])/(zz[1:-1, 1] - zz[1:-1, 0])
    b[1:-1, -1] += Pi_bv[1:-1, -1]*(alp_T*8*pp_T[1:-1]/(6*(zz[1:-1, -1] - zin[1:-1, -1])) + \
               (1 - alp_T)*FF_T[1:-1])/(zz[1:-1, -1] - zz[1:-1, -2])

#=============================================================================
#================================== corners ==================================
#=============================================================================

    b[0, 0] += Pi_bh[0, 0]*(alp_L*8*pp_L[0]/(6*(xin[0, 0] - xx[0, 0])) - \
              (1 - alp_L)*FF_L[0])/(xx[1, 0] - xx[0, 0]) + \
              Pi_bv[0, 0]*(alp_B*8*pp_B[0]/(6*(zin[0, 0] - zz[0, 0])) - \
              (1 - alp_B)*FF_B[0])/(zz[0, 1] - zz[0, 0])

    b[0, -1] += Pi_bh[0, -1]*(alp_L*8*pp_L[-1]/(6*(xin[0, -1] - xx[0, -1])) - \
               (1 - alp_L)*FF_L[-1])/(xx[1, -1] - xx[0, -1]) + \
               Pi_bv[0, -1]*(alp_T*8*pp_T[0]/(6*(zz[0, -1] - zin[0, -1])) + \
               (1 - alp_T)*FF_T[0])/(zz[0, -1] - zz[0, -2])

    b[-1, 0] += Pi_bh[-1, 0]*(alp_R*8*pp_R[0]/(6*(xx[-1, 0] - xin[-1, 0])) + \
               (1 - alp_R)*FF_R[0])/(xx[-1, 0] - xx[-2, 0]) + \
               Pi_bv[-1, 0]*(alp_B*8*pp_B[-1]/(6*(zin[-1, 0] - zz[-1, 0])) - \
               (1 - alp_B)*FF_B[-1])/(zz[-1, 1] - zz[-1, 0])

    b[-1, -1] += Pi_bh[-1, -1]*(alp_R*8*pp_R[-1]/(6*(xx[-1, -1] - xin[-1, -1])) + \
               (1 - alp_R)*FF_R[-1])/(xx[-1, -1] - xx[-2, -1]) + \
               Pi_bv[-1, -1]*(alp_T*8*pp_T[-1]/(6*(zz[-1, -1] - zin[-1, -1])) + \
               (1 - alp_T)*FF_T[-1])/(zz[-1, -1] - zz[-1, -2])
    
        

    b_vec = np.ravel(b, order = 'F')

    # solving
    pp_p[:] = spsolve(A, b_vec)

    # re-shaping
    pp[:] = pp_p.reshape([Nx, Nz], order = 'F')[:]

    # calculating velocities

    u_b[1:-1, :] = -Pe*Pi_bh[1:-1, :]*(pp[1:, :] - pp[:-1, :])/\
                    (xin[1:, :] - xin[:-1, :])
    u_b[0, :] = -Pe*Pi_bh[0, :]*(alp_L*(-8*pp_L[:] + 9*pp[0, :] - pp[1, :])/\
                    (6*(xin[0, :] - xx[0, :])) + (1 - alp_L)*FF_L[:])
    u_b[-1, :] = -Pe*Pi_bh[-1, :]*(alp_R*(8*pp_R[:] - 9*pp[-1, :] + pp[-2, :])/\
                    (6*(xx[-1, :] - xin[-1, :])) + (1 - alp_R)*FF_R[:])

    w_b[:, 1:-1] = -Pe*Pi_bv[:, 1:-1]*(pp[:, 1:] - pp[:, :-1])/\
                    (zin[:, 1:] - zin[:, :-1])
    w_b[:, 0] = -Pe*Pi_bv[:, 0]*(alp_B*(-8*pp_B[:] + 9*pp[:, 0] - pp[:, 1])/\
                    (6*(zin[:, 0] - zz[:, 0])) + (1 - alp_B)*FF_B[:])
    w_b[:, -1] = -Pe*Pi_bv[:, -1]*(alp_T*(8*pp_T[:] - 9*pp[:, -1] + pp[:, -2])/\
                    (6*(zz[:, -1] - zin[:, -1])) + (1 - alp_T)*FF_T[:])

    return pp, u_b, w_b


def velocities(xx, xin, zz, zin, pp, Pi_bh, Pi_bv, pp_BC, FF, alp, Pe):
    
    #========================================================================
    # Calculates velocities from pressure field without solving pressure
    # Poisson equation
    #
    # Inputs:
    #    xx, xin, zz, zin - spatial arrays
    #    pp - pressure field
    #    Pi_bh, Pi_bv - permeability at cell faces
    #    pp_BC - Dirichlet BCs for pressure
    #    FF - Neumann BCs for pressure
    #    alp - boundary conditions
    #    Pe - Peclet number
    #
    # Outputs:
    #    u_b, w_b - horizontal, vertical velocities
    #
    #========================================================================
    
    Nx, Nz = np.shape(xin)[0], np.shape(xin)[1]    # number of grid points
    u_b, w_b = np.zeros([Nx+1, Nz]), np.zeros([Nx, Nz+1])
    
    # alp = 1 - Dirichlet, alp = 0 - Neumann
    alp_L, alp_R, alp_B, alp_T = alp[0], alp[1], alp[2], alp[3]
    
    # boundary values
    # Dirichlet
    pp_L, pp_R, pp_B, pp_T = pp_BC[0], pp_BC[1], pp_BC[2], pp_BC[3]
    # Neumann
    FF_L, FF_R, FF_B, FF_T = FF[0], FF[1], FF[2], FF[3]
    


    u_b[1:-1, :] = -Pe*Pi_bh[1:-1, :]*(pp[1:, :] - pp[:-1, :])/\
                    (xin[1:, :] - xin[:-1, :])
    u_b[0, :] = -Pe*Pi_bh[0, :]*(alp_L*(-8*pp_L[:] + 9*pp[0, :] - pp[1, :])/\
                    (6*(xin[0, :] - xx[0, :])) + (1 - alp_L)*FF_L[:])
    u_b[-1, :] = -Pe*Pi_bh[-1, :]*(alp_R*(8*pp_R[:] - 9*pp[-1, :] + pp[-2, :])/\
                    (6*(xx[-1, :] - xin[-1, :])) + (1 - alp_R)*FF_R[:])

    w_b[:, 1:-1] = -Pe*Pi_bv[:, 1:-1]*(pp[:, 1:] - pp[:, :-1])/\
                    (zin[:, 1:] - zin[:, :-1])
    w_b[:, 0] = -Pe*Pi_bv[:, 0]*(alp_B*(-8*pp_B[:] + 9*pp[:, 0] - pp[:, 1])/\
                    (6*(zin[:, 0] - zz[:, 0])) + (1 - alp_B)*FF_B[:])
    w_b[:, -1] = -Pe*Pi_bv[:, -1]*(alp_T*(8*pp_T[:] - 9*pp[:, -1] + pp[:, -2])/\
                    (6*(zz[:, -1] - zin[:, -1])) + (1 - alp_T)*FF_T[:])

    return u_b, w_b


