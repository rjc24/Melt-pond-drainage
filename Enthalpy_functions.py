# Number of cores to use
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import phys_params as ppar


def Enthalpy_calc(T, C, St, Cr, OPT = 1):
    
    #========================================================================
    # Calculates enthalpy (scalar) from temperature (T), bulk salinity (C), 
    # the Stefan number (St), concentration ratio (Cr).
    # If OPT = 2, also outputs solid fraction (phi).
    #========================================================================

    # dimensionless quantities
    c_p = ppar.c_p

    # Calculating phi
    if T > C:
        phi = 0
    elif T <= C:
        phi = (T - C)/(T - Cr)
            
    # Calculating enthalpy
    H = (c_p*phi + (1 - phi))*T - St*phi

    if OPT == 1:
        return H
    if OPT == 2:
        return H, phi


def Enthalpy_invert(HH, CC, St, Cr):
    
    #========================================================================
    # Calculates temperature (array) for given enthalpy (HH), 
    # bulk salinity (CC), Stefan number (St), concentration ratio (Cr).
    #========================================================================

    # dimensionless quantities
    c_p = ppar.c_p

    TT = np.zeros(np.shape(HH))   # Temperature field

    TT[HH > CC] = HH[HH > CC]
    TT[HH <= CC] = 0.5*(St + HH[HH <= CC] + Cr + (c_p - 1)*CC[HH <= CC] - \
            np.sqrt((Cr + St + HH[HH <= CC] + (c_p - 1)*CC[HH <= CC])**2 - \
                    4*c_p*(Cr*HH[HH <= CC] + St*CC[HH <= CC])))/c_p

    return TT

def Temp_calc(HH, CC, phi, St, Cr):
    #========================================================================
    # Alternative calculation for temperature (array) using enthalpy (HH), 
    # solid fractio (phi), Stefan number (St).
    #========================================================================
    
    # dimensionless quantities
    c_p = ppar.c_p

    TT = np.zeros(np.shape(HH))   # Temperature field
    # Calculating temperature
    TT[:] = (HH[:] + St*phi[:])/(c_p*phi[:] + (1 - phi[:]))

    return TT

def Ent_calc(TT, CC, St, Cr):

    #========================================================================
    # Calculates enthalpy (array) from temperature (TT), bulk salinity (CC), 
    # the Stefan number (St), concentration ratio (Cr).
    #========================================================================
    
    # dimensionless quantities
    c_p = ppar.c_p

    HH, phi = map(np.zeros, 2*(np.shape(TT), ))
    # calculating phi
    phi[TT > CC] = 0
    phi[TT <= CC] = (TT[TT <= CC] - CC[TT <= CC])/(TT[TT <= CC] - Cr)
    # calculating enthalpy
    HH[:] = (c_p*phi[:] + (1 - phi[:]))*TT[:] - St*phi[:]

    return HH


def SFrac_calc(HH, CC, St, Cr):

    #========================================================================
    # Calculates solid fraction (array) from enthalpy (HH), bulk salinity (CC), 
    # the Stefan number (St), concentration ratio (Cr).
    #========================================================================
    
    # dimensionless quantities
    c_p = ppar.c_p
    
    phi = np.zeros(np.shape(HH))
    
    phi[HH > CC] = 0
    phi[HH <= CC] = 1/(2.0*(Cr*(c_p - 1) - St))*\
        ((c_p - 1)*CC[HH <= CC] - St - Cr + HH[HH <= CC] + \
        np.sqrt(((c_p - 1)*CC[HH <= CC] - St - Cr + HH[HH <= CC])**2 - \
        4*(Cr*(c_p - 1) - St)*(HH[HH <= CC] - CC[HH <= CC])))


    return phi

def SFrac_calc_scal(H, C, St, Cr):

    #========================================================================
    # Calculates solid fraction (scalar) from enthalpy (H), bulk salinity (C), 
    # the Stefan number (St), concentration ratio (Cr).
    #========================================================================

    # dimensionless quantities
    c_p = ppar.c_p

    # Calculating phi
    if H > C:
        phi = 0
    elif H <= C:
        phi = 1/(2.0*(Cr*(c_p - 1) - St))*((c_p - 1)*C - St - Cr + H + \
            np.sqrt(((c_p - 1)*C - St - Cr + H)**2 - \
            4*(Cr*(c_p - 1) - St)*(H - C)))

    return phi

def phi_calc(TT, CC, Cr):
    #========================================================================
    # Calculates solid fraction (array) from temperature (TT), bulk salinity (CC), 
    # concentration ratio (Cr).
    #========================================================================
    
    phi = np.zeros(np.shape(TT))

    np.putmask(phi, TT > CC, 0)
    np.putmask(phi, TT <= CC, (TT - CC)/(TT - Cr))

    return phi

def LConc(CC, phi, Cr):
    #========================================================================
    # Calculates liquid concentration (array) from bulk salinity (CC), 
    # solid fraction (phi), concentration ratio (Cr).
    #========================================================================
    
    SS = np.zeros(np.shape(CC))
    SS[:] = (CC[:] - phi[:]*Cr)/(1 - phi[:])
    
    return SS


def SFrac_bounds(phi, xin, xx, zin, zz):

    #========================================================================
    # Calculates solid fraction at cell faces, given phi (cell-centred),  
    # xin, xx, zin, zz (spatial arrays).
    #========================================================================
    
    phi_bh = np.zeros([np.shape(phi)[0]+1, np.shape(phi)[1]])
    phi_bv = np.zeros([np.shape(phi)[0], np.shape(phi)[1]+1])
    
    # horizontal faces
    phi_bh[1:-1, :] = (xin[1:, :] - xx[1:-1, :])/\
        (xin[1:, :] - xin[:-1, :])*phi[:-1, :] + \
        (xx[1:-1, :] - xin[:-1, :])/\
        (xin[1:, :] - xin[:-1, :])*phi[1:, :]
    
    # left extrapolation
    phi_bh[0, phi[0, :] == 0] = 0
    phi_bh[0, phi[0, :] != 0] = (xin[1, phi[0, :] != 0] - xx[0, phi[0, :] != 0])/\
        (xin[1, phi[0, :] != 0] - xin[0, phi[0, :] != 0])*phi[0, phi[0, :] != 0] + \
        (xx[0, phi[0, :] != 0] - xin[0, phi[0, :] != 0])/\
        (xin[1, phi[0, :] != 0] - xin[0, phi[0, :] != 0])*phi[1, phi[0, :] != 0]
        
    # right extrapolation
    phi_bh[-1, phi[-1, :] == 0] = 0
    phi_bh[-1, phi[-1, :] != 0] = (xin[-1, phi[-1, :] != 0] - xx[-1, phi[-1, :] != 0])/\
        (xin[-1, phi[-1, :] != 0] - xin[-2, phi[-1, :] != 0])*phi[-2, phi[-1, :] != 0] + \
        (xx[-1, phi[-1, :] != 0] - xin[-2, phi[-1, :] != 0])/\
        (xin[-1, phi[-1, :] != 0] - xin[-2, phi[-1, :] != 0])*phi[-1, phi[-1, :] != 0]
        
    phi_bh[phi_bh[:, :] < 0] = 0
        

    # vertical faces
    phi_bv[:, 1:-1] = (zin[:, 1:] - zz[:, 1:-1])/\
        (zin[:, 1:] - zin[:, :-1])*phi[:, :-1] + \
        (zz[:, 1:-1] - zin[:, :-1])/\
        (zin[:, 1:] - zin[:, :-1])*phi[:, 1:]

    # bottom extrapolation        
    phi_bv[phi[:, 0] == 0, 0] = 0
    phi_bv[phi[:, 0] != 0, 0] = (zin[phi[:, 0] != 0, 1] - zz[phi[:, 0] != 0, 0])/\
        (zin[phi[:, 0] != 0, 1] - zin[phi[:, 0] != 0, 0])*phi[phi[:, 0] != 0, 0] + \
        (zz[phi[:, 0] != 0, 0] - zin[phi[:, 0] != 0, 0])/\
        (zin[phi[:, 0] != 0, 1] - zin[phi[:, 0] != 0, 0])*phi[phi[:, 0] != 0, 1]
        
    # top extrapolation
    phi_bv[phi[:, -1] == 0, -1] = 0
    phi_bv[phi[:, -1] != 0, -1] = (zin[phi[:, -1] != 0, -1] - zz[phi[:, -1] != 0, -1])/\
        (zin[phi[:, -1] != 0, -1] - zin[phi[:, -1] != 0, -2])*phi[phi[:, -1] != 0, -2] + \
        (zz[phi[:, -1] != 0, -1] - zin[phi[:, -1] != 0, -2])/\
        (zin[phi[:, -1] != 0, -1] - zin[phi[:, -1] != 0, -2])*phi[phi[:, -1] != 0, -1]
        
    phi_bv[phi_bv[:, :] < 0] = 0

    return phi_bh, phi_bv


def Diff_calc(phi, EQN = 1, OPTION = 1):

    #========================================================================
    # Calculates thermal conductivity / solutal diffusivity at cell centres
    # given solid fraction field (phi). EQN = 1 (therm. cond.) or 2 (sol. diff.).
    # OPTION = 1 (realistic properties) or 2 (same in solid / liquid).
    #========================================================================


    if OPTION == 1:
        if EQN == 1:
            # liquid/solid phases have different diffusivity
            k_l, k_s = 0.58, 2.0   # liquid/solid conductivity
            v_s = k_s/k_l
            v_l = 1
        elif EQN == 2:
            # thermal (liquid) properties
            k_l, k_s = 0.58, 2.0   # conductivity
            c_pl = 4200  # heat capacity
            rho_l = 1000 # 900   # density
            kap_l = k_l/(rho_l*c_pl)
            # solute properties
            D_l = 6.8e-10

            v_s = 0
            v_l = D_l/kap_l   # diffusivity ratio

    elif OPTION == 2:
        # liquid/solid phases have same diffusivity
        v_l = v_s = 1.0

    k = np.zeros(np.size(phi))
    k[:] = v_s*phi[:] + v_l*(1 - phi[:])

    return k


def Diff_bounds(phi, xx, xin, zz, zin, EQN = 1, OPTION = 1):

    #========================================================================
    # Calculates thermal conductivity / solutal diffusivity at cell faces
    # given solid fraction field (phi) and spatial arrays (xx, xin, zz, zin). 
    # EQN = 1 (therm. cond.) or 2 (sol. diff.).
    # OPTION = 1 (realistic properties) or 2 (same in solid / liquid).
    #========================================================================

    k_bh = np.zeros([np.shape(phi)[0]+1, np.shape(phi)[1]])
    k_bv = np.zeros([np.shape(phi)[0], np.shape(phi)[1]+1])

    if OPTION == 1:
        if EQN == 1:
            # THERMAL EQUATION
            # liquid/solid phases have different diffusivity
            k_l, k_s = 0.58, 2.0   # liquid/solid conductivity
            v_s = k_s/k_l   # diffusivity ratio
            v_l = 1

            phi_bh, phi_bv = SFrac_bounds(phi, xin, xx, zin, zz)

            k_bh[:] = v_s*phi_bh[:] + v_l*(1 - phi_bh[:])
            k_bv[:] = v_s*phi_bv[:] + v_l*(1 - phi_bv[:])

        elif EQN == 2:
            # SOLUTE EQUATION
            # thermal (liquid) properties
            k_l = 0.58   # conductivity
            c_pl = 4200  # heat capacity
            rho_l = 1000 # 900   # density
            kap_l = k_l/(rho_l*c_pl)
            # solute properties
            D_l = 6.8e-10

            v_s = 0
            v_l = D_l/kap_l   # diffusivity ratio

            phi_bh, phi_bv = SFrac_bounds(phi, xin, xx, zin, zz)

            k_bh[:] = v_s*phi_bh[:] + v_l*(1 - phi_bh[:])
            k_bv[:] = v_s*phi_bv[:] + v_l*(1 - phi_bv[:])
            
            
    elif OPTION == 2:
        # liquid/solid phases have same diffusivity
        v_l = v_s = 1.0

    return k_bh, k_bv


def Var_bounds(TT, xx, xin, zz, zin, TT_BC, FF_TT, alp, EQN = 1):

    #========================================================================
    # Calculates values of given cell-centred field (TT) at cell faces and 
    # domain boundaries given spatial arrays (xx, xin, zz, zin) and
    # boundary conditions - Dirichlet (TT_BC), Neumann (FF_TT), selector (alp).
    # EQN = 1 (most fields) or 2 (solid fraction - ensures phi is non-negative).
    #========================================================================
    
    Nx, Nz = np.shape(TT)

    TT_bh = np.zeros([Nx+1, Nz])
    TT_bv = np.zeros([Nx, Nz+1])
    
    # boundary condition type
    # alp = 1 - Dirichlet, alp = 0 - Neumann
    alp_L, alp_R, alp_B, alp_T = alp[0], alp[1], alp[2], alp[3]
    TT_L, TT_R, TT_B, TT_T = TT_BC[0], TT_BC[1], TT_BC[2], TT_BC[3]
    FF_L, FF_R, FF_B, FF_T = FF_TT[0], FF_TT[1], FF_TT[2], FF_TT[3]
        
    # horizontal faces
    TT_bh[1:-1, :] = (xin[1:, :] - xx[1:-1, :])/\
        (xin[1:, :] - xin[:-1, :])*TT[:-1, :] + \
        (xx[1:-1, :] - xin[:-1, :])/\
        (xin[1:, :] - xin[:-1, :])*TT[1:, :]

    # left extrapolation
    if alp_L == 1:
        TT_bh[0, :] = TT_L
    elif alp_L == 0:
        a_L, b_L, c_L = map(np.zeros, 3*(Nz, ))
        a_L = (TT[1, :] - TT[0, :])/\
            ((xin[1, :] - xin[0, :])*(xin[1, :] + xin[0, :] - 2*xx[0, :])) - \
            FF_L[:]/(xin[1, :] + xin[0, :] - 2*xx[0, :])
        b_L = FF_L[:] - 2*a_L[:]*xx[0, :]
        c_L = TT[0, :] - a_L[:]*xin[0, :]**2 - b_L[:]*xin[0, :]
        TT_bh[0, :] = a_L[:]*xx[0, :]**2 + b_L[:]*xx[0, :] + c_L[:]
        
    if EQN == 2:
        TT_bh[0, TT[0, :] == 0] = 0
        
    # right extrapolation
    if alp_R == 1:
        TT_bh[-1, :] = TT_R
    elif alp_R == 0:
        a_R, b_R, c_R = map(np.zeros, 3*(Nz, ))
        a_R = (TT[-1, :] - TT[-2, :])/\
            ((xin[-1, :] - xin[-2, :])*(xin[-1, :] + xin[-2, :] - 2*xx[-1, :])) - \
            FF_R[:]/(xin[-1, :] + xin[-2, :] - 2*xx[-1, :])
        b_R = FF_R[:] - 2*a_R[:]*xx[-1, :]
        c_R = TT[-1, :] - a_R[:]*xin[-1, :]**2 - b_R[:]*xin[-1, :]
        TT_bh[-1, :] = a_R[:]*xx[-1, :]**2 + b_R[:]*xx[-1, :] + c_R[:]

    if EQN == 2:
        TT_bh[-1, TT[-1, :] == 0] = 0


    # vertical faces
    TT_bv[:, 1:-1] = (zin[:, 1:] - zz[:, 1:-1])/\
        (zin[:, 1:] - zin[:, :-1])*TT[:, :-1] + \
        (zz[:, 1:-1] - zin[:, :-1])/\
        (zin[:, 1:] - zin[:, :-1])*TT[:, 1:]

    # bottom extrapolation    
    if alp_B == 1:
        TT_bv[:, 0] = TT_B
    elif alp_B == 0:
        a_B, b_B, c_B = map(np.zeros, 3*(Nx, ))
        a_B = (TT[:, 1] - TT[:, 0])/\
            ((zin[:, 1] - zin[:, 0])*(zin[:, 1] + zin[:, 0] - 2*zz[:, 0])) - \
            FF_B[:]/(zin[:, 1] + zin[:, 0] - 2*zz[:, 0])
        b_B = FF_B[:] - 2*a_B[:]*zz[:, 0]
        c_B = TT[:, 0] - a_B[:]*zin[:, 0]**2 - b_B[:]*zin[:, 0]
        TT_bv[:, 0] = a_B[:]*zz[:, 0]**2 + b_B[:]*zz[:, 0] + c_B[:]
        
    if EQN == 2:
        TT_bv[TT[:, 0] == 0, 0] = 0

    # top extrapolation
    if alp_T == 1:
        TT_bv[:, -1] = TT_T
    elif alp_T == 0:
        a_T, b_T, c_T = map(np.zeros, 3*(Nx, ))
        a_T = (TT[:, -1] - TT[:, -2])/\
            ((zin[:, -1] - zin[:, -2])*(zin[:, -1] + zin[:, -2] - 2*zz[:, -1])) - \
            FF_T[:]/(zin[:, -1] + zin[:, -2] - 2*zz[:, -1])
        b_T = FF_T[:] - 2*a_T[:]*zz[:, -1]
        c_T = TT[:, -1] - a_T[:]*zin[:, -1]**2 - b_T[:]*zin[:, -1]
        TT_bv[:, -1] = a_T[:]*zz[:, -1]**2 + b_T[:]*zz[:, -1] + c_T[:]

    if EQN == 2:
        TT_bv[TT[:, -1] == 0, -1] = 0


    # ensuring solid fraction is non-negative
    if EQN == 2:
        TT_bh[TT_bh[:, :] <= 0] = 0
        TT_bv[TT_bv[:, :] <= 0] = 0

    return TT_bh, TT_bv



def Var_bounds_up(TT, u_b, w_b, xx, xin, zz, zin, TT_BC, FF_TT, alp, EQN = 1):

    #========================================================================
    # Calculates values of given cell-centred field (TT) at cell faces and 
    # domain boundaries given spatial arrays (xx, xin, zz, zin) and
    # boundary conditions - Dirichlet (TT_BC), Neumann (FF_TT), selector (alp).
    # Uses simple upwinding using horizotal, vertical velocities (u_b, w_b)
    # rather than lin. interpolation as above.
    # EQN = 1 (most fields) or 2 (solid fraction - ensures phi is non-negative).
    #========================================================================
    
    Nx, Nz = np.shape(TT)

    TT_bh, T_L, T_R = map(np.zeros, 3*([Nx+1, Nz], ))
    TT_bv, T_B, T_T = map(np.zeros, 3*([Nx, Nz+1], ))
    
    # boundary condition type
    # alp = 1 - Dirichlet, alp = 0 - Neumann
    alp_L, alp_R, alp_B, alp_T = alp[0], alp[1], alp[2], alp[3]
    TT_L, TT_R, TT_B, TT_T = TT_BC[0], TT_BC[1], TT_BC[2], TT_BC[3]
    FF_L, FF_R, FF_B, FF_T = FF_TT[0], FF_TT[1], FF_TT[2], FF_TT[3]
        
    TT_bhc, TT_bvc = Var_bounds(TT, xx, xin, zz, zin, TT_BC, FF_TT, alp, EQN = 1)
    
    # left-extrapolations
    T_L[1, :] = TT[0, :] + \
        (xx[1, :] - xin[0, :])*(TT[1, :] - TT_bhc[0, :])/(xin[1, :] - xx[0, :])
    T_L[2:-1, :] = TT[1:-1, :] + \
        (xx[2:-1, :] - xin[1:-1, :])*(TT[2:, :] - TT[:-2, :])/(xin[2:, :] - xin[:-2, :])
    T_L[-1, :] = TT[-1, :] + \
        (xx[-1, :] - xin[-1, :])*(TT_bhc[-1, :] - TT[-2, :])/(xx[-1, :] - xin[-2, :])
    
    # right-extrapolations
    T_R[0, :] = TT[0, :] - \
        (xin[0, :] - xx[0, :])*(TT[1, :] - TT_bhc[0, :])/(xin[1, :] - xx[0, :])
    T_R[1:-2, :] = TT[1:-1, :] - \
        (xin[1:-1, :] - xx[1:-2, :])*(TT[2:, :] - TT[:-2, :])/(xin[2:, :] - xin[:-2, :])
    T_R[-2, :] = TT[-1, :] - \
        (xin[-1, :] - xx[-2, :])*(TT_bhc[-1, :] - TT[-2, :])/(xx[-1, :] - xin[-2, :])
    
    # bottom-extrapolations
    T_B[:, 1] = TT[:, 0] + \
        (zz[:, 1] - zin[:, 0])*(TT[:, 1] - TT_bvc[:, 0])/(zin[:, 1] - zz[:, 0])
    T_B[:, 2:-1] = TT[:, 1:-1] + \
        (zz[:, 2:-1] - zin[:, 1:-1])*(TT[:, 2:] - TT[:, :-2])/(zin[:, 2:] - zin[:, :-2])
    T_B[:, -1] = TT[:, -1] + \
        (zz[:, -1] - zin[:, -1])*(TT_bvc[:, -1] - TT[:, -2])/(zz[:, -1] - zin[:, -2])
    
    # top-extrapolations
    T_T[:, 0] = TT[:, 0] - \
        (zin[:, 0] - zz[:, 0])*(TT[:, 1] - TT_bvc[:, 0])/(zin[:, 1] - zz[:, 0])
    T_T[:, 1:-2] = TT[:, 1:-1] - \
        (zin[:, 1:-1] - zz[:, 1:-2])*(TT[:, 2:] - TT[:, :-2])/(zin[:, 2:] - zin[:, :-2])
    T_T[:, -2] = TT[:, -1] - \
        (zin[:, -1] - zz[:, -2])*(TT_bvc[:, -1] - TT[:, -2])/(zz[:, -1] - zin[:, -2])
    
    TT_bh[0, :] = np.fmax(np.sign(u_b[0, :]), 0)*TT_bhc[0, :] - \
                  np.fmin(np.sign(u_b[0, :]), 0)*T_R[0, :]
    TT_bh[1:-1, :] = np.fmax(np.sign(u_b[1:-1, :]), 0)*T_L[1:-1, :] - \
                     np.fmin(np.sign(u_b[1:-1, :]), 0)*T_R[1:-1, :] + \
                     (1 - abs(np.sign(u_b[1:-1, :])))*(T_L[1:-1, :] + T_R[1:-1, :])/2
    TT_bh[-1, :] = np.fmax(np.sign(u_b[-1, :]), 0)*T_L[-1, :] - \
                   np.fmin(np.sign(u_b[-1, :]), 0)*TT_bhc[-1, :]
    
    TT_bv[:, 0] = np.fmax(np.sign(w_b[:, 0]), 0)*TT_bvc[:, 0] - \
                  np.fmin(np.sign(w_b[:, 0]), 0)*T_T[:, 0]
    TT_bv[:, 1:-1] = np.fmax(np.sign(w_b[:, 1:-1]), 0)*T_B[:, 1:-1] - \
                     np.fmin(np.sign(w_b[:, 1:-1]), 0)*T_T[:, 1:-1] + \
                     (1 - abs(np.sign(w_b[:, 1:-1])))*(T_B[:, 1:-1] + T_T[:, 1:-1])/2
    TT_bv[:, -1] = np.fmax(np.sign(w_b[:, -1]), 0)*T_B[:, -1] - \
                   np.fmin(np.sign(w_b[:, -1]), 0)*TT_bvc[:, -1]

    return TT_bh, TT_bv


def Perm_calc(phi, xx, xin, zz, zin, phi_BC, FF_phi, alp, Da_h):
    
    #========================================================================
    # Calculates permeability at cell centres and faces from solid fraction (phi),
    # spatial arrays (xx, xin, zz, zin), boundary conditions - Dirichlet (phi_BC), 
    # Neumann (FF_phi), selector (alp), Darcy number (Da_h).
    #========================================================================
    
    Nx, Nz = np.shape(phi)
    
    Pi = np.zeros([Nx, Nz])
    Pi_bh = np.zeros([Nx+1, Nz])
    Pi_bv = np.zeros([Nx, Nz+1])
    
    # calculating cell-face values of phi   - had EQN = 1 here before
    phi_bh, phi_bv = Var_bounds(phi, xx, xin, zz, zin, phi_BC, FF_phi, alp, EQN = 2)
    
    # cell-centred Pi
    Pi[:, :] = Da_h*(1 - phi[:, :])**3/(Da_h*(1 - phi[:, :])**3 + phi[:, :]**2)

    # face-centred Pi
    # geometric mean
    Pi_bh[1:-1, :] = np.sqrt(Pi[1:, :]*Pi[:-1, :])
    Pi_bv[:, 1:-1] = np.sqrt(Pi[:, 1:]*Pi[:, :-1])

    Pi_bh[0, :] = Da_h*(1 - phi_bh[0, :])**3/(Da_h*(1 - phi_bh[0, :])**3 + \
                                          phi_bh[0, :]**2)
    Pi_bh[-1, :] = Da_h*(1 - phi_bh[-1, :])**3/(Da_h*(1 - phi_bh[-1, :])**3 + \
                                          phi_bh[-1, :]**2)
        
    Pi_bv[:, 0] = Da_h*(1 - phi_bv[:, 0])**3/(Da_h*(1 - phi_bv[:, 0])**3 + \
                                          phi_bv[:, 0]**2)
    Pi_bv[:, -1] = Da_h*(1 - phi_bv[:, -1])**3/(Da_h*(1 - phi_bv[:, -1])**3 + \
                                          phi_bv[:, -1]**2)

    
    return Pi, Pi_bh, Pi_bv

