# Number of cores to use
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from scipy.optimize import newton
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

def phi_calc_scal(T, C, Cr):
    
    #========================================================================
    # Calculates solid fraction (scalar) from temperature (T), bulk salinity (C), 
    # concentration ratio (Cr).
    #========================================================================
    
    if T > C:
        phi = 0
    elif T <= C:
        phi = (T - C)/(T - Cr)

    return phi

def LConc(CC, phi, Cr):
    #========================================================================
    # Calculates liquid concentration (array) from bulk salinity (CC), 
    # solid fraction (phi), concentration ratio (Cr).
    #========================================================================
    
    SS = np.zeros(np.shape(CC))
    SS[:] = (CC[:] - phi[:]*Cr)/(1 - phi[:])
    
    return SS

def LConc_scal(C, phi, Cr):
    
    #========================================================================
    # Calculates liquid concentration (scalar) from bulk salinity (C), 
    # solid fraction (phi), concentration ratio (Cr).
    #========================================================================
    
    S = (C - phi*Cr)/(1 - phi)
    
    return S

def Salinity_calc(SS, phi, Cr):
    
    #========================================================================
    # Calculates bulk salinty (array) from liquid concentration (SS), 
    # solid fraction (phi), concentration ratio (Cr).
    #========================================================================
    
    CC = np.zeros(np.shape(SS))
    CC[:] = (1 - phi[:])*SS[:] + phi[:]*Cr
    
    return CC

def Q_Vflux(a, xx, xin, Q_T):

    #========================================================================
    # Calculates volumetric flux term.
    # Inputs : temperature field (TT), spatial arrays (xx, xin), 
    # channel heating term (Q_T).
    #========================================================================

    Nx = np.size(xin)

    if not np.isnan(a):
        if a < xx[-1]:
            j = [i for (i,n) in enumerate(xx) if n <= a][-1]   # Boundary index
        else:
            a = xx[-1]
            j = Nx
    else:
        a = xx[-1]
        j = Nx

    Q_v = np.zeros(Nx)
    if j >= Nx:
        Q_v[:] = Q_T
    elif j <= Nx-1:
        Q_v[:j] = Q_T
        if a - xx[j] < 1e-4*(xx[j+1] - xx[j]):
            Q_v[j] = 0
        elif xx[j+1] - a < 1e-4*(xx[j+1] - xx[j]):
            Q_v[j] = Q_T
        else:
            Q_v[j] = Q_T*(a**2 - xx[j]**2)/(xx[j+1]**2 - xx[j]**2)
        Q_v[j+1:] = 0

    return Q_v


def SFrac_bounds(phi, xin, xx):

    #========================================================================
    # Calculates solid fraction at cell faces, given phi (cell-centred),  
    # xin, xx (spatial arrays).
    #========================================================================

    phi_b = np.zeros(np.size(phi) + 1)

    # horizontal faces
    phi_b[1:-1] = (xin[1:] - xx[1:-1])/(xin[1:] - xin[:-1])*phi[:-1] + \
        (xx[1:-1] - xin[:-1])/(xin[1:] - xin[:-1])*phi[1:]
    
    # left extrapolation
    if phi[0] == 0:
        phi_b[0] = 0
    else:
        phi_b[0] = (xin[1] - xx[0])/(xin[1] - xin[0])*phi[0] + \
            (xx[0] - xin[0])/(xin[1] - xin[0])*phi[1]

    # right extrapolation
    if phi[-1] == 0:
        phi_b[-1] = 0
    else:
        phi_b[-1] = (xin[-1] - xx[-1])/(xin[-1] - xin[-2])*phi[-2] + \
            (xx[-1] - xin[-2])/(xin[-1] - xin[-2])*phi[-1]

    # ensuring solid fraction is not negative
    np.putmask(phi_b, phi_b < 0, 0)

    return phi_b


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


def Diff_bounds(phi, xx, xin, EQN = 1, OPTION = 1):

    #========================================================================
    # Calculates thermal conductivity / solutal diffusivity at cell faces
    # given solid fraction field (phi) and spatial arrays (xx, xin). 
    # EQN = 1 (therm. cond.) or 2 (sol. diff.).
    # OPTION = 1 (realistic properties) or 2 (same in solid / liquid).
    #========================================================================

    k_b = np.zeros(np.size(xx))

    if OPTION == 1:
        if EQN == 1:
            # THERMAL EQUATION
            # liquid/solid phases have different diffusivity
            k_l, k_s = 0.58, 2.0   # liquid/solid conductivity
            v_s = k_s/k_l   # diffusivity ratio
            v_l = 1
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

            
    elif OPTION == 2:
        # liquid/solid phases have same diffusivity
        v_l = v_s = 1.0

    phi_b = SFrac_bounds(phi, xin, xx)

    k_b[:] = v_s*phi_b[:] + v_l*(1 - phi_b[:])

    return k_b
