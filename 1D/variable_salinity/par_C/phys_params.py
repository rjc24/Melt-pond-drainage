# Physical parameters
k_l, k_s = 0.58, 2.0   # liquid/solid conductivity
c_pl, c_ps = 4200, 2100  # liquid/solid heat capacity
rho_l, rho_s = 1000, 1000 # 900   # liquid/solid density
kap_l, kap_s = k_l/(rho_l*c_pl), k_s/(rho_s*c_ps)
D_l = 6.8e-10   # salt diffusivity
Gam, C_bulk = 1.853/31.41, 1.5   # Liquidus slope, bulk salinity
mu = 1.787e-3   # dynamic viscosity
H_f = -Gam*C_bulk   # Freezing enthalpy
C_f = -H_f/Gam
T_L, T_R = 0.1, -2.0   # Channel (pond) temp, initial ice temp # 0.1
C_R = -T_R/Gam
L = 333400   # latent heat, sp. heat cap.
c_pl, c_ps = 4200, 2100  # liquid/solid heat capacity
T_lat = L/c_pl
L_h = 0.001   # horizontal length scale
a_r = L_h*0.5   # representative interface position for scale
f = 0.02   # friction factor

del_T = H_f - T_R
del_C = -del_T/Gam

# dimensionless parameters
St = L/(c_pl*del_T)
Cr = -H_f/del_T
c_p = c_ps/c_pl
Pr = mu/(rho_l*kap_l)
k = k_s/k_l   # thermal conductivity ratio
D = D_l/kap_l   # inverse Lewis number
D_s = 0
