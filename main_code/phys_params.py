# Physical parameters
# dimensional quantities
c_pl, c_ps = 4200, 2100  # liquid/solid heat capacity
k_l, k_s = 0.58, 2.0   # liquid/solid conductivity
rho_l, rho_s = 1000, 1000 # 900   # liquid/solid density
kap_l, kap_s = k_l/(rho_l*c_pl), k_s/(rho_s*c_ps)   # thermal diffusivities
D_l = 6.8e-10   # salt diffusivity
Gam, C_f = 1.853/31.41, 1.5   # Liquidus slope, initial b. salinity at r = a
H_f = -Gam*C_f   # Freezing enthalpy
T_L, T_R = 0.1, -2.0   # Channel (pond) temp, initial ice temp # 0.1
T_p = 0.1   # pond temperature
C_l_R = -T_R/Gam   # Channel, initial ice (liquid) concentrations
C_L, C_R = 0.5, 4.0
C_T, C_B = 1.5, 35.0
L = 333400   # latent heat
T_lat = L/c_pl
Pi_0 = 2e-8   # reference permeability
d = 1e-3   # Hele-shaw gap width
h = 0.002   # hydraulic head

# differences
del_T = H_f - T_R   # temperature
del_C = C_f - C_l_R   # concentration

# dimensionless quantities
c_p = c_ps/c_pl
St = L/(c_pl*del_T)
Cr = -C_f/del_C
Pe_m = 50
Da_h = 12*Pi_0/d**2
k = k_s/k_l   # thermal conductivity ratio
D = D_l/kap_l   # inverse Lewis number
D_s = 0
