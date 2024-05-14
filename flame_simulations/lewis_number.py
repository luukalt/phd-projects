# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 12:30:52 2019

@author: laaltenburg
"""

   
import cantera as ct
import numpy as np
from premixed_flame_properties import PremixedFlame

#%% Constants
R_gas = 0.082057 # L*atm/(mol*K)
R_gas_default = 8.31446261815324 # J/(K*mol)
kb = 1.380649e-23 # Boltzmann constant in J/K
N_A =  6.02214076e23 # Avogrado's number

#%% Temperature and pressure of unburned mixture
T_u = 273.15 + 20   # temperature of the unburned mixture in K
p_u = 101325 # pressure of the unburned mixture in Pa
p_u_atm = 1
#%% Main combustion reaction
# HYDROGEN: 2*H2 + O2 = 2*H2O
# METHANE: CH4 + 2*O2 = CO2 + 2*H2O

# %% Air composition by volume
percentage_N2 = 78
percentage_O2 = 21
percentage_Ar = 1

#%% Molecular weights of fuel-air mixture components
W_N2 = 2*14.0067 # molecular weight of nitrogen in g/mol
W_O2 = 2*15.999 # molecular weight of oxygen in g/mol
W_Ar = 39.948 # molecular weight of argon in g/mol
W_H2 = 2*1.00784 # molecular weight of hydrogen in g/mol
W_AIR = (percentage_N2/100)*W_N2 + (percentage_O2/100)*W_O2 + (percentage_Ar/100)*W_Ar # molecular weight of AIR in g/mol

#%% [STOCHIOMETRIC] Number of moles of fuel-air mixture components
# IMPORTANT NOTE
# For gasses: If reactants and products are at the same TEMPERATURE AND PRESSURE, the MOLAR RATIO is equal to the VOLUME RATIO
# This follows from the Ideal Gas Law

# Number of moles of the components(N2,O2,Ar) of air with respect to 1 mole of O2
n_O2 = 1
n_N2 = percentage_N2/percentage_O2
n_Ar = percentage_Ar/percentage_O2

# Number of moles of H2 for stoichiometric conditions
n_H2_stoich = 2

# Mole ratio of H2 and AIR at stoichiometric conditions 
n_H2_n_AIR_stoich = n_H2_stoich/(n_N2 + n_O2 + n_Ar)

# Mass of H2 and AIR at stoichiometric conditions
m_H2_stoich = n_H2_stoich*W_H2
m_AIR_stoich = n_N2*W_N2 + n_O2*W_O2 + n_Ar*W_Ar 

# Number of total moles at stoichiometric conditions
n_tot = n_O2 + n_N2 + n_Ar + n_H2_stoich

# Volume determined by Ideal Gas Law
V_ideal = R_gas*T_u

# Density of the fuel-air mixture 
rho_stoich =  ((n_O2*W_O2) + (n_N2*W_N2) + (n_Ar*W_Ar) + (n_H2_stoich*W_H2))/(V_ideal*n_tot)

#%% [NON-STOCHIOMETRIC] Number of moles of fuel-air mixture components

# Define volume fractions of species in air [units: -]
# f_O2 = 0.21
# f_N2 = 0.78
# f_AR = 0.01

f_O2 = 0.21
f_N2 = 0.79
f_AR = 0

# Define composition of DNG in volume percentages
# REST_in_DNG_percentage = 0.6
# CH4_percentage = 81.3*(1 + REST_in_DNG_percentage/100) # Volume percentage of CH4 in DNG
# C2H6_percentage = 3.7*(1 + REST_in_DNG_percentage/100) # Volume percentage of C2H6 in DNG
# N2_in_DNG_percentage = 14.4*(1 + REST_in_DNG_percentage/100) # Volume percentage of N2 in DNG

REST_in_DNG_percentage = 0
CH4_percentage = 100 
C2H6_percentage = 0
N2_in_DNG_percentage = 0


# Equivalence ratio
phi = 1

# Hydrogen volume percentage of fuel
H2_percentage = 0

# DNG percentage of fuel
DNG_percentage = 100 - H2_percentage
                          
# Volume fractions of fuel
f_H2 = H2_percentage/100
f_DNG = DNG_percentage/100
f_CH4 = f_DNG*(CH4_percentage/100)
f_C2H6 = f_DNG*(C2H6_percentage/100)
f_N2_in_DNG = f_DNG*(N2_in_DNG_percentage/100)

# Create standard gas mixture
gas = ct.Solution('gri30.yaml')

# # Define the molar ratios of the fuel-air mixture with respect to oxygen (this means that n_O2=1)
# mix = fuel + ':' + str(phi*n_H2_stoich) + ', O2:1, N2' + ':' + str(n_N2) + ',Ar:' + str(n_Ar)

# 2. Define the fuel and air composition by molar composition [= volume]
fuel = {'H2':f_H2, 'CH4':f_CH4, 'C2H6':f_C2H6, 'N2':f_N2_in_DNG}
air = {'N2':f_N2/f_O2, 'O2':1.0, 'AR':f_AR/f_O2}   

# 3. Set the equivalence ratio
gas.set_equivalence_ratio(phi, fuel, air)


# The transport model 
gas.transport_model = 'Multi'
gas.TP = T_u, p_u
rho_H2 = gas.TD[1]*gas[fuel].Y[0]*(n_tot/n_H2_stoich) # density of hydrogen [kg*m^-3]
rho_u = gas.TD[1]

lambda_u = gas.thermal_conductivity
cp_u = gas.cp_mass

Dij_mixavg = gas.mix_diff_coeffs_mass
Dij_binary = gas.binary_diff_coeffs

index_of_H2 = gas.species_index('H2')
index_of_CH4 = gas.species_index('CH4')

index_of_O2 = gas.species_index('O2')
index_of_N2 = gas.species_index('N2')

phi_limit = 2

if H2_percentage == 0:
    index_of_fuel = index_of_CH4
elif H2_percentage == 100:
    index_of_fuel = index_of_H2
    

if phi < phi_limit:
    DiN2 = Dij_binary[index_of_fuel, index_of_N2] # H2 into N2
    Dij_mixavg = Dij_mixavg[index_of_fuel]
    mass_fraction_of_i = gas['H2'].Y
    Dij_test = (1 - mass_fraction_of_i)/(gas['O2'].X/Dij_binary[index_of_fuel, index_of_O2] + gas['N2'].X/Dij_binary[index_of_fuel, index_of_N2])
    Dij_test = Dij_test[0]
    
    sigma_A = 2.920 # collision diameter in angstrom = 10^-10 m
    sigma_B = 3.621 # collision diameter in angstrom = 10^-10 m
    sigma_AB = (sigma_A + sigma_B)/2 # averageed collision diameter in angstrom = 10^-10 m
    M_A = 2 # molecular weight of species A
    M_B = 28 # molecular weight of species B
    
    eps_A_over_kb =  38
    eps_B_over_kb = 97.53
    eps_AB = np.sqrt(eps_A_over_kb*eps_B_over_kb)
    T_star =  T_u/eps_AB
    sigma_AB = (sigma_A + sigma_B)/2
    Omega_star11 = 0.86
    DiN2_law = 1.8583e-7*np.sqrt(T_u**3*((1/M_A)+(1/M_B)))/(p_u_atm*Omega_star11*1*sigma_AB**2)

elif phi >= phi_limit:
    DiN2 = Dij_binary[index_of_O2, index_of_N2] # O2 into N2
    Dij_mixavg = Dij_mixavg[index_of_O2]
    mass_fraction_of_i = gas['O2'].X
    Dij_test = (1 - mass_fraction_of_i)/(gas['H2'].X/Dij_binary[index_of_O2, index_of_fuel] + gas['N2'].X/Dij_binary[index_of_O2, index_of_N2])
    
    sigma_A = 3.458 # collision diameter in angstrom = 10^-10 m
    sigma_B = 3.621 # collision diameter in angstrom = 10^-10 m
    sigma_AB = (sigma_A + sigma_B)/2 # averageed collision diameter in angstrom = 10^-10 m
    M_A = 32 # molecular weight of species A
    M_B = 28 # molecular weight of species B

    eps_A_over_kb =  107.4
    eps_B_over_kb = 97.53
    eps_AB = np.sqrt(eps_A_over_kb*eps_B_over_kb)
    T_star =  T_u/eps_AB
    sigma_AB = (sigma_A + sigma_B)/2
    Omega_star11 = 1
    DiN2_law = 1.8583e-7*np.sqrt(T_u**3*((1/M_A)+(1/M_B)))/(p_u_atm*Omega_star11*1*sigma_AB**2)


# Assuming gas is a predefined object with properties like density and viscosity
rho = gas.density_mass  # Gas density in kg/m^3
mu = gas.viscosity      # Gas dynamic viscosity in Pa.s (Pascal-seconds)
nu = mu / rho           # Kinematic viscosity in m^2/s


alpha = lambda_u / (rho_u*cp_u)
lewis_binary = alpha / DiN2
lewis_mixavg = alpha / Dij_mixavg
lewis_test = alpha / Dij_test

Sc_binary = nu / DiN2
Sc_mixavg = nu / Dij_mixavg

S_L0 = 0.365
flame = PremixedFlame(phi, H2_percentage)
flame.solve_equations()

# Compute flame thickness
z = flame.flame.grid
T = flame.flame.T
size = len(z)-1
grad = np.zeros(size)

for i in range(size):
    grad[i] = (T[i+1]-T[i])/(z[i+1]-z[i])
    
thickness_thermal = (max(T) - min(T)) / max(grad)
print('laminar flame thickness [thermal thickness] = ', thickness_thermal*1e3, str(" mm"))

diffusivity = flame.lambda_u / (flame.cp_u*flame.rho_u)

delta_f = diffusivity/flame.S_L0



# Printing with formatting to three decimal places
decimals = 2
print(f"Temperature (T): {T_u:.{decimals}f} K")
print(f"Pressure (p): {p_u:.{decimals}f} Pa")
print(f"Gas Density (ρ): {rho:.{decimals}f} kg/m³")
print(f"Gas Dynamic Viscosity (μ): {mu:.{decimals}e} Pa.s")
print(f"Kinematic Viscosity (ν): {nu:.{decimals}e} m²/s")

print(f"Thermal Diffusivity (α): {alpha:.{decimals}e} m²/s")
print(f"Mass Diffusivity [Binary] (D): {DiN2:.{decimals}e} m²/s")
print(f"Mass Diffusivity [Mixture-Averaged] (D): {Dij_mixavg:.{decimals}e} m²/s")
# print(f"Mass Diffusivity [Dij_test] (D): {Dij_test:.{decimals}e} m²/s")
print(f"Mass Diffusivity [DiN2_law] (D): {DiN2_law:.{decimals}e} m²/s")

print(f"Binary Lewis Number: {lewis_binary:.{decimals}f}")
print(f"Mixture-Averaged Lewis Number: {lewis_mixavg:.{decimals}f}")

print(f"Binary Schmidt Number: {Sc_binary:.{decimals}f}")
print(f"Mixture-Averaged Schmidt Number: {Sc_binary:.{decimals}f}")

print(f"Thermal Diffusivity [diffusivity] (D): {diffusivity:.{decimals}e} m²/s")

print(f"Flame Thickness (δ_f): {delta_f*1e3:.3e} mm") 


# #%% Dimensions of the scaled Flamesheet
# H_inlet = 4.9e-3                                                           # height of channel after injector [m]
# D_outer = 83.8e-3                                                          # outer diameter of the Flamesheet burner [m]   
# R_outer = D_outer/2                                                     # outer radius of the Flamesheet burner [m]
# D_inner = D_outer - 2*H_inlet  #diameter_out - 2*height_inlet                             # inner diameter (after fuel injection) [m]
# R_inner = D_inner/2                                                       # inner radius (after fuel injection) [m]
# A_inlet = (R_outer**2 - R_inner**2)*math.pi

# #%% Flow conditions of the scaled Flamesheet
# u_inlet = 70
# Q_inlet = A_inlet*u_inlet
# n_H2_n_AIR = phi*n_H2_n_AIR_stoich # fraction of n_H2 / n_AIR, which depends on the equivalence ratio
# Q_H2 = n_H2_n_AIR*Q_inlet
# m_dot_H2 = rho_H2*Q_H2
# LHV_H2 = 120e6 # Lower Heating Value of hydrogen in J/kg
# P_thermal = m_dot_H2*LHV_H2/1e3 # thermal power after fuel injection [kW]
     

























