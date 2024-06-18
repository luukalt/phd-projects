# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 12:30:52 2019

@author: laaltenburg
"""

import pickle
import cantera as ct
import numpy as np
from premixed_flame_properties import PremixedFlame
from matplotlib import pyplot as plt
import matplotlib.cm as cm

#%% FUNCTIONS
def plot_quantity_vs_phi(filename, n=6):
    
    linestyle = 'None'
    
    markers = ['o', 'o', 'o', 'o', 'o', 'o']
    # markers = ['o', '^', 's', 'p', 'X', '*']
    # markers = ['s', 's', 's', 's', 's', 's']
    
    markersize = 8
    
    colors = cm.viridis(np.linspace(0, 1, len(markers)))
    labels = ['$0$', '$20$', '$40$', '$60$', '$80$', '$100$']
    
    with open(filename + '.txt', 'rb') as f:
        Lewis_lib = pickle.load(f)
    
    # return Lewis_lib

    phi_lists = [[] for i in range(n)]
    H2_percentage_lists = [[] for i in range(n)]
    S_L0_lists = [[] for i in range(n)]
    Le_binary_V_lists = [[] for i in range(n)]
    DiN2_DjN2_V_lists = [[] for i in range(n)]
    
    fig, ax = plt.subplots()
    ax.set_xlabel('$\phi$')
    ax.set_ylabel('Lewis nr')  
    ax.set_xlim(0.35, 1.05)
    # ax.set_ylim(0, 2.5)
    ax.grid()
    
    check_label = ''
    
    for mixture in Lewis_lib:
        
        index = mixture['index']
        phi = mixture['phi']
        H2_percentage = mixture['H2%']
        S_L0 = mixture['S_L0']
        Le_binary_V = mixture['Le_binary_V']
        DiN2_DjN2_V = mixture['DiN2_DjN2_V']
        
        phi_lists[index] = np.append(phi_lists[index], phi)
        H2_percentage_lists[index] = np.append(H2_percentage_lists[index], H2_percentage)
        S_L0_lists[index] = np.append(S_L0_lists[index], S_L0)
        Le_binary_V_lists[index] = np.append(Le_binary_V_lists[index], Le_binary_V)
        DiN2_DjN2_V_lists[index] = np.append(DiN2_DjN2_V_lists[index], DiN2_DjN2_V)
        
        c1 = (5/3)*phi + (5/3)
        c2 = (5/3)*phi - (2/3)
        
        c1 = 1
        c2 = 1
        
        ax.plot(phi, 1/(DiN2_DjN2_V**c2/Le_binary_V**c1), ls=linestyle, marker=markers[index], ms=markersize, mec="k", mfc=colors[index], label=labels[index] if index != check_label else "", zorder=10)
        # ax.plot(phi, Le_binary_V, ls=linestyle, marker=markers[index], ms=markersize, mec="k", mfc=colors[index], label=labels[index] if index != check_label else "", zorder=10)
        
        check_label = index

def create_lib():
    #%% Constants
    R_gas = 0.082057 # L*atm/(mol*K)
    R_gas_default = 8.31446261815324 # J/(K*mol)
    kb = 1.380649e-23 # Boltzmann constant in J/K
    N_A =  6.02214076e23 # Avogrado's number
    decimals = 2
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
    f_O2 = 0.21
    f_N2 = 0.78
    f_AR = 0.01
    
    # f_O2 = 0.21
    # f_N2 = 0.79
    # f_AR = 0
    
    # Define composition of DNG in volume percentages
    REST_in_DNG_percentage = 0.6
    CH4_percentage = 81.3*(1 + REST_in_DNG_percentage/100) # Volume percentage of CH4 in DNG
    C2H6_percentage = 3.7*(1 + REST_in_DNG_percentage/100) # Volume percentage of C2H6 in DNG
    N2_in_DNG_percentage = 14.4*(1 + REST_in_DNG_percentage/100) # Volume percentage of N2 in DNG
    
    # REST_in_DNG_percentage = 0
    # CH4_percentage = 100 
    # C2H6_percentage = 0
    # N2_in_DNG_percentage = 0
    
    mixtures = {}
    
    mixtures[(0, 0, 0.80)] = []
    mixtures[(0, 0, 0.90)] = []
    mixtures[(0, 0, 1.00)] = []
    
    mixtures[(1, 20, 0.80)] = []
    mixtures[(1, 20, 0.90)] = []
    mixtures[(1, 20, 1.00)] = []
    
    mixtures[(2, 40, 0.70)] = []
    mixtures[(2, 40, 0.80)] = []
    mixtures[(2, 40, 0.90)] = []
    mixtures[(2, 40, 1.00)] = []
    
    mixtures[(3, 60, 0.60)] = []
    mixtures[(3, 60, 0.70)] = []
    mixtures[(3, 60, 0.80)] = []
    mixtures[(3, 60, 0.90)] = []
    mixtures[(3, 60, 1.00)] = []
    
    mixtures[(4, 80, 0.50)] = []
    mixtures[(4, 80, 0.60)] = []
    mixtures[(4, 80, 0.70)] = []
    mixtures[(4, 80, 0.80)] = []
    mixtures[(4, 80, 0.90)] = []
    mixtures[(4, 80, 1.00)] = []
    
    mixtures[(5, 100, 0.30)] = []
    mixtures[(5, 100, 0.40)] = []
    mixtures[(5, 100, 0.49)] = []
    mixtures[(5, 100, 0.50)] = []
    mixtures[(5, 100, 0.60)] = []
    mixtures[(5, 100, 0.70)] = []
    mixtures[(5, 100, 0.80)] = []
    mixtures[(5, 100, 0.90)] = []
    mixtures[(5, 100, 1.00)] = []
    
    Lewis_lib = []
    
    for key,values in mixtures.items():
        
        index = key[0]
        H2_percentage = key[1]
        phi = key[2]
        
        # Equivalence ratio
        # phi = 1
        
        # Hydrogen volume percentage of fuel
        # H2_percentage = 50
        
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
        
        # 2. Define the fuel and air composition by molar composition [= volume]
        fuel = {'H2':f_H2, 'CH4':f_CH4, 'C2H6':f_C2H6, 'N2':f_N2_in_DNG}
        air = {'N2':f_N2/f_O2, 'O2':1.0, 'AR':f_AR/f_O2}   
        
        # 3. Set the equivalence ratio
        gas.set_equivalence_ratio(phi, fuel, air)
        
        # The transport model 
        gas.transport_model = 'multicomponent'
        gas.TP = T_u, p_u
        rho_H2 = gas.TD[1]*gas[fuel].Y[0]*(n_tot/n_H2_stoich) # density of hydrogen [kg*m^-3]
        
        rho_u = gas.TD[1]
        lambda_u = gas.thermal_conductivity
        cp_u = gas.cp_mass
        alpha_u = lambda_u / (rho_u*cp_u)
        
        # Indices of species in the gas mixture object
        index_of_H2 = gas.species_index('H2')
        index_of_CH4 = gas.species_index('CH4')
        index_of_C2H6 = gas.species_index('C2H6')
        index_of_O2 = gas.species_index('O2')
        index_of_N2 = gas.species_index('N2')
        indices_of_fuel = [index_of_H2, index_of_CH4, index_of_C2H6]
        
        # 'f' indicates volumetric fraction
        f_fuel = [f_H2, f_CH4, f_C2H6]
        sum_f = sum(f_fuel)
        f_fuel = [f/sum_f for f in f_fuel]
        
        # Mass diffusivity matrix
        Dij_mixavg = gas.mix_diff_coeffs_mass
        Dij_binary = gas.binary_diff_coeffs
        
        phi_limit = 2
        
        Le_binary_list = []
        Le_mixavg_list = []
        DiN2_DjN2_list = []
        
        if phi < phi_limit:
            for index_of_fuel in indices_of_fuel:
        
                DiN2 = Dij_binary[index_of_fuel, index_of_N2] # H2 into N2
                DjN2 = Dij_binary[index_of_O2, index_of_N2]
                DiN2_DjN2 = DiN2/DjN2
                
                Dij_imixavg = Dij_mixavg[index_of_fuel]
                
                print(f"[Binary] Mass Diffusivity (DiN2) of {gas.species_name(index_of_fuel)}: {DiN2:.{decimals}e} m²/s")
                print(f"[Binary] Mass Diffusivity (DjN2) of {gas.species_name(index_of_O2)}: {DjN2:.{decimals}e} m²/s")
                print(f"Preferential diffusion DiN2/DjN2 : {DiN2_DjN2:.{decimals}f}")
        
                print(f"[Mixture-Averaged] Mass Diffusivity (D_imixavg) of {gas.species_name(index_of_fuel)}: {Dij_imixavg:.{decimals}e} m²/s")
                
                print(32*'-')
                Le_binary_list.append(alpha_u / DiN2)
                Le_mixavg_list.append(alpha_u / Dij_imixavg)
                DiN2_DjN2_list.append(DiN2_DjN2)
            
            # Lewis numbers Volume-based (V)
            Le_binary_V_list = [Le_i * f for Le_i, f in zip(Le_binary_list, f_fuel)]
            Le_binary_V = sum(Le_binary_V_list)
            
            Le_mixavg_V_list = [Le_i * f for Le_i, f in zip(Le_mixavg_list, f_fuel)]
            Le_mixavg_V = sum(Le_mixavg_V_list)
            
            # Preferential diffusion numbers Volume-based (V)
            DiN2_DjN2_V_list = [DiN2_DjN2 * f for DiN2_DjN2, f in zip(DiN2_DjN2_list, f_fuel)]
            DiN2_DjN2_V = sum(DiN2_DjN2_V_list)
            
            
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
        
        # elif phi >= phi_limit:
        #     DiN2 = Dij_binary[index_of_O2, index_of_N2] # O2 into N2
        #     Dij_mixavg = Dij_mixavg[index_of_O2]
        #     mass_fraction_of_i = gas['O2'].X
        #     Dij_test = (1 - mass_fraction_of_i)/(gas['H2'].X/Dij_binary[index_of_O2, index_of_fuel] + gas['N2'].X/Dij_binary[index_of_O2, index_of_N2])
            
        #     sigma_A = 3.458 # collision diameter in angstrom = 10^-10 m
        #     sigma_B = 3.621 # collision diameter in angstrom = 10^-10 m
        #     sigma_AB = (sigma_A + sigma_B)/2 # averageed collision diameter in angstrom = 10^-10 m
        #     M_A = 32 # molecular weight of species A
        #     M_B = 28 # molecular weight of species B
        
        #     eps_A_over_kb =  107.4
        #     eps_B_over_kb = 97.53
        #     eps_AB = np.sqrt(eps_A_over_kb*eps_B_over_kb)
        #     T_star =  T_u/eps_AB
        #     sigma_AB = (sigma_A + sigma_B)/2
        #     Omega_star11 = 1
        #     DiN2_law = 1.8583e-7*np.sqrt(T_u**3*((1/M_A)+(1/M_B)))/(p_u_atm*Omega_star11*1*sigma_AB**2)
        
        
        # Assuming gas is a predefined object with properties like density and viscosity
        rho = gas.density_mass  # Gas density in kg/m^3
        mu = gas.viscosity      # Gas dynamic viscosity in Pa.s (Pascal-seconds)
        nu = mu / rho           # Kinematic viscosity in m^2/s
        
        # alpha_u = lambda_u / (rho_u*cp_u)
        # lewis_binary = alpha_u / DiN2
        # lewis_mixavg = alpha_u / Dij_mixavg
        # lewis_test = alpha_u / Dij_test
        
        # Sc_binary = nu / DiN2
        # Sc_mixavg = nu / Dij_mixavg
        
        # S_L0 = 0.365
        flame = PremixedFlame(phi, H2_percentage)
        flame.solve_equations()
        
        # # Compute flame thickness
        # z = flame.flame.grid
        # T = flame.flame.T
        # size = len(z)-1
        # grad = np.zeros(size)
        
        # for i in range(size):
        #     grad[i] = (T[i+1]-T[i])/(z[i+1]-z[i])
            
        # thickness_thermal = (max(T) - min(T)) / max(grad)
        # print('laminar flame thickness [thermal thickness] = ', thickness_thermal*1e3, str(" mm"))
        
        # diffusivity = flame.lambda_u / (flame.cp_u*flame.rho_u)
        
        # delta_f = diffusivity/flame.S_L0
        
        lib_item = {'index': index, 'phi':flame.phi, 'H2%':flame.H2_percentage, 'T_u':flame.T_u, 'p_u':flame.p_u,
                    'S_L0':flame.S_L0, 'T_ad':flame.T_ad, 'f_air':flame.air, 'f_fuel':flame.fuel, 'mass_fractions_u':flame.mass_fractions_u,
                    'LHV_mixture':flame.LHV_mixture,'LHV_fuel':flame.LHV_fuel, 'Le_binary_V':Le_binary_V, 'DiN2_DjN2_V':DiN2_DjN2_V}
        
        # Append item to strained_flame_speed_lib
        Lewis_lib.append(lib_item)
        
        # Write strained_flame_speed_lib to file 
        filename = 'Lewis_lib.txt'
        with open(filename, 'wb') as f:
            pickle.dump(Lewis_lib, f) 
            
        print(f'flame calculated: phi={round(phi, 2)}, H2%={H2_percentage}')

#%% MAIN

if __name__ == "__main__":
    filename = 'Lewis_lib'
    plot_quantity_vs_phi(filename)
                     
    # # Printing with formatting to three decimal places
    # print(32*'-')
    # print(f"Temperature (T): {T_u:.{decimals}f} K")
    # print(f"Pressure (p): {p_u:.{decimals}f} Pa")
    # print(f"Gas Density (ρ): {rho:.{decimals}f} kg/m³")
    # print(f"Gas Dynamic Viscosity (μ): {mu:.{decimals}e} Pa.s")
    # print(f"Kinematic Viscosity (ν): {nu:.{decimals}e} m²/s")
    
    # print(32*'-')
    # print(f"Unstretched laminar flame speed (S_L0): {flame.S_L0:.{decimals}e} m²/s")
    
    # print(32*'-')
    # print(f"Thermal Diffusivity (α): {alpha_u:.{decimals}e} m²/s")
    
    # print(32*'-')
    # print('BINARY')
    # print(f"Binary Lewis Number: {Le_binary_V:.{decimals}f}")
    
    
    # print(32*'-')
    # print('MIXTURE-AVERAGED')
    # print(f"Mixture-Averaged Lewis Number: {Le_mixavg_V:.{decimals}f}")
    
    # print(32*'-')
    # print('PREFERENTIAL DIFFUSION')
    # print(f"Volume weighted Preferential diffusion (DiN2/DjN2)_V : {DiN2_DjN2_V:.{decimals}f}")
    
    # print(f"Binary Schmidt Number: {Sc_binary:.{decimals}f}")
    # print(f"Mixture-Averaged Schmidt Number: {Sc_binary:.{decimals}f}")
    
    
    
    # print(f"Mass Diffusivity [Dij_test] (D): {Dij_test:.{decimals}e} m²/s")
    # print(f"Mass Diffusivity [DiN2_law] (D): {DiN2_law:.{decimals}e} m²/s")
    
    
    # print(f"Thermal Diffusivity [diffusivity] (α): {diffusivity:.{decimals}e} m²/s")
    
    # print(f"Flame Thickness (δ_f): {delta_f*1e3:.3e} mm") 
    
    
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
         

























