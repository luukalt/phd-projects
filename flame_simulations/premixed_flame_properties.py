# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 11:23:39 2022

@author: luuka

premixed flame properties
"""
#%% IMPORT STANDARD PACKAGES
import os
import sys
import pickle
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import cantera as ct

#%% IMPORT USER DEFINED PACKAGES
import sys_paths
import rc_params_settings
from plot_params import fontsize, fontsize_legend

figures_folder = 'figures'
if not os.path.exists(figures_folder):
        os.makedirs(figures_folder)
        
#%% FUNCTIONS

def create_library(filename):
    
    # Create new file
    file_name = "unstreched_laminar_flame_speed_data/" + filename + ".txt"
    if os.path.isfile(file_name):
        expand = 0
        while True:
            expand += 1
            new_file_name = file_name.split(".txt")[0] + str(expand) + ".txt"
            if os.path.isfile(new_file_name):
                continue
            else:
                file_name = new_file_name
                break
        
    S_L0_lib = []
    
    for key,values in mixtures.items():
        
        index = key[0]
        H2_percentage = key[1]
        phi = key[2]
        
        flame = PremixedFlame(phi, H2_percentage)
        flame.solve_equations()
        print(flame.rho_u)
        print(flame.rho_b)
        print(flame.rho_u/flame.rho_b)
        print(flame.T_ad/flame.T_u)
    
        lib_item = {'index': index, 'phi':flame.phi, 'H2%':flame.H2_percentage, 'T_u':flame.T_u, 'p_u':flame.p_u,
                    'S_L0':flame.S_L0, 'T_ad':flame.T_ad, 'f_air':flame.air, 'f_fuel':flame.fuel, 'mass_fractions_u':flame.mass_fractions_u,
                    'LHV_mixture':flame.LHV_mixture,'LHV_fuel':flame.LHV_fuel}
        
        # Append item to strained_flame_speed_lib
        S_L0_lib.append(lib_item)
        
        # Write strained_flame_speed_lib to file 
        with open(file_name, 'wb') as f:
            pickle.dump(S_L0_lib, f) 
            
        print(f'flame calculated: phi={round(phi, 2)}, H2%={H2_percentage}')
        
    return S_L0_lib
    

def heating_value(fuel, T_u=293.15, p_u=ct.one_atm):
    
    """ Returns the LHV and HHV for the specified fuel """
    gas = ct.Solution('gri30.yaml')

    gas.TP = T_u, p_u
    gas.set_equivalence_ratio(1.0, fuel, 'O2:1.0')
    h1 = gas.enthalpy_mass
    Y_fuel = gas[fuel].Y[0]
    
    
    water = ct.Water()
    # Set liquid water state, with vapor fraction x = 0
    water.TQ = T_u, 0
    h_liquid = water.h
    # Set gaseous water state, with vapor fraction x = 1
    water.TQ = T_u, 1
    h_gas = water.h

    # Complete combustion products
    Y_products = {'CO2': gas.elemental_mole_fraction('C'),
                  'H2O': 0.5 * gas.elemental_mole_fraction('H'),
                  'N2': 0.5 * gas.elemental_mole_fraction('N')}

    gas.TPX = None, None, Y_products
    Y_H2O = gas['H2O'].Y[0]
    h2 = gas.enthalpy_mass
    LHV = -(h2-h1)/Y_fuel/1e6
    HHV = -(h2-h1 + (h_liquid-h_gas) * Y_H2O)/Y_fuel/1e6
    return LHV, HHV

#%% AUXILIARY FUNCTIONS

def plot_phi_vs_S_L0(filename, n=6):
    
    linestyle = 'None'
    
    markers = ['o', 'o', 'o', 'o', 'o', 'o']
    # markers = ['o', '^', 's', 'p', 'X', '*']
    # markers = ['s', 's', 's', 's', 's', 's']
    
    markersize = 8
    
    colors = cm.viridis(np.linspace(0, 1, len(markers)))
    labels = ['$0$', '$20$', '$40$', '$60$', '$80$', '$100$']
    
    with open(os.path.join('unstreched_laminar_flame_speed_data', filename + '.txt'), 'rb') as f:
        S_L0_lib = pickle.load(f)
    
    phi_lists = [[] for i in range(n)]
    H2_percentage_lists = [[] for i in range(n)]
    S_L0_lists = [[] for i in range(n)]
    
    default_fig_dim = plt.rcParams["figure.figsize"]
    fig_size = default_fig_dim[0]

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    ax.set_xlabel('$\phi$', fontsize=fontsize)
    ax.set_ylabel('$S_{L0}$ [ms$^{-1}$]', fontsize=fontsize)  
    ax.set_xlim(0.35, 1.05)
    ax.set_ylim(0, 2.5)
    ax.grid()
    
    check_label = ''
    
    for mixture in S_L0_lib:
        
        index = mixture['index']
        phi = mixture['phi']
        H2_percentage = mixture['H2%']
        S_L0 = mixture['S_L0']
        
        # print(phi, H2_percentage, S_L0)
        
        phi_lists[index] = np.append(phi_lists[index], phi)
        H2_percentage_lists[index] = np.append(H2_percentage_lists[index], H2_percentage)
        S_L0_lists[index] = np.append(S_L0_lists[index], S_L0)
        ax.plot(phi, S_L0, ls=linestyle, marker=markers[index], ms=markersize, mec="k", mfc=colors[index], label=labels[index] if index != check_label else "", zorder=10)
        
        check_label = index
    
    S_L0_DNG = S_L0_lib[2]["S_L0"] 
    ax.plot(1, S_L0_DNG, 'rx', ms=10, mew=2, zorder=10)
    
    zipped = zip(phi_lists, H2_percentage_lists, S_L0_lists, colors)
    
    poly_order = 3
    
    print ("{:<8} {:<8} {:<8}".format('phi', 'H2%', 'S_L0 [m/s]'))
    
    print ("{:<8.2f} {:<8} {:<8.3f}".format(1, 0, S_L0_DNG))
    
    for (phi, H2_percentage, S_L0, color) in zipped:
    
        phi_fit = np.linspace(min(phi), max(phi), 10000)
    
        poly_S_L0 = np.poly1d(np.polyfit(phi, S_L0, poly_order))
        S_L0_fit = poly_S_L0(phi_fit)
        ax.plot(phi_fit, S_L0_fit, ls="--", c=color)
        
        idx = np.argwhere(np.diff(np.sign(S_L0_fit - S_L0_DNG))).flatten()
        
        if H2_percentage[0] == 100:
            
            ax.plot(phi_fit[idx], S_L0_fit[idx], 'rx', ms=10, mew=2, zorder=10)
        
        if H2_percentage[0] > 0:
            
            # print(phi_fit[idx])
            print("{:<8.3f} {:<8} {:<8.3f}".format(phi_fit[idx][0], int(H2_percentage[0]), S_L0_fit[idx][0]))
    
    
    ax.legend(title="$H_2\%$", loc="upper left", bbox_to_anchor=(0, 1), ncol=1, prop={"size": fontsize_legend})
    
    # Set aspect ratio to 'auto' to avoid stretching
    ax.set_aspect('auto', adjustable='box')
    
    # Set the size of the axis to ensure both axes have the same dimensions in inches
    ax.set_position([0.1, 0.1, 0.8, 0.8])  # Set the position of the axis in the figure (left, bottom, width, height)

    return S_L0_lib

def plot_phi_vs_T_ad(filename, n=6):
    
    linestyle = 'None'
    
    markers = ['o', 'o', 'o', 'o', 'o', 'o']
    # markers = ['o', '^', 's', 'p', 'X', '*']
    # markers = ['s', 's', 's', 's', 's', 's']
    
    markersize = 8
    
    colors = cm.viridis(np.linspace(0, 1, len(markers)))
    labels = ['$0$', '$20$', '$40$', '$60$', '$80$', '$100$']
    
    with open('unstreched_laminar_flame_speed_data/'+ filename + '.txt', 'rb') as f:
        S_L0_lib = pickle.load(f)
    
    phi_lists = [[] for i in range(n)]
    H2_percentage_lists = [[] for i in range(n)]
    T_ad_lists = [[] for i in range(n)]
    
    default_fig_dim = plt.rcParams["figure.figsize"]
    fig_size = default_fig_dim[0]
    
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    ax.set_xlabel('$\phi$')
    ax.set_ylabel('$T_{ad}$ [K]')
    # ax.set_xlim(0.3, 1.1)
    # ax.set_ylim(0, 2.5)
    ax.grid()
    
    check_label = ''
    
    for mixture in S_L0_lib:
        
        index = mixture['index']
        phi = mixture['phi']
        H2_percentage = mixture['H2%']
        T_ad = mixture['T_ad']
        
        phi_lists[index] = np.append(phi_lists[index], phi)
        H2_percentage_lists[index] = np.append(H2_percentage_lists[index], H2_percentage)
        T_ad_lists[index] = np.append(T_ad_lists[index], T_ad)
        ax.plot(phi, T_ad, ls=linestyle, marker=markers[index], ms=markersize, mec="k", mfc=colors[index], label=labels[index] if index != check_label else "", zorder=10)
        
        check_label = index
        
        print(f'phi={phi}, H2%={H2_percentage}, T_ad={T_ad} [K], T_ad/T_u={T_ad/293.15}')
    
    T_ad_DNG = S_L0_lib[2]["T_ad"] 
    ax.plot(1, T_ad_DNG, 'rx', ms=8, mew=2, zorder=10)
    
    zipped = zip(phi_lists, H2_percentage_lists, T_ad_lists, colors)
    
    poly_order = 3
    
    print ("{:<8} {:<8} {:<8}".format('phi', 'H2%', 'T_ad [K]'))
    
    print ("{:<8.2f} {:<8} {:<8.3f}".format(1, 0, T_ad_DNG))
    
    for (phi, H2_percentage, T_ad, color) in zipped:
    
        phi_fit = np.linspace(min(phi), max(phi), 10000)
    
        poly_T_ad = np.poly1d(np.polyfit(phi, T_ad, poly_order))
        T_ad_fit = poly_T_ad(phi_fit)
        ax.plot(phi_fit, T_ad_fit, ls="--", c=color)
        
        
        idx = np.argwhere(np.diff(np.sign(T_ad_fit - T_ad_DNG))).flatten()
        ax.plot(phi_fit[idx], T_ad_fit[idx], 'rx', ms=8, mew=2, zorder=10)
        
        if H2_percentage[0] > 0:
            
            # print(phi_fit[idx])
            print("{:<8.3f} {:<8} {:<8.3f}".format(phi_fit[idx][0], int(H2_percentage[0]), T_ad_fit[idx][0]))
    
    
    # ax.legend(title="$H_2\%$", loc="upper left", prop={"size": 12})
    ax.legend(title="$H_2\%$", loc="upper left", bbox_to_anchor=(0, 1), ncol=1, prop={"size": 12})
    
    return S_L0_lib       
#%% OBJECTS
class PremixedFlame:
    
    def __init__(self, phi, H2_percentage, T_u=293.15, p_u=ct.one_atm):

        # Color and label for plots
        # self.color = colors[H2_percentages.index(H2_percentage)]
        self.label = str(int(H2_percentage)) + r'$\% H_2$' 
        
        #% Constants
        R_gas_mol = 8314 # Universal gas constant [units: J*K^-1*kmol^-1]
        R_gas_mass = 287 # universal gas constant [units: J*K^-1*kg^-1]
        
        self.T_u = T_u      # Inlet temperature in K
        self.p_u = p_u    # Inlet pressure in Pa
        
        # Molar mass of species [units: kg*kmol^-1]
        M_H = 1.008 
        M_C = 12.011
        M_N = 14.007
        M_O = 15.999
        M_H2 = M_H*2
        M_CH4 = M_C + M_H*4
        M_CO2 = M_C + M_O*4
        
        # Define volume fractions of species in air [units: -]
        f_O2 = 0.21
        f_N2 = 0.78
        f_AR = 0.01

        # f_O2 = 0.21
        # f_N2 = 0.79
        # f_AR = 0
        
        # Define composition of DNG in volume percentages
        self.REST_in_DNG_percentage = 0.6
        self.CH4_percentage = 81.3*(1 + self.REST_in_DNG_percentage/100) # Volume percentage of CH4 in DNG
        self.C2H6_percentage = 3.7*(1 + self.REST_in_DNG_percentage/100) # Volume percentage of C2H6 in DNG
        self.N2_in_DNG_percentage = 14.4*(1 + self.REST_in_DNG_percentage/100) # Volume percentage of N2 in DNG

        # self.REST_in_DNG_percentage = 0
        # self.CH4_percentage = 100 
        # self.C2H6_percentage = 0
        # self.N2_in_DNG_percentage = 0
        
        # rho_H2 = M_H2*self.p_u/(self.T_u*R_gas_mol)
        # rho_CH4 = M_CH4*self.p_u/(self.T_u*R_gas_mol)
        # rho_CO2 = M_CO2*self.p_u/(self.T_u*R_gas_mol)
        
        # Equivalence ratio
        self.phi = phi
        
        # Hydrogen volume percentage of fuel
        self.H2_percentage = H2_percentage
        
        # DNG percentage of fuel
        self.DNG_percentage = 100 - self.H2_percentage
                                  
        # Volume fractions of fuel
        f_H2 = self.H2_percentage/100
        f_DNG = self.DNG_percentage/100
        f_CH4 = f_DNG*(self.CH4_percentage/100)
        f_C2H6 = f_DNG*(self.C2H6_percentage/100)
        f_N2_in_DNG = f_DNG*(self.N2_in_DNG_percentage/100)
        
        # Check if volume fractions of fuel and air are correct
        check_air = f_O2 + f_N2 + f_AR
        check_fuel = f_H2 + f_CH4 + f_C2H6 + f_N2_in_DNG
        if check_air == 1.0 and round(check_fuel,3) == 1.0:
            pass
        else:
            sys.exit("fuel or air composition is incorrect!" + "sum of fuel volume fractions:" + str(check_fuel))
            
        if round(check_fuel,3) == 1.0:
            pass
        else:
            sys.exit("fuel composition is incorrect!")
        
        # Definition of the mixture
        # 1. Set the reaction mechanism
        self.gas = ct.Solution('gri30.yaml')
        
        # 2. Define the fuel and air composition by molar composition [= volume]
        self.fuel = {'H2':f_H2, 'CH4':f_CH4, 'C2H6':f_C2H6, 'N2':f_N2_in_DNG}
        self.air = {'N2':f_N2/f_O2, 'O2':1.0, 'AR':f_AR/f_O2}   
        
        # 3. Set the equivalence ratio
        self.gas.set_equivalence_ratio(phi, self.fuel, self.air)
        
        # Mass fractions of unburnt mixture
        self.mass_fractions_u = self.gas.mass_fraction_dict()
        
        # Total mass fraction of the fuel
        Y_f_tot = 0
        
        for fuel_key in self.fuel.keys():
            
            if self.mass_fractions_u.get(fuel_key) is not None:
                
                if fuel_key in ['H2', 'CH4', 'C2H6']:
                    
                    Y_f = self.mass_fractions_u[fuel_key]
                    
                    Y_f_tot += Y_f
                    
        # Energy content of mixture
        LHV_mixture = 0
        LHV_fuel = 0
        
        for fuel_key in self.fuel.keys():
            
            if self.mass_fractions_u.get(fuel_key) is not None:
                
                if fuel_key in ['H2', 'CH4', 'C2H6']:
                    
                    Y_f = self.mass_fractions_u[fuel_key]
                    
                    LHV_f, HHV_f = heating_value(fuel_key)
                    
                    # print(LHV_f)
                    
                    LHV_fuel += LHV_f*Y_f/Y_f_tot
                    
                    LHV_mixture += LHV_f*Y_f
                    
                    
        
        self.LHV_fuel = LHV_fuel
        self.LHV_mixture = LHV_mixture
        
        # 4. Set the transport model
        self.gas.transport_model= 'Multi'
        
        # 5. Set the unburnt mixture temperature and pressure
        self.gas.TP = self.T_u, self.p_u
        
        # Properties of the unburnt mixture 
        self.h_u = self.gas.enthalpy_mass
        self.cp_u = self.gas.cp_mass
        self.cv_u = self.gas.cv_mass
        self.rho_u = self.gas.density_mass
        self.mu_u = self.gas.viscosity
        self.nu_u = self.mu_u/self.rho_u
        self.lambda_u= self.gas.thermal_conductivity
        self.alpha_u = self.lambda_u/(self.rho_u*self.cp_u)
        
        self.i_species = self.gas.species_index
        self.D_binary = self.gas.binary_diff_coeffs # Binary diffusion coefficients
        self.D_mix = self.gas.mix_diff_coeffs      # Mixture diffusion Coefficients
        
        # Check molar and mass fractions
        self.X_H2 = self.gas["H2"].X[0]
        self.X_CH4 = self.gas["CH4"].X[0]
        self.X_C2H6 = self.gas["C2H6"].X[0]
        self.X_O2 = self.gas["O2"].X[0]
        self.X_N2 = self.gas["N2"].X[0]
        self.X_Ar = self.gas["AR"].X[0]
        
        self.Y_H2 = self.gas["H2"].Y[0]
        self.Y_CH4 = self.gas["CH4"].Y[0]
        self.Y_C2H6 = self.gas["C2H6"].Y[0]
        self.Y_O2 = self.gas["O2"].Y[0]
        self.Y_N2 = self.gas["N2"].Y[0]
        self.Y_Ar = self.gas["AR"].Y[0]
        
    def solve_equations(self):
        
        # # Check molar and mass fractions
        # self.X_H2 = self.gas["H2"].X[0]
        # self.X_CH4 = self.gas["CH4"].X[0]
        # self.X_C2H6 = self.gas["C2H6"].X[0]
        # self.X_O2 = self.gas["O2"].X[0]
        # self.X_N2 = self.gas["N2"].X[0]
        # self.X_Ar = self.gas["AR"].X[0]
        
        # Set domain size (1D)
        self.width = 0.05 # units: m
        
        # Create object for freely-propagating premixed flames
        flame = ct.FreeFlame(self.gas, width=self.width)
        
        # Set the criteria used to refine one domain
        flame.set_refine_criteria(ratio=3, slope=0.07, curve=0.1)
        
        # Solve the equations
        flame.solve(loglevel=0, auto=True)
        
        self.flame = flame
        
        # Result 1: laminar flame speed
        self.S_L0 = flame.velocity[0] # m/s
        
        # Result 2: adiabtaic flame temperature
        self.T_ad = self.gas.T
        
        # Properties of burnt mixture
        self.h_b = self.gas.enthalpy_mass
        self.cp_b = self.gas.cp_mass
        self.cv_b = self.gas.cv_mass
        self.rho_b = self.gas.density_mass
        self.mu_b = self.gas.viscosity
        self.nu_b = self.mu_b/self.rho_b
        self.lambda_b = self.gas.thermal_conductivity
        self.alpha_b = self.lambda_b/(self.rho_b*self.cp_b)
        self.X_CO2 = self.gas["CO2"].X[0]
        self.X_NO2 = self.gas["NO2"].X[0]
        self.X_NO = self.gas["NO"].X[0]
        self.X_N2O = self.gas["N2O"].X[0]

#%% MAIN

if __name__ == "__main__":
    
    print(f"Running Cantera Version: {ct.__version__}")
    
    # Pure heating values
    fuels = ["H2", "CH4", "C2H6"]
    print("fuel   LHV (MJ/kg)   HHV (MJ/kg)")
    for fuel in fuels:
        LHV, HHV = heating_value(fuel)
        print(f"{fuel:8s} {LHV:7.3f}      {HHV:7.3f}")
    
    LHV_H2, HHV_H2 = heating_value("H2")
    LHV_CH4, HHV_CH4 = heating_value("CH4")
    
    
    mixtures = {}
    
    # mixtures[(0, 0, 0.80)] = []
    # mixtures[(0, 0, 0.90)] = []
    # mixtures[(0, 0, 1.00)] = []

    # mixtures[(1, 20, 0.80)] = []
    # mixtures[(1, 20, 0.90)] = []
    # mixtures[(1, 20, 1.00)] = []

    # mixtures[(2, 40, 0.70)] = []
    # mixtures[(2, 40, 0.80)] = []
    # mixtures[(2, 40, 0.90)] = []
    # mixtures[(2, 40, 1.00)] = []

    # mixtures[(3, 60, 0.60)] = []
    # mixtures[(3, 60, 0.70)] = []
    # mixtures[(3, 60, 0.80)] = []
    # mixtures[(3, 60, 0.90)] = []
    # mixtures[(3, 60, 1.00)] = []

    # mixtures[(4, 80, 0.50)] = []
    # mixtures[(4, 80, 0.60)] = []
    # mixtures[(4, 80, 0.70)] = []
    # mixtures[(4, 80, 0.80)] = []
    # mixtures[(4, 80, 0.90)] = []
    # mixtures[(4, 80, 1.00)] = []
    
    # mixtures[(5, 100, 0.30)] = []
    # mixtures[(5, 100, 0.40)] = []
    # mixtures[(5, 100, 0.49)] = []
    # mixtures[(5, 100, 0.50)] = []
    # mixtures[(5, 100, 0.60)] = []
    # mixtures[(5, 100, 0.70)] = []
    # mixtures[(5, 100, 0.80)] = []
    # mixtures[(5, 100, 0.90)] = []
    # mixtures[(5, 100, 1.00)] = []
    
    # mixtures[(0, 100, 0.48)] = []
    # mixtures[(0, 0, 1)] = []
    # mixtures[(0, 20, 0.88)] = []
    # mixtures[(0, 40, 0.79)] = []
    # mixtures[(0, 60, 0.70)] = []
    # mixtures[(0, 80, 0.60)] = []
    # mixtures[(0, 100, 0.48)] = []

    # filename = 'S_L0_lib6'
    # S_L0_lib = create_library(filename)    
    
    filename = 'S_L0_lib5'
    # S_L0_lib = plot_phi_vs_S_L0(filename)
    S_L0_lib = plot_phi_vs_T_ad(filename)
    
    # Get a list of all currently opened figures
    figure_ids = plt.get_fignums()

    # # Apply tight_layout to each figure
    # for fid in figure_ids:
    #     fig = plt.figure(fid)
    #     # fig.tight_layout()
    #     fig.savefig(f"figures/S_L0_phi_fig{fid}_{filename}.eps", format="eps", dpi=300, bbox_inches="tight")
    #     fig.savefig(f"figures/S_L0_phi_fig{fid}_{filename}.svg", format="svg", dpi=300, bbox_inches="tight")
        
    # # # Equivalence ratios    
    # phis = [1] # Set equivalence ratios ranging from 0.4 to 0.8
    
    # # Hydrogen percentages by volume of fuel
    # H2_percentages = [0] #[0, 20, 40, 60, 80, 100] # Set hydrogen volume percentages of the fuel ranging from 0 to 100 
    
    # # # Define colors to make distinction between different mixtures based on hydrogen percentage
    # colors = cm.viridis(np.linspace(0, 1, len(H2_percentages)))
    
    # # # Initialize list for premixed flame objects
    # flames = []
    
    # # Create flame objects and start simulations
    # for phi in phis:
    #     for H2_percentage in H2_percentages:
            
    #         flame = PremixedFlame(phi, H2_percentage, T_u=273.15+24, p_u=ct.one_atm)
            
    #         m_f = flame.Y_H2 + flame.Y_CH4
    #         LHV_f = flame.Y_H2/m_f*LHV_H2 + flame.Y_CH4/m_f*LHV_CH4    
    #         print(LHV_f)
            
            # flame.solve_equations()
            # flames.append(flame)
    
          # T_u = flame.T_u
          # p_u = flame.p_u
          
          # #compute flame thickness
          # z = flame.flame.grid
          # T = flame.flame.T
          # size = len(z)-1
          # grad = np.zeros(size)
          
          # for i in range(size):
          #     grad[i] = (T[i+1]-T[i])/(z[i+1]-z[i])
              
          # thickness_thermal = (max(T) - min(T)) / max(grad)
          # thickness_diff = flame.alpha_u/flame.S_L0
          # thickness_blint = 2*thickness_diff*(max(T)/min(T))**0.7
          
          # fig1, ax1 = plt.subplots()
          # ax1.plot(z * 100, T, "-o")
          # ax1.set_xlabel("Distance (cm)")
          # ax1.set_ylabel("Temperature (K)")
          
          
    #       # # species_A = 'H2'
    #       # species_A = 'CH4'
          
    #       # species_B = 'N2'
    #       # D_mix = flame.D_mix[flame.i_species(species_A)]
    #       # D_binary = flame.D_binary[flame.i_species(species_A), flame.i_species(species_B)]
          
    #       # print('Mixture solved: phi=' + str(phi) + ', H2%=' + str(H2_percentage))
          
    #       # print("")
    #       # print('Kinematic viscosity of mixture' + ' ' + str(flame.nu_u) + ' (m2/s)')
    #       # print('Mass diffusivity of '+ species_A +' in ' + species_B + ' ' + str(D_binary) + ' (m2/s)')
    #       # print('Thermal diffusivity of mixture' + ' ' + str(flame.alpha_u) + ' (m2/s)')
          
    #       # print("")
    #       # print('Lewis number of '+ species_A +' ' + str(flame.alpha_u/D_binary))
    #       # print('Schmidt number of '+ species_A +' ' + str(flame.nu_u/D_binary))

    #       # print("")
    #       # # print('laminar flame thickness = ', thickness, str(" m"))
    #       # print('laminar flame thickness [thermal thickness] = ', thickness_thermal*1e3, str(" mm"))
    #       # print('laminar flame thickness [diffusive thickness] = ', thickness_diff*1e3, str(" mm"))
    #       # print('laminar flame thickness [blint thickness] = ', thickness_blint*1e3, str(" mm"))
    #       # print('laminar flame thickness [schmidt number = nu_u/D = 1, nu_u] = ', (flame.nu_u/flame.S_L0)*1e3, str(" mm"))
    #       # print('laminar flame thickness [schmidt number = nu_u/D = 1, D] = ', (D_binary/flame.S_L0)*1e3, str(" mm"))
          
    #       # print("")
    #       # print('unburnt mixture density = ' + str(round(flame.rho_u, 3)) + ' kg.m^-3')
    #       # print('unburnt mixture dynamic viscosity = ' + str(flame.mu_u) + ' Pa.s')
    #       # print('unburnt mixture kinematic viscosity = ' + str(flame.nu_u) + ' m^2.s^-1')
          
    #       # print("")
    #       # print('Laminar flame speed = ' + str(round(flame.S_L0, 4)) + ' m/s')
    #       # print('unburnt mixture temperature = ' + str(round(flame.T_u, 2)) + ' K')
    #       # print('Adiabtic flame temperature = ' + str(round(flame.T_ad, 2)) + ' K')
          
    #       # L=0.065*25.16e-3
    #       # L2 = 25.16e-3
    #       # print(L/thickness_thermal)
    #       # print(L/thickness_diff)
          
    #       # print(L2/thickness_thermal)
    #       # print(L2/thickness_diff)
    #       # print('------------------------------------------------------')