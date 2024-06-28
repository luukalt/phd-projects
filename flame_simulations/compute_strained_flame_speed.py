
# coding: utf-8

"""
Simulate two counter-flow jets of reactants shooting into each other. This
simulation differs from the similar premixed_counterflow_flame.py example as the
latter simulates a jet of reactants shooting into products.

Requires: cantera >= 2.5.0
Keywords: combustion, 1D flow, premixed flame, strained flame, plotting
"""
#%% IMPORT PACKAGES
import cantera as ct
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import sys
import pickle
import logging

#%% START
plt.close("all")

#%% FIGURE SETTINGS
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 14.0})

#%% CONSTANTS
R_gas_mol = 8314 # Universal gas constant [units: J*K^-1*kmol^-1]
R_gas_mass = 287 # universal gas constant [units: J*K^-1*kg^-1]

# Define volume fractions of species in air [units: -]
# f_O2 = 0.21
# f_N2 = 0.78 # 0.78
# f_AR = 0.01

f_O2 = 0.21
f_N2 = 0.79 # 0.78
f_AR = 0

# Define composition of DNG in volume percentages
# REST_in_DNG_percentage = 0.6
# CH4_percentage = 81.3*(1 + REST_in_DNG_percentage/100) # Volume percentage of CH4 in DNG
# C2H6_percentage = 3.7*(1 + REST_in_DNG_percentage/100) # Volume percentage of C2H6 in DNG
# N2_in_DNG_percentage = 14.4*(1 + REST_in_DNG_percentage/100) # Volume percentage of N2 in DNG

REST_in_DNG_percentage = 0 #0.6
CH4_percentage = 100 #81.3*(1 + REST_in_DNG_percentage/100) # Volume percentage of CH4 in DNG
C2H6_percentage = 0 #3.7*(1 + REST_in_DNG_percentage/100) # Volume percentage of C2H6 in DNG
N2_in_DNG_percentage = 0 #14.4*(1 + REST_in_DNG_percentage/100) # Volume percentage of N2 in DNG
    
#%% FUNCTIONS
# Differentiation function for data that has variable grid spacing Used here to
# compute normal strain-rate
def derivative(x, y):
    dydx = np.zeros(y.shape, y.dtype.type)

    dx = np.diff(x)
    dy = np.diff(y)
    dydx[0:-1] = dy/dx

    dydx[-1] = (y[-1] - y[-2])/(x[-1] - x[-2])

    return dydx


def computeStrainRates(oppFlame):
    # Compute the derivative of axial velocity to obtain normal strain rate
    strainRates = derivative(oppFlame.grid, oppFlame.velocity)
    
    indices = []
    for idx in range(0, len(strainRates) - 1):
      
        # checking for successive opposite index
        if strainRates[idx] > 0 and strainRates[idx + 1] < 0 or strainRates[idx] < 0 and strainRates[idx + 1] > 0:
            indices.append(idx + 1)
    
    strainRates_upstream = strainRates[:indices[0]]
    
    # Obtain the location of the max. strain rate upstream of the pre-heat zone.
    # This is the characteristic strain rate
    maxStrLocation = abs(strainRates_upstream).argmax()
    minVelocityPoint = oppFlame.velocity[:maxStrLocation].argmin()

    # Characteristic Strain Rate = K
    strainRatePoint = abs(strainRates_upstream[:minVelocityPoint]).argmax()
    K = abs(strainRates_upstream[strainRatePoint])
    
    i_min = indices[0]
    i_max = indices[1]
    i_maxgradT = np.argmax(np.gradient(oppFlame.T))

    return strainRates, strainRatePoint, K, i_min, i_max, i_maxgradT


def computeConsumptionSpeed(oppFlame):

    Tb = max(oppFlame.T)
    Tu = min(oppFlame.T)
    rho_u = max(oppFlame.density)

    integrand = oppFlame.heat_release_rate/oppFlame.cp

    total_heat_release = np.trapz(integrand, oppFlame.grid)
    
    if Tb-Tu == 0:
        Sc = 0
    else:
        Sc = total_heat_release/(Tb - Tu)/rho_u
        
        
    return Sc


# This function is called to run the solver
def solveOpposedFlame(oppFlame, massFlux, loglevel=0,
                      ratio=2, slope=0.2, curve=0.2, prune=0.05):
    """
    Execute this function to run the Oppposed Flow Simulation This function
    takes a CounterFlowTwinPremixedFlame object as the first argument
    """

    oppFlame.reactants.mdot = massFlux
    oppFlame.set_refine_criteria(ratio=ratio, slope=slope, curve=curve, prune=prune)

    # oppFlame.show_solution()
    oppFlame.solve(loglevel, auto=True)

    # Compute the strain rate, just before the flame. This is not necessarily
    # the maximum We use the max. strain rate just upstream of the pre-heat zone
    # as this is the strain rate that computations compare against, like when
    # plotting Su vs. K
    strainRates, strainRatePoint, K, i_min, i_max, i_maxgradT = computeStrainRates(oppFlame)

    return K, strainRates, strainRatePoint, i_min, i_max, i_maxgradT


def defineMixture(T_u, p_u, phi, H2_percentage):
    
    # DNG percentage of fuel
    DNG_percentage = 100 - H2_percentage
                              
    # Volume fractions of fuel
    f_H2 = H2_percentage/100
    f_DNG = DNG_percentage/100
    f_CH4 = f_DNG*(CH4_percentage/100)
    f_C2H6 = f_DNG*(C2H6_percentage/100)
    f_N2_in_DNG = f_DNG*(N2_in_DNG_percentage/100)
    
    # Check if volume fractions of fuel and air are correct
    # check_air = f_O2 + f_N2 + f_AR
    check_air = f_O2 + f_N2
    
    check_fuel = f_H2 + f_CH4 + f_C2H6 + f_N2_in_DNG
    
    if check_air == 1.0 and round(check_fuel,3) == 1.0:
        # print("hello")
        pass
    else:
        sys.exit("fuel or air composition is incorrect!" + "sum of fuel volume fractions:" + str(check_fuel))
        
    if round(check_fuel,3) == 1.0:
        # print("hello")
        pass
    else:
        sys.exit("fuel composition is incorrect!")
    
    # Select the reaction mechanism
    gas = ct.Solution('gri30.yaml')
    # # gas = ct.Solution(os.getcwd() + '/reaction_mechanisms/oconair/chem.cti') 
    # # gas = ct.Solution(os.getcwd() + '/reaction_mechanisms/li/h2o2mech_Lietal_2003.dat') 
    
    # # Define the fuel and air composition
    fuel = {'H2':f_H2, 'CH4':f_CH4, 'C2H6':f_C2H6, 'N2':f_N2_in_DNG}
    air = {'N2':f_N2/f_O2, 'O2':1.0, 'AR':f_AR/f_O2}
    # air = {'N2':f_N2/f_O2, 'O2':1.0}
    
    # Set the equivalence ratio
    gas.set_equivalence_ratio(phi, fuel, air)
    # gas.set_equivalence_ratio(phi, 'H2', {'O2': 1.0, 'N2': 3.76})
    
    # Set temperature and pressure.
    gas.TP = T_u, p_u
    
    return gas
          
def computeStrainedFlameSpeed(T_u=293.15, p_u=101325, phi=1, H2_percentage=0, axial_velocity=10):
    
    # Define mixture 
    gas = defineMixture(T_u, p_u, phi, H2_percentage)

    # Set the velocity
    # axial_velocity = axial_velocity # in m/s

    # Domain half-width of 2.5 cm, meaning the whole domain is 5 cm wide
    width = 0.025

    # Done with initial conditions
    # Compute the mass flux, as this is what the Flame object requires
    massFlux = gas.density*axial_velocity  # units kg/m2/s

    # Create the flame object
    oppFlame = ct.CounterflowTwinPremixedFlame(gas, width=width)

    # Uncomment the following line to use a Multi-component formulation. Default is
    # mixture-averaged
    oppFlame.transport_model = 'Multi'

    # Now run the solver. The solver returns the peak temperature, strain rate and
    # the point which we ascribe to the characteristic strain rate.
    (K, strainRates, strainRatePoint, i_min, i_max, i_maxgradT) = solveOpposedFlame(oppFlame, massFlux, loglevel=0)

    # You can plot/see all state space variables by calling oppFlame.foo where foo
    # is T, Y[i], etc. The spatial variable (distance in meters) is in oppFlame.grid
    # Thus to plot temperature vs distance, use oppFlame.grid and oppFlame.T
    
    Tb = max(oppFlame.T)
    Tu = min(oppFlame.T)
    
    # When Tb = Tu, the flame is extinct.
    if Tb == Tu:
        print("flame extinct!")
    else:
        # pass    
        xloc_min = oppFlame.grid[i_min]
        xloc_max = oppFlame.grid[i_max]
        xloc_maxgradT = oppFlame.grid[i_maxgradT]
        
        S_u_min = oppFlame.velocity[i_min]
        S_u_max = oppFlame.velocity[i_max]
        S_u_maxgradT = oppFlame.velocity[i_maxgradT]

        # fig2, (ax21, ax22) = plt.subplots(1, 2)
        
        # # Axial Velocity Plot
        # ax21.plot(oppFlame.grid, oppFlame.velocity, 'r', lw=2)
        # ax21.set_xlim(oppFlame.grid[0], oppFlame.grid[-1])
        # ax21.set_xlabel('Distance (m)')
        # ax21.set_ylabel('Axial Velocity (m/s)')
        
        # ax21.plot(xloc_maxgradT, S_u_maxgradT, 'bs', lw=2)
        # ax21.plot(oppFlame.grid, oppFlame.velocity, 'ms', lw=2)
        # ax21.plot(oppFlame.grid[i_min], oppFlame.velocity[i_min], 'ko', lw=2)
        # ax21.plot(oppFlame.grid[i_max], oppFlame.velocity[i_max], 'ko', lw=2)
        
        # # Identify the point where the strain rate is calculated
        # ax21.plot(oppFlame.grid[strainRatePoint],
        #           oppFlame.velocity[strainRatePoint], 'gs')
        # ax21.annotate('Strain-Rate point',
        #               xy=(oppFlame.grid[strainRatePoint],
        #                   oppFlame.velocity[strainRatePoint]),
        #               xytext=(0.001, 0.1),
        #               arrowprops={'arrowstyle': '->'})
        
        # # Temperature Plot
        # ax22.plot(oppFlame.grid, oppFlame.T, 'b', lw=2)
        # ax22.set_xlim(oppFlame.grid[0], oppFlame.grid[-1])
        # ax22.set_xlabel('Distance (m)')
        # ax22.set_ylabel('Temperature (K)')

        # ax22.plot(xloc_maxgradT, oppFlame.T[i_maxgradT], 'bs', lw=2)
    
        
        # fig2.tight_layout()
    
    # print("Peak temperature: {0:.1f} K".format(T))
    logger.info("Progess... phi:{0:.2f}".format(phi) + ", H2%:{:d}".format(H2_percentage) + ", T_u:{0:.2f} K".format(T_u) + ", p_u:{:d} Pa".format(p_u))
    logger.info("Characteristic Strain Rate (upstream of flame front): {0:.1f} 1/s".format(K))
    logger.info("Velocity at (dT/dx)_max: {0:.2f} m/s".format(S_u_maxgradT))
    logger.info("Axial velocity: {0:.2f} m/s".format(axial_velocity))
    # print("--------------------------------------------")
    
    strained_flame_data = {"oppFLame":oppFlame, 
                           "strain_rate_charact":K,
                           "strainRates":strainRates,
                           "strainRatePoint":strainRatePoint,
                           "S_u_min":(i_min, xloc_min, S_u_min), 
                           "S_u_max":(i_max, xloc_max, S_u_max), 
                           "S_u_maxgradT":(i_maxgradT, xloc_maxgradT, S_u_maxgradT)}
    # del gas
    # del oppFlame
    
    return strained_flame_data


#%% AUXILIARY FUNCTIONS
def createFlameSpeedLibrary():
    
    labels = ['$0$', '$20$', '$40$', '$60$', '$80$', '$100$']
    
    T_u = 293.15
    p_u = 101325
    
    # List op (phi, H2_percentage, initial axial velocity estimate) tuples
    list_H0 = [(0.8, 0, 20), (0.9, 0, 20), (1.0, 0, 20)]
    list_H20 = [(0.8, 20, 30), (0.9, 20, 34), (1.0, 20, 34)]
    list_H40 = [(0.7, 40, 28), (0.8, 40, 28), (0.9, 40, 28), (1.0, 40, 32)]
    list_H60 = [(0.6, 60, 32), (0.7, 60, 32), (0.8, 60, 32), (0.9, 60, 32), (1.0, 60, 32)]
    list_H80 = [(0.5, 80, 28), (0.6, 80, 28), (0.7, 80, 32), (0.8, 80, 32), (0.9, 80, 32), (1.0, 80, 41)]
    list_H100 = [(0.4, 100, 49), (0.5, 100, 66), (0.6, 100, 87), (0.7, 100, 96), (0.8, 100, 156), (0.9, 100, 171), (1.0, 100, 348)]
    
    multiple_list = [list_H0, list_H20, list_H40, list_H60, list_H80, list_H100]    
    
    # multiple_list = [[(0.4, 100, 49)]]
    
    error_limit = 0.01
    
    strained_flame_speed_lib = []
    
    for i, single_list in enumerate(multiple_list):
        
        for pair in single_list:
            
            phi = pair[0]
            H2_percentage = pair[1]
            axial_velocity = pair[2]
            
            S_u_maxgradT_new = 10
            error = 1
            
            axial_velocity_up = 1
            axial_velocity_down = -0.1
            direction = 0
            
            while error > error_limit:
                
                axial_velocity += direction
                
                try:
                    strained_flame_data = computeStrainedFlameSpeed(T_u=T_u, p_u=p_u, phi=phi, H2_percentage=H2_percentage, axial_velocity=axial_velocity)
                    
                    S_u_maxgradT = strained_flame_data["S_u_maxgradT"][-1]
                    S_u_maxgradT_old = S_u_maxgradT_new
                    S_u_maxgradT_new = S_u_maxgradT
                    
                    error = np.abs((S_u_maxgradT_old - S_u_maxgradT_new) / S_u_maxgradT_old)
                    
                    direction = axial_velocity_up
                    
                    logger.info("Error: {0:.6f}".format(error) + " (> {0:.3f})".format(error_limit))
                    logger.info("--------------------------------------------")
                except:
                    # break
                    direction = axial_velocity_down
        
        
            logger.info("Finished! phi:{0:.2f}".format(phi) + ", H2%:{:d}".format(H2_percentage) + ", T_u:{0:.2f} K".format(T_u) + ", p_u:{0:.0f} Pa".format(p_u))       
            logger.info("Velocity at (dT/dx)_max: {0:.3f} m/s".format(S_u_maxgradT))
            logger.info("--------------------------------------------")
            
            oppFlame = strained_flame_data["oppFLame"]
            oppFlame.write_csv("phi{0:.2f}".format(phi) + "_" +
                               "H{:d}".format(H2_percentage) + "_" +
                               "Tu{0:.0f}".format(T_u) + "_" +
                               "pu{0:.0f}".format(p_u) +
                               ".csv", quiet=False)
            
            # data item
            lib_item = {'phi':phi, 'H2%':H2_percentage, 
                        'S_u_maxgradT':strained_flame_data["S_u_maxgradT"], 'S_u_min':strained_flame_data["S_u_min"], 'S_u_max':strained_flame_data["S_u_max"],
                        "strain_rate_charact":strained_flame_data["strain_rate_charact"], 'label_index':i, 'label_text':labels[i]}
            
            # Append item to strained_flame_speed_lib
            strained_flame_speed_lib.append(lib_item)
            
    # Write strained_flame_speed_lib to file 
    with open('strained_flame_speed_lib.txt', 'wb') as f:
        pickle.dump(strained_flame_speed_lib, f) 
        
    

def plot_phi_vs_flamespeed():
    
    linestyle = 'None'
    
    
    markers = ['o', 'o', 'o', 'o', 'o', 'o']
    # markers = ['o', '^', 's', 'p', 'X', '*']
    
    markersize = [8, 8, 8, 8, 8, 8]
    
    labels = ['$0$', '$20$', '$40$', '$60$', '$80$', '$100$']
    colors = cm.viridis(np.linspace(0, 1, len(markers)))
    
    with open('strained_flame_speed_data\Tu293_pu101325\strained_flame_speed_lib.txt', 'rb') as f:
        strained_flame_speed_lib = pickle.load(f)
    
    # fig1, ax1 = plt.subplots()
    # ax1.set_xlabel('$K$ [s$^{-1}$]')
    # ax1.set_ylabel('$S_{L,ext} [ms$^-1$]$')
    # # ax1.set_title("Laminar flame speed versus characteristic strain rate for phi=" + str(phi))
    # ax1.set_xscale('log')
    # ax1.set_xlim(1e3, 1e5)
    # ax1.set_ylim(0, 5)
    # ax1.grid()
    
    fig_scale = 1
    default_fig_dim = plt.rcParams["figure.figsize"]
    
    fig2, ax2 = plt.subplots(figsize=(fig_scale*default_fig_dim[0], fig_scale*default_fig_dim[1]), dpi=100)
    ax2.set_xlabel('$\phi$ [-]')
    ax2.set_ylabel('$S_{L,ext}$ [ms$^-1$]') 
    ax2.set_xlim(0.3, 1.1)
    ax2.set_ylim(0, 10)
    ax2.grid()
    
    n = 6
    phi_lists = [[] for i in range(n)]
    S_u_maxgradT_lists = [[] for i in range(n)]
    
    check_label = ""
    
    for flame_item in strained_flame_speed_lib:
        phi = flame_item["phi"]
        S_u_maxgradT = flame_item["S_u_maxgradT"][-1]
        i = flame_item["label_index"]
        
        phi_lists[i] = np.append(phi_lists[i], phi)
        S_u_maxgradT_lists[i] = np.append(S_u_maxgradT_lists[i], S_u_maxgradT)
        
        ax2.plot(phi, S_u_maxgradT, ls=linestyle, marker=markers[i], ms=markersize[i], mec='k', mfc=colors[i], label=labels[i] if i != check_label else "")
    
        check_label = i
        
    zipped = zip(phi_lists, S_u_maxgradT_lists, colors)
    
    poly_order = 3
    
    for (phi, S_u_maxgradT, color) in zipped:
    
        phi_fit = np.linspace(min(phi), max(phi))
    
        poly_S_u_maxgradT = np.poly1d(np.polyfit(phi, S_u_maxgradT, poly_order))
        S_u_maxgradT_fit = poly_S_u_maxgradT(phi_fit)
        ax2.plot(phi_fit, S_u_maxgradT_fit, ls="--", c=color)
    
    ax2.legend(title="$H_2\%$", loc="upper left", bbox_to_anchor=(0, 1), ncol=1, prop={"size": 12})
    fig2.tight_layout()

    return strained_flame_speed_lib

#%% MAIN

# Before executing code, Python interpreter reads source file and define few special variables/global variables. 
# If the python interpreter is running that module (the source file) as the main program, it sets the special __name__ variable to have a value “__main__”. If this file is being imported from another module, __name__ will be set to the module’s name. Module’s name is available as value to __name__ global variable. 
if __name__ == "__main__":
    
    # #now we will Create and configure logger 
    # logging.basicConfig(filename="log.txt", 
    # 					format='%(asctime)s %(message)s', 
    # 					filemode='w',
    #                     force=True) 

    # #Let us Create an object 
    # logger=logging.getLogger() 

    # #Now we are going to Set the threshold of logger to DEBUG 
    # logger.setLevel(logging.DEBUG) 
    
    # # Fill the library!
    # createFlameSpeedLibrary()
    
    # # Shutdown the logging
    # logging.shutdown()
    
    strained_flame_speed_lib = plot_phi_vs_flamespeed()
    

    
