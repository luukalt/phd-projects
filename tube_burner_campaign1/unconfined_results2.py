# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 12:25:33 2021

@author: Gersom
"""
#%% IMPORT PACKAGES
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle
from premixed_flame_properties import PremixedFlame

#%% START
plt.close("all")

#%% FIGURE SETTINGS
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Computer Modern Roman"],
#     "font.size": 14.0})

#%% CONSTANTS
# Diameter of the copper tube
D_copper = 25.67 #mm
D_copper *= 1e-3

# Diameter of the quartz tube
D_quartz = 25.16 #mm
D_quartz *= 1e-3

# n different hydrogen percentages
n = 6

#%% Flashback data for the quartz tube [Gersom Willems] 
flashback_quartz_data = {}
flashback_quartz_data[(0, 0, 0.80, "quartz", D_quartz)] = [(0.7956, 0.8432), (0.7981, 0.8470), (0.7984, 0.8345)]
flashback_quartz_data[(0, 0, 0.90, "quartz", D_quartz)] = [(0.9008, 1.6216), (0.9018, 1.5605), (0.9072, 1.5640)]
flashback_quartz_data[(0, 0, 1.00, "quartz", D_quartz)] = [(1.0072, 1.7519), (1.0043, 1.7353), (1.0066, 1.7231)]

flashback_quartz_data[(1, 20, 0.80, "quartz", D_quartz)] = [(0.8013, 1.8335), (0.8022, 1.9211)]
flashback_quartz_data[(1, 20, 0.90, "quartz", D_quartz)] = [(0.9058, 2.3139), (0.9046, 2.2986)]
flashback_quartz_data[(1, 20, 1.00, "quartz", D_quartz)] = [(1.0044, 2.3972), (1.0007, 2.4396), (1.0003, 2.4246)]

flashback_quartz_data[(2, 40, 0.70, "quartz", D_quartz)] = [(0.7018, 2.2524), (0.7017, 2.2214), (0.7029, 2.2540)]
flashback_quartz_data[(2, 40, 0.80, "quartz", D_quartz)] = [(0.8009, 3.0136), (0.8031, 2.9562), (0.8014, 3.0194)]
flashback_quartz_data[(2, 40, 0.90, "quartz", D_quartz)] = [(0.9050, 3.4461), (0.9009, 3.4443), (0.9010, 3.4742)]
flashback_quartz_data[(2, 40, 1.00, "quartz", D_quartz)] = [(0.9996, 3.6493), (1.0033, 3.5868), (1.0024, 3.5664)]

flashback_quartz_data[(3, 60, 0.60, "quartz", D_quartz)] = [(0.6010, 2.8861), (0.6013, 2.9840), (0.6011, 2.9513)]
flashback_quartz_data[(3, 60, 0.70, "quartz", D_quartz)] = [(0.7021, 4.0432), (0.7008, 4.0750), (0.7010, 4.1068)]
flashback_quartz_data[(3, 60, 0.80, "quartz", D_quartz)] = [(0.8021, 5.0689), (0.8011, 5.1576), (0.8003, 5.1703)]
flashback_quartz_data[(3, 60, 0.90, "quartz", D_quartz)] = [(0.9015, 5.5938), (0.9004, 5.6831), (0.9024, 5.7260)]
flashback_quartz_data[(3, 60, 1.00, "quartz", D_quartz)] = [(0.9972, 5.7698), (0.9992, 5.6723), (1.0027, 5.6105)]

flashback_quartz_data[(4, 80, 0.50, "quartz", D_quartz)] = [(0.5028, 3.7789), (0.5038, 3.8709)]
flashback_quartz_data[(4, 80, 0.60, "quartz", D_quartz)] = [(0.6010, 5.8615), (0.6009, 6.1318), (0.6006, 5.8304)]
flashback_quartz_data[(4, 80, 0.70, "quartz", D_quartz)] = [(0.7004, 7.8087), (0.7020, 7.9178), (0.7006, 7.8599)]
flashback_quartz_data[(4, 80, 0.80, "quartz", D_quartz)] = [(0.8012, 9.3564), (0.8016, 9.5569), (0.8022, 9.6970)]
flashback_quartz_data[(4, 80, 0.90, "quartz", D_quartz)] = [(0.9001, 10.6811), (0.9001, 10.5438), (0.9001, 10.7483)]
flashback_quartz_data[(4, 80, 1.00, "quartz", D_quartz)] = [(0.9998, 10.9360), (1.0005, 10.8975), (0.9997, 10.6980)]

flashback_quartz_data[(5, 100, 0.40, "quartz", D_quartz)] = [(0.4020, 5.5762), (0.3989, 5.6370), (0.3988, 5.6719)]
flashback_quartz_data[(5, 100, 0.49, "quartz", D_quartz)] = [(0.49, 8.8), (0.49, 8.7)]
flashback_quartz_data[(5, 100, 0.50, "quartz", D_quartz)] = [(0.4994, 9.2008), (0.5011, 9.4383), (0.4997, 9.4055)]
flashback_quartz_data[(5, 100, 0.60, "quartz", D_quartz)] = [(0.6007, 13.3463), (0.6000, 13.5406), (0.5982, 13.4489)]
flashback_quartz_data[(5, 100, 0.70, "quartz", D_quartz)] = [(0.6950, 15.8375), (0.7009, 16.0897), (0.6996, 16.2997)]
flashback_quartz_data[(5, 100, 0.80, "quartz", D_quartz)] = [(0.8009, 19.1840), (0.7996, 18.7333), (0.8015, 18.7776)]
flashback_quartz_data[(5, 100, 0.90, "quartz", D_quartz)] = [(0.9006, 20.7951), (0.9004, 20.8345), (0.9011, 20.7866)]
flashback_quartz_data[(5, 100, 1.00, "quartz", D_quartz)] = [(1.0000, 22.2811), (1.0008, 22.6272), (1.0000, 22.2180)]


#%% PROCESS DATA
# Read unstretched laminar flame speed data
filename = "S_L0_lib2"
with open('unstreched_laminar_flame_speed_data/'+ filename + '.txt', 'rb') as f:
    S_L0_lib = pickle.load(f)
    
# Read strained laminar flame speed data    
with open('strained_flame_speed_data\Tu293_pu101325\strained_flame_speed_lib.txt', 'rb') as f:
    strained_flame_speed_lib = pickle.load(f)
    

# Initialize lists
phi_lists = [[] for i in range(n)]
U_bulk_lists = [[] for i in range(n)]
Re_D_lists = [[] for i in range(n)]
S_L0_lists = [[] for i in range(n)]
S_u_maxgradT_lists = [[] for i in range(n)]
u_flux_max_lists = [[] for i in range(n)]
u_tau_lists = [[] for i in range(n)]

check_label = ""
check_key = ""

for key, values in flashback_quartz_data.items():
    
    for value in values:
        
        index = key[0]
        H2_percentage = key[1]
        phi = key[2]
        material = key[3]
        D = key[4]
        
        mixture =  PremixedFlame(phi, H2_percentage)
        
        for flame in S_L0_lib:
            if phi == flame["phi"] and H2_percentage == flame["H2%"]:
                S_L0 = flame["S_L0"]
        
        for flame in strained_flame_speed_lib:
            if phi == flame["phi"] and H2_percentage == flame["H2%"]:
                S_u_maxgradT = flame["S_u_maxgradT"][-1]
                
        if key != check_key:
            S_L0_lists[index] = np.append(S_L0_lists[index], S_L0)
            S_u_maxgradT_lists[index] = np.append(S_u_maxgradT_lists[index], S_u_maxgradT)
        else:
            S_L0 = S_L0_lists[index][-1] 
            S_L0_lists[index] = np.append(S_L0_lists[index], S_L0)
            
            S_u_maxgradT = S_u_maxgradT_lists[index][-1] 
            S_u_maxgradT_lists[index] = np.append(S_u_maxgradT_lists[index], S_u_maxgradT)
        
        U_bulk = value[1]
        Re_D = D*U_bulk/mixture.nu_u
        u_tau = (0.03955*U_bulk**(-7/4)*Re_D**(1/4)*D**(-1/4))**(1/2)
        
        # at y+ = 16.4
        u_fluc_max = 1.5*u_tau
        
        phi_lists[index] = np.append(phi_lists[index], phi)
        U_bulk_lists[index] = np.append(U_bulk_lists[index], U_bulk)
        Re_D_lists[index] = np.append(Re_D_lists[index], Re_D)
        u_flux_max_lists[index] = np.append(u_flux_max_lists[index], u_fluc_max)
        u_tau_lists[index] = np.append(u_tau_lists[index], u_tau)
        
        check_label = index
        check_key = key
        
  
#%% PLOTS

linewidth = 1
linestyle = 'None'
labels = ['$0$', '$20$', '$40$', '$60$', '$80$', '$100$']

markers = ['o', 'o', 'o', 'o', 'o', 'o']
markersize = [8, 8, 8, 8, 8, 8]

colors = cm.viridis(np.linspace(0, 1, len(markers)))

fig_scale = 1
default_fig_dim = plt.rcParams["figure.figsize"]

width, height = default_fig_dim[0], default_fig_dim[1]
width, height = 6, 6

fontsize_xlabel = 16
fontsize_ylabel = 16
fontsize_ylabel_fraction = 24

fig1, ax1 = plt.subplots(figsize=(fig_scale*width, fig_scale*height), dpi=100, constrained_layout=True)
ax1.set_xlabel(r'$\phi$', fontsize=fontsize_xlabel)
ax1.set_ylabel(r'$U_{b}$ [ms$^{-1}$]', fontsize=fontsize_ylabel) 

fig2, ax2 = plt.subplots(figsize=(fig_scale*width, fig_scale*height), dpi=100, constrained_layout=True)
ax2.set_xlabel(r'$\phi$', fontsize=fontsize_xlabel)
# ax2.set_ylabel(r'$\frac{U_{b}}{S_{L0}}$', rotation=0, labelpad=0, fontsize=fontsize_ylabel_fraction) 
ax2.set_ylabel(r'$U_{b}/S_{L0}$', fontsize=fontsize_ylabel)

fig3, ax3 = plt.subplots(figsize=(fig_scale*width, fig_scale*height), dpi=100)
ax3.set_xlabel(r'$\phi$', fontsize=fontsize_xlabel)
ax3.set_ylabel(r'$Re_{D}$', fontsize=fontsize_ylabel) 

fig4, ax4 = plt.subplots(figsize=(fig_scale*width, fig_scale*height), dpi=100)
ax4.set_xlabel(r'$\phi$', fontsize=fontsize_xlabel)
ax4.set_ylabel(r'$S_{L0}$ [ms$^{-1}$]', fontsize=fontsize_ylabel)  

fig5, ax5 = plt.subplots(figsize=(fig_scale*width, fig_scale*height), dpi=100)
ax5.set_xlabel(r'$\phi$', fontsize=fontsize_xlabel)
ax5.set_ylabel(r'$\frac{U_{b}}{S_{L,ext}}$', rotation=0, labelpad=15, fontsize=fontsize_ylabel_fraction)  

check_label = ""


# QUARTZ
for i in range(n):
    lenght_j = len(U_bulk_lists[i])
    for j in range(lenght_j):

        phi_test = phi_lists[i][j]
        U_bulk_test = U_bulk_lists[i][j]
        
        Re_D_test = Re_D_lists[i][j]
        S_L0_test = S_L0_lists[i][j]
        S_u_maxgradT_test = S_u_maxgradT_lists[i][j]
        u_fluc = u_flux_max_lists[i][j]
        
        ax1.plot(phi_test, U_bulk_test, ls=linestyle,  marker=markers[i], ms=markersize[i], mec='k', mfc=colors[i], label=labels[i] if i != check_label else "")
        ax2.plot(phi_test, U_bulk_test/S_L0_test, ls=linestyle,  marker=markers[i], ms=markersize[i], mec='k', mfc=colors[i], label=labels[i] if i != check_label else "")
        ax3.plot(phi_test, Re_D_test, ls=linestyle,  marker=markers[i], ms=markersize[i], mec='k', mfc=colors[i], label=labels[i] if i != check_label else "")
        ax4.plot(phi_test, S_L0_test, ls=linestyle,  marker=markers[i], ms=markersize[i], mec='k', mfc=colors[i], label=labels[i] if i != check_label else "")
        ax5.plot(phi_test, U_bulk_test/S_u_maxgradT_test, ls=linestyle,  marker=markers[i], ms=markersize[i], mec='k', mfc=colors[i], label=labels[i] if i != check_label else "")
        
        
        if (phi_test == 0.49 and labels[i] == '$100$') or (phi_test == 1.0 and labels[i] == '$0$'):
            
            # ax1.plot(phi_test, U_bulk_test, ls=linestyle,  marker=markers[i], ms=markersize[i], mec='k', mfc=colors[i], label=labels[i] if i != check_label else "")
            # ax2.plot(phi_test, U_bulk_test/S_L0_test, ls=linestyle,  marker=markers[i], ms=markersize[i], mec='k', mfc=colors[i], label=labels[i] if i != check_label else "")
            
            ax1.plot(phi_test, U_bulk_test, 'rx', ms=10, mew=2, zorder=10)
            ax2.plot(phi_test, U_bulk_test/S_L0_test, 'rx', ms=10, mew=2, zorder=10)
            
            print(phi_test, U_bulk_test, S_L0_test, U_bulk_test/S_L0_test)
            print(32*'-')
        
        check_label = i
        
#% Create polynomial fits
zipped = zip(phi_lists, U_bulk_lists, Re_D_lists, S_L0_lists, S_u_maxgradT_lists, colors)

poly_order = 3
for (phi, U_bulk, Re_D, S_L0, S_u_maxgradT, color) in zipped:
    
    phi_fit = np.linspace(min(phi), max(phi))
    
    poly_U_bulk = np.poly1d(np.polyfit(phi, U_bulk, poly_order))
    U_bulk_fit = poly_U_bulk(phi_fit)
    ax1.plot(phi_fit, U_bulk_fit, ls="--", c=color)
    
    poly_U_bulk_S_L0 = np.poly1d(np.polyfit(phi, U_bulk/S_L0, poly_order))
    U_bulk_S_L0_fit = poly_U_bulk_S_L0(phi_fit)
    ax2.plot(phi_fit, U_bulk_S_L0_fit, ls="--", c=color)
    
    poly_Re_D = np.poly1d(np.polyfit(phi, Re_D, poly_order))
    Re_D_fit = poly_Re_D(phi_fit)
    ax3.plot(phi_fit, Re_D_fit, ls="--", c=color)
    
    poly_S_L0 = np.poly1d(np.polyfit(phi, S_L0, poly_order))
    S_L0_fit = poly_S_L0(phi_fit)
    ax4.plot(phi_fit, S_L0_fit, ls="--", c=color)
    
    poly_U_bulk_S_u_maxgradT = np.poly1d(np.polyfit(phi, U_bulk/S_u_maxgradT, poly_order))
    U_bulk_S_u_maxgradT_fit = poly_U_bulk_S_u_maxgradT(phi_fit)
    ax5.plot(phi_fit, U_bulk_S_u_maxgradT_fit, ls="--", c=color)
    
# Add critical Reynolds number (laminar to turbulent) to figure   
# ax3.axhline(2300, ls='-.', lw=2, c='k', label='$Re_{D, cr}=2300$')
ax3.axhline(2300, ls='-.', lw=1, c='k')
ax3.text(0.375, 2700, r'$Re_{D, transition} = 2300$') 
    
# temperature and pressure of unburnt mixture
T_u = 293.15
p_u = 101325

# set limits
ax1.set_xlim(0.35, 1.05)
ax1.set_ylim(0, 25)

ax2.set_xlim(0.35, 1.05)
ax2.set_ylim(0, 40)

num_ticks = 6
custom_y_ticks = np.linspace(0, 40, num_ticks) # Replace with your desired tick positions
custom_y_tick_labels = [int(tick) for tick in custom_y_ticks] # Replace with your desired tick labels
ax2.set_yticks(custom_y_ticks)
ax2.set_yticklabels(custom_y_tick_labels)  # Use this line to set custom tick labels


ax3.set_xlim(0.35, 1.05)

ax5.set_xlim(0.35, 1.05)
ax5.set_ylim(0, 10)

# Add titles to figures
# ax1.set_title('Flashback propensity map [$U_{bulk}$] of $H_{2}$-DNG/air mixtures \n $T_u=$' + str(round(T_u,2)) + ' K, $p_u$=' + str(p_u*1e-5) + ' bar') 
# ax2.set_title('$U_{bulk}$ normalized by $S_{L0}$ \n $T_u=$' + str(round(T_u,2)) + ' K, $p_u$=' + str(p_u*1e-5) + ' bar') 
# ax3.set_title('Reynolds number based on $U_{bulk}$ \n $T_u=$' + str(round(T_u,2)) + ' K, $p_u$=' + str(p_u*1e-5) + ' bar') 
# ax4.set_title('Unstretched laminar flame speed of $H_{2}$-DNG/air mixtures \n $T_u=$' + str(round(T_u,2)) + ' K, $p_u$=' + str(p_u*1e-5) + ' bar')
# ax5.set_title('$U_{bulk}$ normalized by $S_{L,ext}$ \n $T_u=$' + str(round(T_u,2)) + ' K, $p_u$=' + str(p_u*1e-5) + ' bar') 

# Turn on grids
ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()
ax5.grid()

# Show legends in figures
ax1.legend(title="$H_2\%$", loc="upper left", bbox_to_anchor=(0, 1), ncol=1, prop={"size": 16})
ax2.legend(title="$H_2\%$", loc="upper right", bbox_to_anchor=(1, 1), ncol=1, prop={"size": 16})
ax3.legend(title="$H_2\%$", loc="upper left", bbox_to_anchor=(0, 1), ncol=1, prop={"size": 16})
ax4.legend(title="$H_2\%$", loc="upper left", bbox_to_anchor=(0, 1), ncol=1, prop={"size": 16})
ax5.legend(title="$H_2\%$", loc="upper left", bbox_to_anchor=(0, 1), ncol=1, prop={"size": 16})

# Get a list of all currently opened figures
figure_ids = plt.get_fignums()

# Adjust the padding between and around subplots
for fid in figure_ids:
    fig = plt.figure(fid)
    fig.tight_layout()
    fig.savefig(f"figures/fb_maps_fig{fid}_{filename}.eps", format="eps", dpi=300, bbox_inches="tight")











