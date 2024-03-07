# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 12:25:33 2021

@author: Gersom
"""
#%% IMPORT PACKAGES
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from premixed_flame_properties import PremixedFlame

#%% START
plt.close("all")

#%% FIGURE SETTINGS
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 14.0})

#%% CONSTANTS
# Diameter of the copper tube
D_copper = 25.67 #mm
D_copper *= 1e-3

# Diameter of the quartz tube
D_quartz = 25.16 #mm
D_quartz *= 1e-3

# n different hydrogen percentages
n = 6

#%% Flashback data for the copper tube [Filippo Faldella] 
# flashback_copper_data = {}
# flashback_copper_data[(0, 0, 0.80, "copper", D_copper)] = [(0.8002, 1.4974)]
# flashback_copper_data[(0, 0, 0.90, "copper", D_copper)] = [(0.9010, 1.9094)]
# flashback_copper_data[(0, 0, 1.00, "copper", D_copper)] = [(1.0011, 2.1717)]

# flashback_copper_data[(1, 0, 0.80, "copper", D_copper)] = [(0.7988, 2.1717)]
# flashback_copper_data[(1, 0, 0.90, "copper", D_copper)] = [(0.9003, 2.5837)]
# flashback_copper_data[(1, 0, 1.00, "copper", D_copper)] = [(0.9989, 2.8460)]

# flashback_copper_data[(2, 40, 0.80, "copper", D_copper)] = [(0.8003, 3.0707)]
# flashback_copper_data[(2, 40, 0.90, "copper", D_copper)] = [(0.9009, 3.5577)]
# flashback_copper_data[(2, 40, 1.00, "copper", D_copper)] = [(1.0000, 3.8574)]

# flashback_copper_data[(3, 60, 0.70, "copper", D_copper)] = [(0.6980, 3.9323)]
# flashback_copper_data[(3, 60, 0.80, "copper", D_copper)] = [(0.7995, 4.7190)]
# flashback_copper_data[(3, 60, 0.90, "copper", D_copper)] = [(0.8996, 5.2435)]
# flashback_copper_data[(3, 60, 1.00, "copper", D_copper)] = [(0.9982, 5.5431)]

# flashback_copper_data[(4, 80, 0.60, "copper", D_copper)] = [(0.6006, 5.3558)]
# flashback_copper_data[(4, 80, 0.70, "copper", D_copper)] = [(0.7005, 6.8917)]
# flashback_copper_data[(4, 80, 0.80, "copper", D_copper)] = [(0.8010, 8.0904)]
# flashback_copper_data[(4, 80, 0.90, "copper", D_copper)] = [(0.9009, 9.1019)]
# flashback_copper_data[(4, 80, 1.00, "copper", D_copper)] = [(1.0007, 9.2892)]

# flashback_copper_data[(5, 100, 0.60, "copper", D_copper)] = [(0.5999, 9.6264)]
# flashback_copper_data[(5, 100, 0.70, "copper", D_copper)] = [(0.6991, 11.1248)]
# flashback_copper_data[(5, 100, 0.80, "copper", D_copper)] = [(0.7996, 12.9979)]
# flashback_copper_data[(5, 100, 0.90, "copper", D_copper)] = [(0.8988, 13.1852)]
# flashback_copper_data[(5, 100, 1.00, "copper", D_copper)] = [(0.9993, 14.1967)]

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
flashback_quartz_data[(5, 100, 0.50, "quartz", D_quartz)] = [(0.4994, 9.2008), (0.5011, 9.4383), (0.4997, 9.4055)]
flashback_quartz_data[(5, 100, 0.60, "quartz", D_quartz)] = [(0.6007, 13.3463), (0.6000, 13.5406), (0.5982, 13.4489)]
flashback_quartz_data[(5, 100, 0.70, "quartz", D_quartz)] = [(0.6950, 15.8375), (0.7009, 16.0897), (0.6996, 16.2997)]
flashback_quartz_data[(5, 100, 0.80, "quartz", D_quartz)] = [(0.8009, 19.1840), (0.7996, 18.7333), (0.8015, 18.7776)]
flashback_quartz_data[(5, 100, 0.90, "quartz", D_quartz)] = [(0.9006, 20.7951), (0.9004, 20.8345), (0.9011, 20.7866)]
flashback_quartz_data[(5, 100, 1.00, "quartz", D_quartz)] = [(1.0000, 22.2811), (1.0008, 22.6272), (1.0000, 22.2180)]


#%% PLOTS 
# # Plot parameters
# markers = ['o','v','s','D', 'p', '>']
# # markers_old = ['o','o','o','o', 'o', 'o']
# # facecolors = ['lightsalmon','lightgreen','grey', 'skyblue', 'violet', 'sandybrown']
# # edgecolors = ['darkred', 'darkgreen', 'black', 'darkblue', 'indigo', 'darkorange']
# colors_old = cm.viridis(np.linspace(0, 1, len(markers)))
# colors_old[-1] = "sandybrown"
# labels = ['0\% $H_{2}$', '20\% $H_{2}$', '40\% $H_{2}$', '60\% $H_{2}$', '80\% $H_{2}$', '100\% $H_{2}$']

# markersize = 8
# linewidth = 1
# linestyle = 'None'

# fig1, ax1 = plt.subplots()
# ax1.set_xlabel('$\phi$ [-]')
# ax1.set_ylabel('$U_{b}$ [ms$^-1$]') 

# fig2, ax2 = plt.subplots()
# ax2.set_xlabel('$\phi$ [-]')
# ax2.set_ylabel('$U_{b}/S_{L,0}$ [-]') 

# fig3, ax3 = plt.subplots()
# ax3.set_xlabel('$\phi$ [-]')
# ax3.set_ylabel('$Re_{D}$ [-]') 

# fig4, ax4 = plt.subplots()
# ax4.set_xlabel('$\phi$ [-]')
# ax4.set_ylabel('$S_{L,0}$ [ms$^-1$]')  

#%% COPPER

# # Initialize lists
# phi_lists1 = [[] for i in range(n)]
# U_bulk_lists1 = [[] for i in range(n)]
# Re_D_lists1 = [[] for i in range(n)]
# S_L0_lists1 = [[] for i in range(n)]
# u_flux_max_lists1 = [[] for i in range(n)]
# u_tau_lists1 = [[] for i in range(n)]
# flame_lists1 = [[] for i in range(n)]

# check_label = ''
# check_key = ''

# for key, values in flashback_copper_data.items():
    
#     for value in values:
        
#         index = key[0]
#         H2_percentage = key[1]
#         phi = value[0]
#         material = key[3]
#         D = key[4]
#         # phi = key[2]
                
#         if key != check_key:
#             flame = PremixedFlame(phi, H2_percentage)
#             flame.solve_equations()
#             S_L0 = flame.S_L0 
#             S_L0_lists1[index] = np.append(S_L0_lists1[index], S_L0)
#             flame_lists1[index] = np.append(flame_lists1[index], flame)
#             print('flame calculated: phi=' + str(round(phi, 2)) + ', H2%=' + str(H2_percentage))
#         else:
#             S_L0 = S_L0_lists1[index][-1] 
#             S_L0_lists1[index] = np.append(S_L0_lists1[index], S_L0)
#             flame = flame_lists1[index][-1]
#             flame_lists1[index] = np.append(flame_lists1[index], flame)
        
#         U_bulk = value[1]
#         Re_D = D*U_bulk/flame.nu_u
#         u_tau = (0.03955*U_bulk**(-7/4)*Re_D**(1/4)*D**(-1/4))**(1/2)
        
#         # at y+ = 16.4
#         u_fluc_max = 1.5*u_tau
        
#         phi_lists1[index] = np.append(phi_lists1[index], phi)
#         U_bulk_lists1[index] = np.append(U_bulk_lists1[index], U_bulk)
#         Re_D_lists1[index] = np.append(Re_D_lists1[index], Re_D)
#         u_flux_max_lists1[index] = np.append(u_flux_max_lists1[index], u_fluc_max)
#         u_tau_lists1[index] = np.append(u_tau_lists1[index], u_tau)
        
#         # ax1.plot(phi, U_bulk, ls=linestyle, marker=markers[index], ms=markersize, mec="k", mfc=colors_old[index], label=labels[index] if index != check_label else "")
#         # ax2.plot(phi, U_bulk/S_L0, ls=linestyle, marker=markers[index], ms=markersize, mec="k", mfc=colors_old[index], label=labels[index] if index != check_label else "")
#         # ax3.plot(phi, Re_D, ls=linestyle, marker=markers[index], ms=markersize, mec="k", mfc=colors_old[index], label=labels[index] if index != check_label else "")
#         # ax4.plot(phi, S_L0, ls=linestyle, marker=markers[index], ms=markersize, mec="k", mfc=colors_old[index], label=labels[index] if index != check_label else "")
        
#         check_label = index
#         check_key = key


#%% QUARTZ

# Initialize lists
phi_lists2 = [[] for i in range(n)]
U_bulk_lists2 = [[] for i in range(n)]
Re_D_lists2 = [[] for i in range(n)]
S_L0_lists2 = [[] for i in range(n)]
u_flux_max_lists2 = [[] for i in range(n)]
u_tau_lists2 = [[] for i in range(n)]
flame_lists2 = [[] for i in range(n)]

check_label = ''
check_key = ''

for key, values in flashback_quartz_data.items():
    
    for value in values:
        
        index = key[0]
        H2_percentage = key[1]
        phi = value[0]
        material = key[3]
        D = key[4]
        # phi = key[2]
                
        if key != check_key:
            flame = PremixedFlame(phi, H2_percentage)
            flame.solve_equations()
            S_L0 = flame.S_L0 
            S_L0_lists2[index] = np.append(S_L0_lists2[index], S_L0)
            flame_lists2[index] = np.append(flame_lists2[index], flame)
            print('flame calculated: phi=' + str(round(phi, 2)) + ', H2%=' + str(H2_percentage))
        else:
            S_L0 = S_L0_lists2[index][-1] 
            S_L0_lists2[index] = np.append(S_L0_lists2[index], S_L0)
            flame = flame_lists2[index][-1]
            flame_lists2[index] = np.append(flame_lists2[index], flame)
        
        U_bulk = value[1]
        Re_D = D*U_bulk/flame.nu_u
        u_tau = (0.03955*U_bulk**(-7/4)*Re_D**(1/4)*D**(-1/4))**(1/2)
        
        # at y+ = 16.4
        u_fluc_max = 1.5*u_tau
        
        phi_lists2[index] = np.append(phi_lists2[index], phi)
        U_bulk_lists2[index] = np.append(U_bulk_lists2[index], U_bulk)
        Re_D_lists2[index] = np.append(Re_D_lists2[index], Re_D)
        u_flux_max_lists2[index] = np.append(u_flux_max_lists2[index], u_fluc_max)
        u_tau_lists2[index] = np.append(u_tau_lists2[index], u_tau)
        
        # ax1.plot(phi, U_bulk, ls=linestyle, marker=markers[index], ms=markersize, mec="k", mfc=colors_old[index], label=labels[index] if index != check_label else "")
        # ax2.plot(phi, U_bulk/S_L0, ls=linestyle, marker=markers[index], ms=markersize, mec="k", mfc=colors_old[index], label=labels[index] if index != check_label else "")
        # ax3.plot(phi, Re_D, ls=linestyle, marker=markers[index], ms=markersize, mec="k", mfc=colors_old[index], label=labels[index] if index != check_label else "")
        # ax4.plot(phi, S_L0, ls=linestyle, marker=markers[index], ms=markersize, mec="k", mfc=colors_old[index], label=labels[index] if index != check_label else "")
        
        check_label = index
        check_key = key
  
#%% MAIN
plt.close('all')

linewidth = 1
linestyle = 'None'
labels = ['$0$', '$20$', '$40$', '$60$', '$80$', '$100$']

markersize = [8, 8, 8, 8, 8, 8]

markers = ['o', 'o', 'o', 'o', 'o', 'o']
colors = cm.viridis(np.linspace(0, 1, len(markers)))
edgecolors = cm.viridis(np.linspace(0, 1, 2*len(markers)))
# colors_old[-1] = "sandybrown"
# colors2_old = cm.Greys(np.linspace(0, 1, len(markers_old)))
# facecolors = ['lightsalmon','lightgreen','grey', 'skyblue', 'violet', 'sandybrown']
# edgecolors = ['darkred', 'darkgreen', 'black', 'darkblue', 'indigo', 'darkorange']


fig_old, ax_old = plt.subplots()
ax_old.set_xlabel('$\phi$ [-]')
ax_old.set_ylabel('$U_{b}$ [ms$^-1$]') 

fig1, ax1 = plt.subplots()
ax1.set_xlabel('$\phi$ [-]')
ax1.set_ylabel('$U_{b}$ [ms$^-1$]') 

fig2, ax2 = plt.subplots()
ax2.set_xlabel('$\phi$ [-]')
ax2.set_ylabel('$U_{bulk}/S_{L0}$ [-]') 

fig3, ax3 = plt.subplots()
ax3.set_xlabel('$\phi$ [-]')
ax3.set_ylabel('$Re_{D}$ [-]') 

fig4, ax4 = plt.subplots()
ax4.set_xlabel('$\phi$ [-]')
ax4.set_ylabel('$S_{L0}$ [ms$^-1$]')  

fig5, ax5 = plt.subplots()
ax5.set_xlabel('$\phi$ [-]')
ax5.set_ylabel('$u/S_{L,0}$ [-]')  

check_label = ''

# # COPPER
# for i in range(n):
#     lenght_j = len(U_bulk_lists1[i])
#     for j in range(lenght_j):

#         phi_test = phi_lists1[i][j]
#         U_bulk_test = U_bulk_lists1[i][j]
        
#         flame_test = flame_lists1[i][j]
#         Re_D_test = D_copper*U_bulk_test/flame_test.nu_u
#         S_L0_test = flame_test.S_L0
#         u_fluc = u_flux_max_lists1[i][j]
        
#         S_T_test = S_L0_test*(1 + u_fluc/S_L0_test)
        
#         # ax_old.plot(phi_test, U_bulk_test, ls=linestyle,  marker=markers[i], ms=markersize[i], mec='k', mfc=colors_old[i], label=labels[i] if i != check_label else "")
#         # ax1.plot(phi_test, U_bulk_test, ls=linestyle,  marker=markers[i], ms=markersize[i], mec='k', mfc=colors_old[i], label=labels[i] if i != check_label else "")
#         # ax2.plot(phi_test, U_bulk_test/S_L0_test, ls=linestyle,  marker=markers[i], ms=markersize[i], mec='k', mfc=colors_old[i], label=labels[i] if i != check_label else "")
#         # ax3.plot(phi_test, Re_D_test, ls=linestyle,  marker=markers[i], ms=markersize[i], mec='k', mfc=colors_old[i], label=labels[i] if i != check_label else "")
#         # ax4.plot(phi_test, S_L0_test, ls=linestyle,  marker=markers[i], ms=markersize[i], mec='k', mfc=colors_old[i], label=labels[i] if i != check_label else "")
#         # ax5.plot(phi_test, u_fluc/S_L0_test, ls=linestyle,  marker=markers[i], ms=markersize[i], mec='k', mfc=colors_old[i], label=labels[i] if i != check_label else "")
        
#         ax_old.plot(phi_test, U_bulk_test, ls=linestyle,  marker=markers[i], ms=markersize[i], mec='k', mfc=colors[i])
#         ax1.plot(phi_test, U_bulk_test, ls=linestyle,  marker=markers[i], ms=markersize[i], mec='k', mfc=colors[i])
#         ax2.plot(phi_test, U_bulk_test/S_L0_test, ls=linestyle,  marker=markers[i], ms=markersize[i], mec='k', mfc=colors[i])
#         ax3.plot(phi_test, Re_D_test, ls=linestyle,  marker=markers[i], ms=markersize[i], mec='k', mfc=colors[i])
#         ax4.plot(phi_test, S_L0_test, ls=linestyle,  marker=markers[i], ms=markersize[i], mec='k', mfc=colors[i])
#         ax5.plot(phi_test, u_fluc/S_L0_test, ls=linestyle,  marker=markers[i], ms=markersize[i], mec='k', mfc=colors[i])
        
#         check_label = i
        
# #% Create polynomial fits
# lists_zipped1 = zip(phi_lists1, U_bulk_lists1, Re_D_lists1, S_L0_lists1, colors)

# poly_order = 3
# for (phi, U_bulk, Re_D, S_L0, color) in lists_zipped1:
    
#     phi_fit = np.linspace(min(phi), max(phi))
    
#     poly_U_bulk = np.poly1d(np.polyfit(phi, U_bulk, poly_order))
#     U_bulk_fit = poly_U_bulk(phi_fit)
#     ax_old.plot(phi_fit, U_bulk_fit, ls="--", c=color)
    
#     poly_U_bulk = np.poly1d(np.polyfit(phi, U_bulk, poly_order))
#     U_bulk_fit = poly_U_bulk(phi_fit)
#     ax1.plot(phi_fit, U_bulk_fit, ls="--", c=color)
    
#     poly_U_bulk_S_L0 = np.poly1d(np.polyfit(phi, U_bulk/S_L0, poly_order))
#     U_bulk_S_L0_fit = poly_U_bulk_S_L0(phi_fit)
#     ax2.plot(phi_fit, U_bulk_S_L0_fit, ls="--", c=color)
    
#     poly_Re_D = np.poly1d(np.polyfit(phi, Re_D, poly_order))
#     Re_D_fit = poly_Re_D(phi_fit)
#     ax3.plot(phi_fit, Re_D_fit, ls="--", c=color)
    
#     poly_S_L0 = np.poly1d(np.polyfit(phi, S_L0, poly_order))
#     S_L0_fit = poly_S_L0(phi_fit)
#     ax4.plot(phi_fit, S_L0_fit, ls="--", c=color)

# QUARTZ
for i in range(n):
    lenght_j = len(U_bulk_lists2[i])
    for j in range(lenght_j):

        phi_test = phi_lists2[i][j]
        U_bulk_test = U_bulk_lists2[i][j]
        
        flame_test = flame_lists2[i][j]
        Re_D_test = D_quartz*U_bulk_test/flame_test.nu_u
        S_L0_test = flame_test.S_L0
        u_fluc = u_flux_max_lists2[i][j]
        
        S_T_test = S_L0_test*(1 + u_fluc/S_L0_test)
        
        ax_old.plot(phi_test, U_bulk_test, ls=linestyle,  marker=markers[i], ms=markersize[i], mec='k', mfc=colors[i], label=labels[i] if i != check_label else "")
        ax1.plot(phi_test, U_bulk_test, ls=linestyle,  marker=markers[i], ms=markersize[i], mec='k', mfc=colors[i], label=labels[i] if i != check_label else "")
        ax2.plot(phi_test, U_bulk_test/S_L0_test, ls=linestyle,  marker=markers[i], ms=markersize[i], mec='k', mfc=colors[i], label=labels[i] if i != check_label else "")
        ax3.plot(phi_test, Re_D_test, ls=linestyle,  marker=markers[i], ms=markersize[i], mec='k', mfc=colors[i], label=labels[i] if i != check_label else "")
        ax4.plot(phi_test, S_L0_test, ls=linestyle,  marker=markers[i], ms=markersize[i], mec='k', mfc=colors[i], label=labels[i] if i != check_label else "")
        ax5.plot(phi_test, u_fluc/S_L0_test, ls=linestyle,  marker=markers[i], ms=markersize[i], mec='k', mfc=colors[i], label=labels[i] if i != check_label else "")
        
        check_label = i
        
#% Create polynomial fits
lists_zipped2 = zip(phi_lists2, U_bulk_lists2, Re_D_lists2, S_L0_lists2, colors)

poly_order = 3
for (phi, U_bulk, Re_D, S_L0, color) in lists_zipped2:
    
    phi_fit = np.linspace(min(phi), max(phi))
    
    poly_U_bulk = np.poly1d(np.polyfit(phi, U_bulk, poly_order))
    U_bulk_fit = poly_U_bulk(phi_fit)
    ax_old.plot(phi_fit, U_bulk_fit, ls="--", c=color)
    
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
    
# Add critical Reynolds number (laminar to turbulent) to figure   
# ax3.axhline(2300, ls='-.', lw=2, c='k', label='$Re_{D, cr}=3000$')
    
# temperature and pressure of unburnt mixture
T_u = flame.T_u
p_u = flame.p_u

# set limits
ax_old.set_xlim(0.3, 1.1)
ax_old.set_ylim(0, 25)
ax1.set_ylim(0, 25)

# Add titles to figures
ax1.set_title('Flashback propensity map [$U_{bulk}$] of $H_{2}$-DNG/air mixtures \n $T_u=$' + str(round(T_u,2)) + ' K, $p_u$=' + str(p_u*1e-5) + ' bar') 
# ax2.set_title('$U_{bulk}$ normalized by $S_{L,0}$ \n $T_u=$' + str(round(T_u,2)) + ' K, $p_u$=' + str(p_u*1e-5) + ' bar') 
# ax3.set_title('Reynolds number based on $U_{bulk}$ \n $T_u=$' + str(round(T_u,2)) + ' K, $p_u$=' + str(p_u*1e-5) + ' bar') 
ax4.set_title('Unstretched laminar flame speed of $H_{2}$-DNG/air mixtures \n $T_u=$' + str(round(T_u,2)) + ' K, $p_u$=' + str(p_u*1e-5) + ' bar')

# Turn on grids
ax_old.grid()
ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()

# Show legends in figures
ax_old.legend(loc='upper left', prop={'size': 12})
# ax_old.legend(bbox_to_anchor=(1., 1.025), prop={'size': 12})# ax3.legend(loc='upper left', prop={'size': 12})
ax1.legend(loc='upper left', prop={'size': 12})
ax2.legend(loc='upper right', prop={'size': 12})
ax3.legend(loc='upper left', prop={'size': 12})
# ax3.legend(bbox_to_anchor=(1., 1.025), prop={'size': 12})# ax3.legend(loc='upper left', prop={'size': 12})
ax4.legend(loc='upper left', prop={'size': 12})

# Adjust the padding between and around subplots
fig_old.tight_layout() 
fig1.tight_layout() 
fig2.tight_layout() 
fig3.tight_layout() 
fig4.tight_layout() 

# # Save figures as vector files (.svg)
# fig_old.savefig('Figure_1_final.svg')
# fig2.savefig('phi_U_bulk_normalized.svg')
# fig3.savefig('phi_Re_D.svg')
# fig4.savefig('phi_S_L0.svg')











