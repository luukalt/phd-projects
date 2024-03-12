# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 08:25:24 2023

@author: laaltenburg

[A] Conservation of mass
A1: mass conservation in axial direction
A2: mass conservation in radial direction

[B] Conservation of momentum [Axial]
B1: Advection of U_x in axial direction
B2: Advection of U_x in radial direction
B3: Viscous term
B4: Reynolds normal stress term
B5: Reynolds shear stress term

[C] Conservation of momentum [Radial]  
C1: Advection of U_r in axial direction
C2: Advection of U_r in radial direction
C3: Viscous term
C4: Reynolds shear stress term
C5: Reynolds normal stress term

# axial direction: axis=0
# radial direction: axis=1
"""

#%% IMPORT PACKAGES
import os
import sys

# Add the 'main' folder to sys.path
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
flame_front_detection_directory = os.path.abspath(os.path.join(parent_folder, 'flame_front_detection'))
flame_simulations_directory = os.path.abspath(os.path.join(parent_folder, 'flame_simulations'))
plot_parameters_directory = os.path.abspath(os.path.join(parent_folder, 'plot_parameters'))

# Add the flame_object_directory to sys.path
sys.path.append(parent_folder)
sys.path.append(flame_front_detection_directory)
sys.path.append(flame_simulations_directory)
sys.path.append(plot_parameters_directory)

import pickle
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scipy.interpolate import griddata
from scipy.interpolate import interp1d

from custom_colormaps import parula
from premixed_flame_properties import PremixedFlame
from cone_angle import cone_angle
from intersect import intersection

from nonreact_flow_fields import non_react_dict
from favre_averaging import process_df

import flame_object
import premixed_flame_properties
import rc_params_settings

# # from parameters import set_mpl_params
# from parameters_ls import r_range_left, r_range_right, poly_left_fit, poly_right_fit
    
#%% FUNCTIONS

def plot_streamlines_reacting_flow(r_uniform, x_uniform, u_r_uniform, u_x_uniform):
    
    strms = []
    streamlines = []
    flame_front_indices = []
    paths = []
    
    fig1, ax1 = plt.subplots()
    
    # Cone angle
    distances_above_tube = [.25, .75, 1.25,]
    r_range_left, poly_left_fit, r_range_right, poly_right_fit, alpha = cone_angle(spydata_dir, name, distances_above_tube)

    ax1.plot(r_range_left, poly_left_fit, c='k', ls='dashed')
    ax1.plot(r_range_right, poly_right_fit, c='k', ls='dashed')
    
    colors = []
    
    for (r_start, x_start) in start_points:
        
        start_point = np.array([[r_start, x_start]])
        strm = ax1.streamplot(r_uniform, x_uniform, u_r_uniform, u_x_uniform, linewidth=1,
                              start_points=start_point, 
                              # broken_streamlines=False, 
                              density=1,
                              )
        
        skip = 2
        ax1.quiver(r_uniform[::skip], x_uniform[::skip], u_r_uniform[::skip], u_x_uniform[::skip], angles='xy', scale_units='xy', scale=20, width=0.005, color='grey')
        
        colors.append(strm.lines.get_color())
        strms.append(strm)
       
    for c, strm in enumerate(strms):
        
        segments = strm.lines.get_segments()
        line = np.array([segment[0] for segment in segments[:-1]] + [segments[-1][1]])
        
        line_r, line_x = line[:,0], line[:,1]
        distance = np.cumsum(np.sqrt(np.ediff1d(line_r, to_begin=0)**2 + np.ediff1d(line_x, to_begin=0)**2))
        
        distance = distance/distance[-1]
        
        
        fr, fx = interp1d(distance, line_r), interp1d(distance, line_x)
        
        n_interp_points = np.linspace(0, 1, 501)
        r_interp, x_interp = fr(n_interp_points), fx(n_interp_points)
        
        line = np.column_stack((r_interp, x_interp))
        
        # Filter line points where x > x_norm_min
        streamline_r_max = .5
        streamline_x_max = 2.2
        # streamline = line[(np.abs(line[:,0]) < streamline_r_max) & (line[:,1] > 0.05) & (line[:,1] < streamline_x_max)]
        streamline = line[(np.abs(r_interp) < streamline_r_max) & (x_interp >= streamline_x_start) & (x_interp < streamline_x_max)]
        
        streamline_r, streamline_x = streamline[:,0], streamline[:,1]
        streamline = np.column_stack((streamline_r, streamline_x))
        
        ax1.plot(streamline_r, streamline_x, color=colors[c], marker='None', ls='-', lw=2)
        ax1.plot(streamline_r[0], streamline_x[0], color=colors[c], marker='^', ls='None', mec='k', ms=ms5)
        ax1.plot(streamline_r[-1], streamline_x[-1], color=colors[c], marker='o', ls='None', mec='k', ms=ms5)
        
        # ax1.set_title(name)
        fontsize = 20
        ax1.set_xlabel(r'$r/D$', fontsize=fontsize)
        ax1.set_ylabel(r'$x/D$', fontsize=fontsize)
        
        ax1.tick_params(axis='both', labelsize=fontsize)
        
        # ax1.set_ylabel(cbar_titles[i], rotation=0, labelpad=15, fontsize=14)
        ax1.grid(False)
        ax1.set_xlim([-.55, .55])  # replace with your desired x limits
        ax1.set_ylim([0., 2.25])  # replace with your desired x limits
        # ax1.legend()
        ax1.set_aspect('equal')
        
        # Find intersecion between streamline and average cone angle
        
        if streamline_r[0] < 0:
            r_intersect, x_intersect = intersection(r_range_left, poly_left_fit, streamline_r, streamline_x)

        else:
            r_intersect, x_intersect = intersection(r_range_right, poly_right_fit, streamline_r, streamline_x)
        
        intersect_point = np.array([r_intersect[0], x_intersect[0]])
        
        # Mask out y-values in y2_interp that are greater than x_intersection
        valid_indices = np.where(streamline_x <= x_intersect)[0]
        
        streamline_point_difference = streamline[valid_indices] - intersect_point
        distances_to_front = np.linalg.norm(streamline_point_difference, axis=1)
        
        # Get the index of the minimum distance
        min_distance_idx = valid_indices[np.argmin(distances_to_front)]
        
        # Retrieve the closest coordinates
        r_closest, x_closest = streamline[min_distance_idx]
        # y_closest = streamline_x[min_distance_idx]
        
        ax1.plot(r_intersect, x_intersect, marker='*', ms=ms2, c=colors[c], mec='k')
        # ax1.plot(r_closest, x_closest, 'sr')
        
        # Compute the distances between consecutive points on the line
        distances = np.sqrt(np.sum(np.diff(streamline, axis=0)**2, axis=1))
        
        # Compute the cumulative distances along the line
        cumulative_distances = np.concatenate(([0], np.cumsum(distances)))
        
        flame_front_indices.append(min_distance_idx)
        streamlines.append(streamline)
        paths.append(cumulative_distances)
    
    # ax1.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=1, prop={"size": 12})
            
    return streamlines, paths, flame_front_indices, colors    


def fans_comp_terms(df, flame):
    
    """
    Parameters:
    df_piv (pandas.DataFrame): DataFrame containing velocity data.
    D (float): Diameter of the pipe [mm].
    U (float): Bulk velocity of the fluid [m/s].
    Re_D (float): Reynolds number (Re) based on the diameter D.
    """
    
    # Kinematic viscosity
    D = flame.D_in*1e-3
    U = flame.u_bulk_measured
    Re_D = flame.Re_D
    rho_u = flame.properties.rho_u
    
    nu = U*D/Re_D
    
    # Get the column headers
    headers = df.columns
    u_r_norm = pd.pivot_table(df, values='Velocity u [m/s]', index='y_shift_norm', columns='x_shift_norm')/U
    
    # Create x-y meshgrid
    r_norm_array = u_r_norm.columns
    x_norm_array = u_r_norm.index
    r_norm, x_norm = np.meshgrid(r_norm_array, x_norm_array)
    
    var_index = 'y_shift [m]'
    var_column = 'x_shift [m]'
    
    # rho = pd.pivot_table(df, values='rho [kg/m^3]', index=var_index, columns=var_column)
    rho = pd.pivot_table(df, values='rho [kg/m^3]', index=var_index, columns=var_column)
    
    u_r = pd.pivot_table(df, values='Velocity u [m/s]', index=var_index, columns=var_column)
    u_x = pd.pivot_table(df, values='Velocity v [m/s]', index=var_index, columns=var_column)
    
    u_r_tilde = pd.pivot_table(df, values='u_favre [m/s]', index=var_index, columns=var_column)
    u_x_tilde = pd.pivot_table(df, values='v_favre [m/s]', index=var_index, columns=var_column)

    R_rr = pd.pivot_table(df, values='rho*u_fluc_favre*u_fluc_favre', index=var_index, columns=var_column)
    R_rx = pd.pivot_table(df, values='rho*u_fluc_favre*v_fluc_favre', index=var_index, columns=var_column)
    R_xx = pd.pivot_table(df, values='rho*v_fluc_favre*v_fluc_favre', index=var_index, columns=var_column)
    
    # Create x-y meshgrid
    r_array = u_r_tilde.columns
    x_array = u_r_tilde.index
    r, x = np.meshgrid(r_array, x_array)
    
    dr = np.mean(np.diff(r_array))
    dx = np.mean(np.diff(x_array))
    
    # [A] Conservation of mass
    drho_u_xdx = np.gradient(rho*u_x_tilde, x_array, axis=0)
    A1 = drho_u_xdx #np.gradient(u_x, x_array, axis=0)
    A2 = (1/r)*np.gradient(r*rho*u_r_tilde, r_array, axis=1)
    
    # [B] Conservation of momentum [Axial]
    drho_u_x_u_xdx = np.gradient(rho*u_x_tilde*u_x_tilde, x_array, axis=0)
    drho_u_x_u_rdr = (1/r)*np.gradient(r*rho*u_x_tilde*u_r_tilde, r_array, axis=1)
    B1 = drho_u_x_u_xdx
    B2 = drho_u_x_u_rdr
    
    # Assumption of continuity equation = 0
    rho_u_x_du_xdx = rho*u_x_tilde*np.gradient(u_x_tilde, x_array, axis=0)
    rho_u_r_du_xdr = rho*u_r_tilde*np.gradient(u_x_tilde, r_array, axis=1)
    B1 = rho_u_x_du_xdx
    B2 = rho_u_r_du_xdr
    
    # B3 = np.zeros(A1.shape)
    du_xdx = np.gradient(u_x_tilde, x_array, axis=0)
    du_xdr = np.gradient(u_x_tilde, r_array, axis=1)
    B3 = nu*(np.gradient(du_xdx, x_array, axis=0) + (1/r)*np.gradient(r*du_xdr, r_array, axis=1))
    B4 = np.gradient(R_xx, x_array, axis=0)
    B5 = (1/r)*np.gradient(r*R_rx, r_array, axis=1)
    
    
    # [C] Conservation of momentum [Radial]
    drho_u_x_u_rdx = np.gradient(rho*u_x_tilde*u_r_tilde, x_array, axis=0)
    drho_u_r_u_rdr = (1/r)*np.gradient(r*rho*u_r_tilde*u_r_tilde, r_array, axis=1)
    C1 = drho_u_x_u_rdx
    C2 = drho_u_r_u_rdr
    
    # Assumption of continuity equation = 0
    rho_u_x_du_rdx = rho*u_x_tilde*np.gradient(u_r_tilde, x_array, axis=0)
    rho_u_r_du_rdr = rho*u_r_tilde*np.gradient(u_r_tilde, r_array, axis=1)
    C1 = rho_u_x_du_rdx
    C2 = rho_u_r_du_rdr
    
    # C3 = np.zeros(A1.shape)
    du_rdx = np.gradient(u_r_tilde, x_array, axis=0)
    du_rdr = np.gradient(u_r_tilde, r_array, axis=1)
    C3 = nu*(np.gradient(du_rdx, x_array, axis=0) + (1/r)*np.gradient(r*du_rdr, r_array, axis=1) - u_r/(r**2))
    C4 = np.gradient(R_rx, x_array, axis=0)
    C5 = (1/r)*np.gradient(r*R_rr, r_array, axis=1)
    
    dpdx = -B1 - B2 - B3 - B4 - B5 
    dpdr = -C1 - C2 - C3 - C4 - C5
    
    norm_mass = rho_u*U/D
    norm_mom = rho_u*(U**2)/D
    
    # mass_cons = [A1+A2, A1, A2]
    mass_cons = [A1+A2]
    
    mom_x = [B1, B2, dpdx, B4, B5]
    mom_r = [C1, C2, dpdr, C4, C5]
    
    mass_cons = [pd.DataFrame(data=term, index=x_norm_array, columns=r_norm_array)/norm_mass if isinstance(term, np.ndarray) else pd.DataFrame(data=term.values, index=x_norm_array, columns=r_norm_array)/norm_mass for term in mass_cons]
    mom_x = [pd.DataFrame(data=term, index=x_norm_array, columns=r_norm_array)/norm_mom if isinstance(term, np.ndarray) else pd.DataFrame(data=term.values, index=x_norm_array, columns=r_norm_array)/norm_mom for term in mom_x]
    mom_r = [pd.DataFrame(data=term, index=x_norm_array, columns=r_norm_array)/norm_mom if isinstance(term, np.ndarray) else pd.DataFrame(data=term.values, index=x_norm_array, columns=r_norm_array)/norm_mom for term in mom_r]
    
    return mass_cons, mom_x, mom_r


def plot_mass_cons(ax, mass_cons, r_norm_values, x_norm_values, lines, flame_front_indices, colors):
    
    # colors = ['r', 'g', 'b']
    # fig1, ax1 = plt.subplots()
    # ax1.set_title('Conservation of mass')
    ax.grid(True)
    # ax.set_ylim([-.25, 8.])  # replace with your desired x limits
    
    x_label = r'$s/D$'
    y_label = r'$\mathbf{\nabla^*} \cdot \overline{\rho^*\mathbf{u^{*}}}$'
    
    export_data = {
        'cumulative_distances': [],
        'term_along_line': [],
        'flame_front_index': []
    }
    
    # First mass conservation
    for line, flame_front_index, color in zip(lines, flame_front_indices, colors):
        
        for j, terms in enumerate([mass_cons]):
            
            for k, term in enumerate(terms):
                
                # Compute the distances between consecutive points on the line
                distances = np.sqrt(np.sum(np.diff(line, axis=0)**2, axis=1))
                
                # Compute the cumulative distances along the line
                cumulative_distances = np.concatenate(([0], np.cumsum(distances)))
                
                term_values = term.values.flatten()   
                term_along_line = griddata((r_norm_values, x_norm_values), term_values, line, method=interpolation_method)
                
                ax.plot(cumulative_distances[:], term_along_line[:], ls='solid', marker='None', label=k)
                
                ax.plot(cumulative_distances[flame_front_index], term_along_line[flame_front_index], c=color, marker='*', ms=ms1, mec='k')
                ax.plot(cumulative_distances[0], term_along_line[0], c=color, marker='>', ms=ms4, mec='k')
                ax.plot(cumulative_distances[-1], term_along_line[-1], c=color, marker='o', ms=ms5, mec='k')
                
                ax.set_xlabel(x_label, fontsize=20)
                ax.set_ylabel(y_label, fontsize=20) 
                
                # Append data to each key in export_data
                export_data['cumulative_distances'].append(cumulative_distances.tolist())
                export_data['term_along_line'].append(term_along_line.tolist())
                export_data['flame_front_index'].append(flame_front_index)
    
    # Exporting the data
    with open('exported_data.pkl', 'wb') as file:
        pickle.dump(export_data, file)
        
    # marker_dummy1 = Line2D([0], [0], label='Average flame front location', marker='*', markersize=ms1, mec='k', mfc='None', linestyle='')
    
    # access legend objects automatically created from data
    # handles, labels = ax.get_legend_handles_labels()
    
    # add manual symbols to auto legend
    # handles.extend([marker_dummy1])
    
    # ax.legend(handles=handles)
    
    return ax


def plot_fans_terms(mass_cons, mom_x, mom_r, r_norm_values, x_norm_values, lines, flame_front_indices, colors):
    
    width, height = 6, 6
    fig, ax = plt.subplots(figsize=(width, height))
    ax.grid(True)
    
    x_label = r'$s/D$'
    y_label = r'$\mathbf{\nabla^*} \cdot \overline{\rho^*\mathbf{u^{*}}}$'
    
    # First mass conservation
    for line, flame_front_index, color in zip(lines, flame_front_indices, colors):
        
        for j, terms in enumerate([mass_cons]):
            
            for k, term in enumerate(terms):
                
                # Compute the distances between consecutive points on the line
                distances = np.sqrt(np.sum(np.diff(line, axis=0)**2, axis=1))
                
                # Compute the cumulative distances along the line
                cumulative_distances = np.concatenate(([0], np.cumsum(distances)))
                
                term_values = term.values.flatten()   
                term_along_line = griddata((r_norm_values, x_norm_values), term_values, line, method=interpolation_method)
                
                ax.plot(cumulative_distances[:], term_along_line[:], ls='solid', marker='None', label=k)
                
                ax.plot(cumulative_distances[flame_front_index], term_along_line[flame_front_index], c=color, marker='*', ms=ms1, mec='k')
                ax.plot(cumulative_distances[0], term_along_line[0], c=color, marker='>', ls='None', mec='k', ms=ms4)
                ax.plot(cumulative_distances[-1], term_along_line[-1], c=color, marker='o', ms=ms5, mec='k')
                
                ax.set_xlabel(x_label, fontsize=20)
                ax.set_ylabel(y_label, fontsize=20) 
    
    marker_dummy1 = Line2D([0], [0], label='Average flame front location', marker='*', markersize=ms1, 
         mec='k', mfc='None', linestyle='')
    
    # access legend objects automatically created from data
    handles, labels = ax.get_legend_handles_labels()
    
    # add manual symbols to auto legend
    handles.extend([marker_dummy1])
    
    ax.legend(handles=handles)
    
    # Second momentum equations
    mom_x_labels = [
                    '[1] Axial advection',
                    '[2] Radial advection',
                    '[3] Pressure gradient',
                    # '[4] Viscous diffusion',
                    '[4] Reynolds normal stress',
                    '[5] Reynolds shear stress'
                    ]
    
    mom_r_labels = [
                    '[1] Axial advection',
                    '[2] Radial advection',
                    '[3] Pressure gradient',
                    # '[4] Viscous diffusion',
                    '[4] Reynolds shear stress',
                    '[5] Reynolds normal stress'
                    ]
    
    mom_markers = ['v', '^', 'o', 's', 'p', 'd']
    mom_markers = ['None', 'None', 'None', 'None', 'None', 'None']
    
    
    linestyles = ['-', '-', '--', '-', '-', '-']
    
    colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    fontsize_label = 24
    
    for line, flame_front_index, color in zip(lines, flame_front_indices, colors):
        
        width, height = 9, 9
        fig2, ax2 = plt.subplots(figsize=(width, height))
        # fig2, ax2 = plt.subplots()
        
        # ax2.set_title('Conservation of axial momentum for streamline ' + str(round(line[0][0], 1)))
        ax2.grid(True)
        ax2.set_xlabel(x_label, fontsize=fontsize_label)
        ax2.set_ylabel(r'Dimensionless terms', fontsize=fontsize_label)
        
        fig3, ax3 = plt.subplots(figsize=(width, height))
        # ax3.set_title('Conservation of radial momentum r=' + str(round(line[0][0], 1)))
        ax3.grid(True)
        ax3.set_xlabel(x_label, fontsize=fontsize_label)
        ax3.set_ylabel(r'Dimensionless terms', fontsize=fontsize_label)
        
        bandwith = .35
        ax2.set_ylim([-2*bandwith, bandwith])  # replace with your desired y limits
        ax3.set_ylim([-2*bandwith, bandwith])  # replace with your desired y limits
        
        for j, terms in enumerate([mom_x, mom_r]):
            
            for k, term in enumerate(terms):
                
                # Compute the distances between consecutive points on the line
                distances = np.sqrt(np.sum(np.diff(line, axis=0)**2, axis=1))
                
                # Compute the cumulative distances along the line
                cumulative_distances = np.concatenate(([0], np.cumsum(distances)))
                
                term_values = term.values.flatten()   
                term_along_line = griddata((r_norm_values, x_norm_values), term_values, line, method=interpolation_method)
                
                color, linestyle, mom_marker = map(lambda x: x[k], [colors, linestyles, mom_markers])

                if j == 0:
                    
                    ax2.plot(cumulative_distances, term_along_line, c=colors[k], ls=linestyles[k], marker=mom_markers[k], label=mom_x_labels[k], lw=2)
                    ax2.plot(cumulative_distances[flame_front_index], term_along_line[flame_front_index], c=color, marker='*', ms=ms1, mec='k')
                    ax2.plot(cumulative_distances[0], term_along_line[0], c=color, marker='>', ls='None', mec='k', ms=ms4)
                    ax2.plot(cumulative_distances[-1], term_along_line[-1], c=color, marker='o', ms=ms5, mec='k')
                    
                if j == 1:
    
                    ax3.plot(cumulative_distances, term_along_line, c=colors[k], ls=linestyles[k], marker=mom_markers[k], ms=ms1, label=mom_r_labels[k], lw=2)
                    ax3.plot(cumulative_distances[flame_front_index], term_along_line[flame_front_index], c=color, marker='*', ms=ms1, mec='k')
                    ax3.plot(cumulative_distances[0], term_along_line[0], c=color, marker='>',  mec='k', ms=ms4)
                    ax3.plot(cumulative_distances[-1], term_along_line[-1], c=color, marker='o', ms=ms5, mec='k')
                    
        # ax2.legend(title='Axial terms', loc="upper left", bbox_to_anchor=(1, 1), ncol=1, prop={"size": 16})
        
        # ax3.legend(title='Radial terms', loc="upper left", bbox_to_anchor=(1, 1), ncol=1, prop={"size": 16})
        
        # ax2.legend(loc='upper left', prop={'size': 18}, bbox_to_anchor=(.225, .45))
        # ax2.legend(loc='upper left', prop={'size': 18}, bbox_to_anchor=(.0, .425))
        
        ax2.legend(loc='lower left', prop={'size': 18})
        ax3.legend(loc='lower left', bbox_to_anchor=(0, .1), prop={'size': 18})
        
        ax2.tick_params(axis='both', labelsize=20)
        ax3.tick_params(axis='both', labelsize=20)
        
        
def plot_pressure_along_streamline(dpdr, dpdx, r_norm_values, x_norm_values, lines, flame_front_indices, colors):
    
    fig1, ax1 = plt.subplots()
    
    fontsize_label = 24
    
    x_label = r'$s/D$'
    y_label = r'$\overline{p^*} - \overline{p_{0}^*}$'
    
    ax1.set_xlabel(x_label, fontsize=fontsize_label)
    ax1.set_ylabel(y_label, fontsize=fontsize_label)
    
    # Define the region of interest for zooming
    ax1_x1, ax1_x2, ax1_y1, ax1_y2 = -.1, .5, -.01, .06  # for example, zoom in on this region

    # Draw a rectangle or any other shape to indicate the zoom area
    ax1.add_patch(plt.Rectangle((ax1_x1, ax1_y1), ax1_x2 - ax1_x1, ax1_y2 - ax1_y1, fill=False, color='k', linestyle='solid', lw=2))
    
    if flame.Re_D == 3000:
        y_lims = [-.325, .125]
        bbox_to_anchor=(ax1_x1 + 1.725 , ax1_y1 + .055, ax1_x2 - ax1_x1, ax1_y2 - ax1_y1)
        width="225%"
        height="225%"
    elif flame.Re_D == 4000:
        y_lims = [-.325, .125]
        # bbox_to_anchor=(ax1_x1 + .15 , ax1_y1 -.225, ax1_x2 - ax1_x1, ax1_y2 - ax1_y1)
        bbox_to_anchor=(ax1_x1 + 1.725 , ax1_y1 + .055, ax1_x2 - ax1_x1, ax1_y2 - ax1_y1)
        width="225%"
        height="225%"
    elif flame.Re_D == 12500:
        y_lims = [-.12, .1]
        bbox_to_anchor=(ax1_x1 + 2., ax1_y1 + .035, ax1_x2 - ax1_x1, ax1_y2 - ax1_y1)
        width="175%"
        height="175%"
    elif flame.Re_D == 16000:
        y_lims = [-.12, .1]
        bbox_to_anchor=(ax1_x1 + 2., ax1_y1 + .035, ax1_x2 - ax1_x1, ax1_y2 - ax1_y1)
        width="175%"
        height="175%"
        
    ax1_inset = inset_axes(ax1, width=width, height=height, loc='upper left',
                      # bbox_to_anchor=(x1, y1, x2 - x1, y2 - y1),
                      bbox_to_anchor=bbox_to_anchor,
                      bbox_transform=ax1.transData,
                      borderpad=0)
    
    
    # Set the limits for the inset axes
    ax1_inset.set_xlim(ax1_x1, ax1_x2)
    ax1_inset.set_ylim(ax1_y1, ax1_y2)
    
    ax1_inset.set_xticks([.0, .2, .4])
    ax1_inset.set_yticks([.0, .02, .04, .06])
    
    # ax_inset = fig1.add_axes([.575, .55, .3, .3]) # x, y, width, height (in figure coordinate)
    
    styles_react = ['solid', 'solid', 'solid']
    styles_nonreact = ['dashed', 'dashed', 'dashed']
    
    for line, flame_front_index, color in zip(lines, flame_front_indices, colors):
        
        line_r, line_x = line[:,0], line[:,1]
        
        # Compute the distances between consecutive points on the line
        distances = np.sqrt(np.sum(np.diff(line, axis=0)**2, axis=1))
        
        # Compute the cumulative distances along the line
        cumulative_distances = np.concatenate(([0], np.cumsum(distances)))
          
        dpdr_values = dpdr.values.flatten()
        dpdx_values = dpdx.values.flatten()
        
        dpdr_along_line = griddata((r_norm_values, x_norm_values), dpdr_values, line, method=interpolation_method)
        dpdx_along_line = griddata((r_norm_values, x_norm_values), dpdx_values, line, method=interpolation_method)
        
        p_along_line = np.zeros(dpdr_along_line.shape[0])
    
        for i in range(1, dpdr_along_line.shape[0]):
            
            ds = cumulative_distances[i] - cumulative_distances[i-1]
            dr = line_r[i] - line_r[i-1]
            dx = line_x[i] - line_x[i-1]
            
            drds = dr/ds
            dxds = dx/ds
            
            
            # dr *= D_in*1e-3
            # dx *= D_in*1e-3
            
            p_along_line[i] = p_along_line[i-1] + 0.5 * (dpdr_along_line[i] + dpdr_along_line[i-1]) * dr + 0.5 * (dpdx_along_line[i] + dpdx_along_line[i-1]) * dx
            
        
        ax1.plot(cumulative_distances, p_along_line, c=color, marker='None', ls='solid')
        ax1.plot(cumulative_distances[flame_front_index], p_along_line[flame_front_index], c=color, marker='*', ms=ms1, mec='k')
        ax1.plot(cumulative_distances[0], p_along_line[0], color=color, marker='>', ls='None', mec='k', ms=ms4)
        ax1.plot(cumulative_distances[-1], p_along_line[-1], color=color, marker='o', ls='None', mec='k', ms=ms5)
        
        # Create an inset with zoomed-in plot
        ax1_inset.plot(cumulative_distances[0], p_along_line[0], color=color, marker='>', ls='None', mec='k', ms=ms4)
        ax1_inset.plot(cumulative_distances, p_along_line, c=color, marker='None', ls='solid')
        
        
    # make proxy artists
    # make list of one line -- doesn't matter what the coordinates are
    # dummy_line = [[(0, 0)]]
    # set up the proxy artist
    # lc_react = mcol.LineCollection(3 * dummy_line, linestyles=styles_react, colors=colors)
    # lc_nonreact = mcol.LineCollection(3 * dummy_line, linestyles=styles_nonreact, colors=colors)
    
    
    ax1.set_xlim(right=3)  # replace with your desired x limits
    ax1.set_ylim(y_lims)  # replace with your desired x limits
    ax1.grid(True)
    
    # ax_inset.set_xlim(x1, x2)
    # ax_inset.set_ylim(y1, y2)
    
    # Hide the x and y axis ticks
    # ax_inset.set_xticks([])
    # ax_inset.set_yticks([])
    
    ax1.tick_params(axis='both', labelsize=20)
    
    x_label = r'$s/D$'
    y_label = r'$\frac{dp^{*}}{ds^{*}}$'

    
    # # create the legend
    # if flame.Re_D == 4000:
    #     ax1.legend([lc_react, lc_nonreact], ['reacting', 'non reacting'], handler_map={type(lc_react): HandlerDashedLines()},
    #                 handlelength=3, handleheight=3, loc='upper left', prop={'size': 16}, bbox_to_anchor=(.45, .65))
    # elif flame.Re_D == 16000:
    #     ax1.legend([lc_react, lc_nonreact], ['reacting', 'non reacting'], handler_map={type(lc_react): HandlerDashedLines()},
    #                     handlelength=3, handleheight=3, loc='upper left', prop={'size': 16}, bbox_to_anchor=(.45, 1.))
        
    # ax2.legend([lc_react, lc_nonreact], ['reacting', 'non reacting'], handler_map={type(lc_react): HandlerDashedLines()},
    #            handlelength=3, handleheight=3)        

#%% MAIN
if __name__ == '__main__':
    
    # data_dir = 'U:\\High hydrogen\\laaltenburg\\data\\tube_burner_campaign2\\selected_runs\\'
    
    data_dir = 'U:\\staff-umbrella\\High hydrogen\\laaltenburg\\data\\tube_burner_campaign2\\selected_runs\\'
    
    frame_nr = 0
    segment_length_mm = 1 # units: mm
    window_size = 31 # units: pixels
    
    #%%% Define cases
    react_names_ls =    [
                        # ('react_h0_c3000_ls_record1', 57),
                        ('react_h0_s4000_ls_record1', 58),
                        # ('react_h100_c12000_ls_record1', 61),
                        # ('react_h100_c12500_ls_record1', 61),
                        # ('react_h100_s16000_ls_record1', 62)
                        ]
    
    react_names_hs =    [
                        # ('react_h0_f2700_hs_record1', 57),
                        # ('react_h0_c3000_hs_record1', 57),
                        # ('react_h0_s4000_hs_record1', 58),
                        # ('react_h100_c12500_hs_record1', 61),
                        # ('react_h100_s16000_hs_record1', 62)
                        ]
    
    if react_names_ls:
        spydata_dir = os.path.join(parent_folder, 'spydata\\udf')
    elif react_names_hs:
        spydata_dir = os.path.join(parent_folder, 'spydata')
    
    react_names = react_names_ls + react_names_hs
    
    # piv_method = 'PIV_MP(3x16x16_75%ov_ImgCorr)'
    piv_method = 'PIV_MP(3x16x16_0%ov_ImgCorr)'
    
    for name, nonreact_run_nr in react_names:
    
        fname = f'{name}_segment_length_{segment_length_mm}mm_wsize_{window_size}pixels'
    
        with open(os.path.join(spydata_dir, fname + '.pkl'), 'rb') as f:
            flame = pickle.load(f)
        
        name = flame.name
        session_nr = flame.session_nr
        recording = flame.record_name
        piv_method = piv_method
        run_nr = flame.run_nr
        Re_D_set = flame.Re_D
        u_bulk_set = flame.u_bulk_measured
        u_bulk_measured = flame.u_bulk_measured
        
        mixture = PremixedFlame(flame.phi, flame.H2_percentage, flame.T_lab, flame.p_lab)
        mixture.solve_equations()
        
        flame.properties.rho_b = mixture.rho_b

    # Save the mean DataFrame to a CSV file
    input_file_path = os.path.join(os.path.join(parent_folder, 'spydata'), flame.name, 'AvgFavreFinal.csv')
    
    df_favre_avg = pd.read_csv(input_file_path, index_col='index')
    headers = df_favre_avg.columns
    
    # bottom_limit = -.5
    # top_limit = 2.25
    # left_limit = -0.575
    # right_limit = 0.575
    
    bottom_limit = .05
    top_limit = 2.2
    left_limit = -100
    right_limit = 100 #0.575
                           
    index_name = 'y_shift_norm'
    column_name = 'x_shift_norm'
    
    df_favre_avg = df_favre_avg[(df_favre_avg[index_name] > bottom_limit) & (df_favre_avg[index_name] < top_limit) & (df_favre_avg[column_name] > left_limit) & (df_favre_avg[column_name] < right_limit)]
    
    df_favre_avg['Velocity |V| [m/s]'] = np.sqrt(df_favre_avg['Velocity u [m/s]']**2 + df_favre_avg['Velocity v [m/s]']**2)
    
    df_favre_avg['|V|_favre [m/s]'] = np.sqrt(df_favre_avg['u_favre [m/s]']**2 + df_favre_avg['v_favre [m/s]']**2)
    
    df_favre_avg['u_favre [counts] [m/s]'] = df_favre_avg['Wmean*u [counts]'].div(df_favre_avg['Wmean [counts]']).fillna(0)
    df_favre_avg['v_favre [counts] [m/s]'] = df_favre_avg['Wmean*v [counts]'].div(df_favre_avg['Wmean [counts]']).fillna(0)
    
    df_favre_avg['|V|_favre [counts] [m/s]'] = np.sqrt(df_favre_avg['u_favre [counts] [m/s]']**2 + df_favre_avg['v_favre [counts] [m/s]']**2)
    
    # var = 'rho [kg/m^3]'
    # var = 'v_favre [m/s]'
    # var = '0.5*(R_uu + R_vv) [m^2/s^2]'
    # var = 'Velocity v [m/s]'
    # var = 'test'
    
    # var = 'Velocity |V| [m/s]'
    # var = '|V|_favre [m/s]'
    var = '|V|_favre [counts] [m/s]'
    
    # var = 'Wmean [counts]'
    # var_counts = 'Wmean [counts]'
    var_counts_norm = 'Wmean_norm [counts]'
    
    pivot_var = pd.pivot_table(df_favre_avg, values=var, index=index_name, columns=column_name)

    # Create x-y meshgrid
    r_norm_array = pivot_var.columns
    x_norm_array = pivot_var.index
    r_norm, x_norm = np.meshgrid(r_norm_array, x_norm_array)
    r_norm_values = r_norm.flatten()
    x_norm_values = x_norm.flatten()
    
    fig, ax = plt.subplots()
    ax.set_title(var)
    colormap = parula
    flow_field = ax.pcolor(r_norm, x_norm, pivot_var.values/(u_bulk_measured**1), cmap=colormap, vmin=0, vmax=1.6)
    cbar = ax.figure.colorbar(flow_field)
    
    # Overlay scatter plots for X and Y values
    # Define the specific values you want to highlight
    value_X = 1
    value_Y = 0
    
    # Initialize lists to store data
    coords_X_x, coords_X_y, counts_X, values_X = [], [], [], []
    coords_Y_x, coords_Y_y, counts_Y, values_Y = [], [], [], []

    
    # Iterate through the DataFrame to find matching rows
    for index, row in df_favre_avg.iterrows():
        
        coord_r, coord_x = row['x_shift_norm'], row['y_shift_norm']
        
        if coord_x > 0.1:
            
            if row[var] == value_X:
                coords_X_x.append(row[column_name])
                coords_X_y.append(row[index_name])
                counts_X.append(row[var_counts_norm])
                values_X.append(row[var])
                
            elif row[var] == value_Y:
                coords_Y_x.append(row[column_name])
                coords_Y_y.append(row[index_name])
                counts_Y.append(row[var_counts_norm])
                values_Y.append(row[var])
            
    # Overlay scatter plots for X and Y values
    # ax.scatter(coords_X_x, coords_X_y, color='black', label=f'Value = {value_X}')
    # ax.scatter(coords_Y_x, coords_Y_y, color='red', label=f'Value = {value_Y}')
    
    # print(np.mean(counts_X)/np.mean(counts_Y))
    # print(flame.properties.rho_u/flame.properties.rho_b)
    
    fig2, ax2 = plt.subplots()
    # Scatter plot for counts_X and counts_Y
    ax2.scatter(values_X, counts_X, color='black', label=f'Counts for Value = {value_X}')
    ax2.scatter(values_Y, counts_Y, color='red', label=f'Counts for Value = {value_Y}')
    
    # Adding labels and title
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Counts')
    ax2.set_title('Counts as a Function of Value')
    ax2.legend()
    
    fontsize = 20
    ax.set_aspect('equal')
    ax.set_xlabel(r'$r/D$', fontsize=fontsize)
    ax.set_ylabel(r'$x/D$', fontsize=fontsize)
    
    # Flatten the meshgrid and pivot table values
    points = np.column_stack((r_norm.flatten(), x_norm.flatten()))  # (r, x) coordinate pairs
    values = pivot_var.values.flatten()  # Corresponding values at each (r, x)
    
    # Create a uniform grid
    r_uniform = np.linspace(r_norm_values.min(), r_norm_values.max(), len(r_norm_array))
    x_uniform = np.linspace(x_norm_values.min(), x_norm_values.max(), len(x_norm_array))
    r_uniform, x_uniform = np.meshgrid(r_uniform, x_uniform)
    
    pivot_u_r = pd.pivot_table(df_favre_avg, values='Velocity u [m/s]', index=index_name, columns=column_name)
    pivot_u_x = pd.pivot_table(df_favre_avg, values='Velocity v [m/s]', index=index_name, columns=column_name)
    
    pivot_u_r_norm = pivot_u_r/u_bulk_measured
    pivot_u_x_norm = pivot_u_x/u_bulk_measured
    
    pivot_u_r_norm_values = pivot_u_r_norm.values.flatten()
    pivot_u_x_norm_values = pivot_u_x_norm.values.flatten()
    
    #%%% Interpolation method
    # interpolation_method = 'nearest'
    interpolation_method = 'linear'
    
    # Interpolate the velocity components to the uniform grid
    u_r_uniform = griddata((r_norm_values, x_norm_values), pivot_u_r_norm_values, (r_uniform, x_uniform), method=interpolation_method)
    u_x_uniform = griddata((r_norm_values, x_norm_values), pivot_u_x_norm_values, (r_uniform, x_uniform), method=interpolation_method)
    
    
    # # Point where you want to interpolate
    # streamline = streamlines[0]
    # point_of_interest = streamline[0] #np.array([[.1, .1]])  # (r=0, x=0)
    
    # # Perform interpolation
    # interpolated_value = griddata(points, values, point_of_interest, method=interpolation_method)
    
    # # Check if interpolation returned a valid result
    # if interpolated_value.size > 0 and not np.isnan(interpolated_value[0]):
    #     print(f"Interpolated value at r={point_of_interest[0]}, x={point_of_interest[1]}: {var}={interpolated_value[0]}")
    # else:
    #     print("Interpolation at r={point_of_interest[0]}, x={point_of_interest[1]} is not possible with the given data.")
    
    mass_cons, mom_x, mom_r = fans_comp_terms(df_favre_avg, flame)
    
    # Marker sizes
    ms1 = 18
    ms2 = 16
    ms3 = 14
    ms4 = 12
    ms5 = 10
    ms6 = 8
    
    r_starts = [.1, .2, .3]
    x_starts = np.linspace(0.2, 0.2, len(r_starts))
    start_points = [(r_starts[i], x_starts[i]) for i in range(len(r_starts))]
    
    streamline_x_start = .1
    
    ax.scatter(0, streamline_x_start, color='k', label=f'Value = {value_X}')
    
    streamlines, paths, flame_front_indices, colors = plot_streamlines_reacting_flow(r_uniform, x_uniform, u_r_uniform, u_x_uniform)
    
    fig, ax = plt.subplots()
    ax = plot_mass_cons(ax, mass_cons, r_norm_values, x_norm_values, streamlines, flame_front_indices, colors)
    plot_fans_terms(mass_cons, mom_x, mom_r, r_norm_values, x_norm_values, streamlines, flame_front_indices, colors)
    
    dpdx = mom_x[2] 
    dpdr = mom_r[2]
    plot_pressure_along_streamline(dpdr, dpdx, r_norm_values, x_norm_values, streamlines, flame_front_indices, colors)
    
    #%% Plot contour
    # from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter 

    # # Create a function to update the plot for each frame
    # def update(contour_nr):
    #     ax.clear()  # Clear the axes for each frame
        
    #     segmented_contour = flame.frames[frame_nr].contour_data.segmented_contours[contour_nr]
        
    #     segmented_contour_r = segmented_contour[:, 0, 0]
    #     segmented_contour_x = segmented_contour[:, 0, 1]
        
    #     ax.plot(segmented_contour_r, -segmented_contour_x, label=contour_nr + 1)
    #     ax.set_xlim(left=0, right=520)
    #     ax.set_ylim(top=0, bottom=-900)
    #     ax.legend()
    #     ax.set_aspect('equal')
    
    # # Set the number of frames
    # contour_nrs = 100 #len(flame.n_images)
    
    # # Create a figure and axis
    # fig, ax = plt.subplots()
    
    # # Create the animation
    # fps = 10
    # animation = FuncAnimation(fig, update, frames=contour_nrs, interval=1000/fps)  # You can adjust the interval as needed
    
    # # Save the animation as an MP4 file
    # writer = PillowWriter(fps=fps) 
    # animation.save('contour_animation.gif', writer=writer)
    
    #%% Non-reacting flow
    non_react_flow = non_react_dict[nonreact_run_nr]
    name = non_react_flow[0]
    session_nr = non_react_flow[1] 
    recording = non_react_flow[2]
    piv_method = non_react_flow[3]
    Re_D = non_react_flow[4]
    u_bulk_set = non_react_flow[5]
    u_bulk_measured = non_react_flow[6]
    
    Avg_Stdev_file = os.path.join(data_dir, f'session_{session_nr:03d}', recording, piv_method, 'Avg_Stdev', 'Export', 'B0001.csv')
    
    df_piv = pd.read_csv(Avg_Stdev_file)
    
    D_in = flame.D_in # Inner diameter of the quartz tube, units: mm
    offset = 1  # Distance from calibrated y=0 to tube burner rim
    
    wall_center_to_origin = 2
    wall_thickness = 1.5
    offset_to_wall_center = wall_center_to_origin - wall_thickness/2
    
    df_piv = process_df(df_piv, D_in, offset_to_wall_center, offset)
    
    # Get the column headers
    headers = df_piv.columns
    
    df_piv_cropped = df_piv[(df_piv[index_name] > bottom_limit) & (df_piv[index_name] < top_limit) & (df_piv[column_name] > left_limit) & (df_piv[column_name] < right_limit)]
    
    pivot_u_r = pd.pivot_table(df_piv_cropped, values='Velocity u [m/s]', index=index_name, columns=column_name)
    pivot_u_x = pd.pivot_table(df_piv_cropped, values='Velocity v [m/s]', index=index_name, columns=column_name)
    
    #%% Save images
    # Get a list of all currently opened figures
    # figure_ids = plt.get_fignums()
    # figure_ids = [14]
    
    # if react_names_ls:
    #     folder = 'ls'
    # else:
    #     folder = 'hs'

    # # Apply tight_layout to each figure
    # for fid in figure_ids:
    #     fig = plt.figure(fid)
    #     fig.tight_layout()
    #     # filename = f'H{flame.H2_percentage}_Re{flame.Re_D}_fig{fid}_favre'
    #     filename = f'H{flame.H2_percentage}_Re{flame.Re_D}_fig{fid}_favre'
        
    #     # Constructing the paths
    #     if fid == 1:
            
    #         png_path = os.path.join('figures', f'{folder}', f"{filename}.png")
    #         pkl_path = os.path.join('pickles', f'{folder}', f"{filename}.pkl")
            
    #         # Saving the figure in EPS format
    #         fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
            
    #     else:
            
    #         eps_path = os.path.join('figures', f'{folder}', f"{filename}.eps")
    #         pkl_path = os.path.join('pickles', f'{folder}', f"{filename}.pkl")
            
    #         # Saving the figure in EPS format
    #         fig.savefig(eps_path, format='eps', dpi=300, bbox_inches='tight')
        
        
        # # Pickling the figure
        # with open(pkl_path, 'wb') as f:
        #     pickle.dump(fig, f)
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    