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


#%% IMPORT STANDARD PACKAGES
import os
import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from scipy.interpolate import griddata

#%% IMPORT USER DEFINED PACKAGES
from sys_paths import parent_directory
import sys_paths
import rc_params_settings
from parameters import flame, piv_method, interpolation_method
from plot_params import colormap, fontsize
from time_averaging_terms import fans_terms
from plot_functions import plot_streamlines_reacting_flow, plot_mass_cons, plot_fans_terms, plot_pressure_along_streamline

figures_folder = 'figures'
if not os.path.exists(figures_folder):
        os.makedirs(figures_folder)
        
pickles_folder = 'pickles'
if not os.path.exists(pickles_folder):
        os.makedirs(pickles_folder)
        
#%% MAIN
if __name__ == '__main__':  
    
    name = flame.name
    session_nr = flame.session_nr
    recording = flame.record_name
    piv_method = piv_method
    run_nr = flame.run_nr
    Re_D_set = flame.Re_D
    u_bulk_set = flame.u_bulk_measured
    u_bulk_measured = flame.u_bulk_measured
    
    # Favre average directory
    favre_avg_file_path = os.path.join(os.path.join(parent_directory, 'spydata'), flame.name, 'AvgFavreFinal.csv')
    
    # Read the favre avrage file to dataframe
    df_favre_avg = pd.read_csv(favre_avg_file_path, index_col='index')
    
    # Get the column headers of the favre avrage file
    headers = df_favre_avg.columns
    
    # Non-dimensional limits in r- (left, right) and x-direction (bottom, top)
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
    
    # Cropped favre average dataframe based on non-dimensional limits in r- (left, right) and x-direction (bottom, top)
    df_favre_avg = df_favre_avg[(df_favre_avg[index_name] > bottom_limit) & (df_favre_avg[index_name] < top_limit) & (df_favre_avg[column_name] > left_limit) & (df_favre_avg[column_name] < right_limit)]
    
    # Add extra columns to the favre average dataframe
    df_favre_avg['Velocity |V| [m/s]'] = np.sqrt(df_favre_avg['Velocity u [m/s]']**2 + df_favre_avg['Velocity v [m/s]']**2)
    df_favre_avg['|V|_favre [m/s]'] = np.sqrt(df_favre_avg['u_favre [m/s]']**2 + df_favre_avg['v_favre [m/s]']**2)
    df_favre_avg['u_favre [counts] [m/s]'] = df_favre_avg['Wmean*u [counts]'].div(df_favre_avg['Wmean [counts]']).fillna(0)
    df_favre_avg['v_favre [counts] [m/s]'] = df_favre_avg['Wmean*v [counts]'].div(df_favre_avg['Wmean [counts]']).fillna(0)
    df_favre_avg['|V|_favre [counts] [m/s]'] = np.sqrt(df_favre_avg['u_favre [counts] [m/s]']**2 + df_favre_avg['v_favre [counts] [m/s]']**2)
    
    # var1 = 'Velocity |V| [m/s]'
    # var2 = '|V|_favre [counts] [m/s]'
    # var3 = '|V|_favre [m/s]'
    # var_list = [var1, var2, var3]
    labels = ['Reynolds',
              'Favre [intensity count]',
              'Favre [flame front detection]',
              ]
    cbar_titles = [r'$\frac{|\overline{V}|}{U_{b}}$',
                   r'$\frac{|\overline{V}|}{U_{b}}$',
                   r'$\frac{|\overline{V}|}{U_{b}}$',
                  ]
                                
               
    var1 = 'Wmean [counts]'
    var2 = 'Wmean [states]'
    var3 = 'rho [kg/m^3]'
    var_list = [var1, var2, var3]
    labels = [var1,
              var2,
              var3
              ]
    
    cbar_titles = [r'$\frac{\overline{I}}{\overline{I}_{max}}$',
                   r'State',
                   r'$\overline{\rho^{*}}$',
                  ]
    
    
    var_counts_norm = 'Wmean_norm [counts]'
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # fig2, ax2 = plt.subplots()
    
    width, height = 9, 6
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    
    # cbar_max = 2
    
    handles = []
    
    for color, var, label, cbar_title in zip(colors[:len(var_list)], var_list, labels, cbar_titles):
        
        pivot_var = pd.pivot_table(df_favre_avg, values=var, index=index_name, columns=column_name)

        # Create x-y meshgrid
        r_norm_array = pivot_var.columns
        x_norm_array = pivot_var.index
        r_norm, x_norm = np.meshgrid(r_norm_array, x_norm_array)
        r_norm_values = r_norm.flatten()
        x_norm_values = x_norm.flatten()
        
        fig, ax = plt.subplots(figsize=(width, height))
        # ax.set_title(label)
        
        cbar_min = np.min(df_favre_avg[var])
        
        cbar_max = np.max(df_favre_avg[var])
        
        pivot_var /= cbar_max
        
        flow_field = ax.pcolor(r_norm, x_norm, pivot_var.values, cmap=colormap, vmin=0, vmax=1)
        
        # num_ticks = 6
        # custom_cbar_ticks = np.linspace(0, cbar_max, num_ticks)
        
        # if cbar_max < 1:
        #     custom_cbar_tick_labels = [f'{tick:.2f}' for tick in custom_cbar_ticks] # Replace with your desired tick labels
        # else:
        #     custom_cbar_tick_labels = [f'{tick:.1f}' for tick in custom_cbar_ticks]
            
        cbar = ax.figure.colorbar(flow_field)
        # cbar.set_ticks(custom_cbar_ticks)
        # cbar.set_ticklabels(custom_cbar_tick_labels)
        if var == var2:
            cbar.set_label(cbar_title, fontsize=20)
        else:
            
            cbar.set_label(cbar_title, rotation=0, labelpad=25, fontsize=24) 
        # cbar.ax.tick_params(labelsize=fontsize)
        
        ax.set_aspect('equal')
        ax.set_xlabel(r'$r/D$', fontsize=fontsize)
        ax.set_ylabel(r'$x/D$', fontsize=fontsize)
        
        custom_y_ticks = [.5, 1., 1.5, 2.]
        ax.set_yticks(custom_y_ticks)
        
        ax.set_xlim(left=-.55, right=.55)
        ax.set_ylim(bottom=.05, top=2.2)
        
        ax.tick_params(axis='both', labelsize=fontsize)
        
        
        # Find the two closest indices to the given index
        distances_radial_tube = [.0, .3,]
        markers = ['o', '^']
        
        scatter_handles = []
        
        for marker, distance_radial_tube in zip(markers, distances_radial_tube):
            
            distance_radial_tube_right = pivot_var.columns[pivot_var.columns <= distance_radial_tube].max()
            distance_radial_tube_left = pivot_var.columns[pivot_var.columns >= distance_radial_tube].min()
            
            pivot_var_interp = pivot_var.loc[: ,[distance_radial_tube_left, distance_radial_tube_right]]
            pivot_var_interp.loc[:, distance_radial_tube] = np.nan
            pivot_var_interp.sort_index(axis=1, inplace=True)
            pivot_var_interp.interpolate(method='index', axis=1, inplace=True)
            profile_contour_dist = pivot_var_interp.loc[:, distance_radial_tube]
            x_line = pivot_var.index
            
            scatter_handle = ax3.scatter(x_line, profile_contour_dist, marker=marker, color= color, edgecolors='k', label=var)
        
            # ax.scatter([distance_radial_tube] * len(x_line), x_line, marker=marker, color='None', edgecolors='k')
            
            # Append the handle to the list of handles
            scatter_handles.append(scatter_handle)
        
        # Append all scatter handles to the handles list
        handles.append(tuple(scatter_handles))
        
    # Create the legend with custom handler map
    handler_map = {tuple(handle): HandlerTuple(ndivide=None) for handle in handles}
    ax3.legend(handles, labels, handler_map=handler_map, title="Time-averaging", prop={"size": 14})

    ax3.set_xlabel(r'$x/D$', fontsize=fontsize)
    ax3.set_ylabel(cbar_title, rotation=0, labelpad=15, fontsize=28)
    ax3.set_ylim(bottom=1.1, top=1.8)
    ax3.tick_params(axis='both', labelsize=fontsize)
    ax3.grid(True)
    ax3.set_aspect('auto', adjustable='box')
    ax3.set_position([0.1, 0.1, 0.8, 0.8])  # Set the position of the axis in the figure (left, bottom, width, height)

    
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
    
    # fig2, ax2 = plt.subplots()
    # # Scatter plot for counts_X and counts_Y
    # ax2.scatter(values_X, counts_X, color='black', label=f'Counts for Value = {value_X}')
    # ax2.scatter(values_Y, counts_Y, color='red', label=f'Counts for Value = {value_Y}')
    
    # # Adding labels and title
    # ax2.set_xlabel('Value')
    # ax2.set_ylabel('Counts')
    # ax2.set_title('Counts as a Function of Value')
    # ax2.legend()
    
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
    
    mass_cons, mom_x, mom_r = fans_terms(df_favre_avg, flame)
    dpdx = mom_x[2] 
    dpdr = mom_r[2]
    
    # r_starts = [.1, .2, .3]
    r_starts = [.2]
    x_starts = np.linspace(0.2, 0.2, len(r_starts))
    start_points = [(r_starts[i], x_starts[i]) for i in range(len(r_starts))]
    
    # ax.scatter(0, streamline_x_start, color='k', label=f'Value = {value_X}')
    
    streamlines, paths, flame_front_indices, colors = plot_streamlines_reacting_flow(r_uniform, x_uniform, u_r_uniform, u_x_uniform, start_points)
    
    #%%% Plots
    # plot_mass_cons(mass_cons, r_norm_values, x_norm_values, streamlines, flame_front_indices, colors)
    plot_fans_terms(mass_cons, mom_x, mom_r, r_norm_values, x_norm_values, streamlines, flame_front_indices, colors)
    plot_pressure_along_streamline(dpdr, dpdx, r_norm_values, x_norm_values, streamlines, flame_front_indices, colors)
    
    #%%% Non-reacting flow
    # non_react_flow = non_react_dict[nonreact_run_nr]
    # name = non_react_flow[0]
    # session_nr = non_react_flow[1] 
    # recording = non_react_flow[2]
    # piv_method = non_react_flow[3]
    # Re_D = non_react_flow[4]
    # u_bulk_set = non_react_flow[5]
    # u_bulk_measured = non_react_flow[6]
    
    # Avg_Stdev_file = os.path.join(data_dir, f'session_{session_nr:03d}', recording, piv_method, 'Avg_Stdev', 'Export', 'B0001.csv')
    
    # df_piv = pd.read_csv(Avg_Stdev_file)
    
    # D_in = flame.D_in # Inner diameter of the quartz tube, units: mm
    # offset = 1  # Distance from calibrated y=0 to tube burner rim
    
    # wall_center_to_origin = 2
    # wall_thickness = 1.5
    # offset_to_wall_center = wall_center_to_origin - wall_thickness/2
    
    # df_piv = process_df(df_piv, D_in, offset_to_wall_center, offset)
    
    # # Get the column headers
    # headers = df_piv.columns
    
    # df_piv_cropped = df_piv[(df_piv[index_name] > bottom_limit) & (df_piv[index_name] < top_limit) & (df_piv[column_name] > left_limit) & (df_piv[column_name] < right_limit)]
    
    # pivot_u_r = pd.pivot_table(df_piv_cropped, values='Velocity u [m/s]', index=index_name, columns=column_name)
    # pivot_u_x = pd.pivot_table(df_piv_cropped, values='Velocity v [m/s]', index=index_name, columns=column_name)
    
    # %%% Save images
    # Get a list of all currently opened figures
    figure_ids = plt.get_fignums()
    figure_ids = [2, 4]
    
    if 'ls' in flame.name:
        folder = 'ls'
    else:
        folder = 'hs'
    
    figures_subfolder = os.path.join(figures_folder, folder)
    if not os.path.exists(figures_subfolder):
            os.makedirs(figures_subfolder)
    
    pickles_subfolder = os.path.join(pickles_folder, folder)
    if not os.path.exists(pickles_subfolder):
            os.makedirs(pickles_subfolder)

    # Apply tight_layout to each figure
    for fid in figure_ids:
        fig = plt.figure(fid)
        filename = f'H{flame.H2_percentage}_Re{flame.Re_D}_fig{fid}_favre'
        
        # Constructing the paths
        if fid == 1:
            
            png_path = os.path.join('figures', f'{folder}', f"{filename}.png")
            pkl_path = os.path.join('pickles', f'{folder}', f"{filename}.pkl")
            
            # Saving the figure in EPS format
            fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
            
        else:
            
            eps_path = os.path.join('figures', f'{folder}', f"{filename}.eps")
            pkl_path = os.path.join('pickles', f'{folder}', f"{filename}.pkl")
            
            # Saving the figure in EPS format
            fig.savefig(eps_path, format='eps', dpi=300, bbox_inches='tight')
        
        # Pickling the figure
        with open(pkl_path, 'wb') as f:
            pickle.dump(fig, f)
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    