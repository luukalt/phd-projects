# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 10:25:15 2023

@author: laaltenburg

"""

#%% IMPORT STANDARD PACKAGES
import os
import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.collections as mcol
from scipy.interpolate import griddata

#%% IMPORT USER DEFINED PACKAGES
from sys_paths import parent_directory
import sys_paths
import rc_params_settings
from parameters import data_dir, piv_method, flame, interpolation_method, nonreact_run_nr 
from plot_params import colormap, fontsize, ms1, ms2, ms3, ms4, ms5, ms6
from cone_angle import cone_angle
from nonreact_flow_fields import non_react_dict
from time_averaging_terms import ns_incomp_terms
from functions import process_df, contour_correction
from plot_functions import plot_cartoons, plot_streamlines_reacting_flow, plot_streamlines_nonreacting_flow, plot_mass_cons, plot_ns_terms, plot_pressure_along_streamline, plot_mass_cons_old

figures_folder = 'figures'
if not os.path.exists(figures_folder):
        os.makedirs(figures_folder)

pickles_folder = 'pickles'
if not os.path.exists(pickles_folder):
        os.makedirs(pickles_folder)
        
#%% MAIN
# Before executing code, Python interpreter reads source file and define few special variables/global variables. 
# If the python interpreter is running that module (the source file) as the main program, it sets the special __name__ variable to have a value “__main__”. If this file is being imported from another module, __name__ will be set to the module’s name. Module’s name is available as value to __name__ global variable. 
if __name__ == '__main__':
    
    name = flame.name
    session_nr = flame.session_nr
    recording = flame.record_name
    piv_method = piv_method
    run_nr = flame.run_nr
    Re_D_set = flame.Re_D
    u_bulk_set = flame.u_bulk_measured
    u_bulk_measured = flame.u_bulk_measured
    
    if flame.H2_percentage == 0:
        # Colorbar limits for DNG-AIR
        cbar_max = [2,
                    2,
                    .01, 
                    .02,
                    .05,
                    .1,
                    ]
    elif flame.H2_percentage == 100:
    # Colorbar limits for H2-AIR
        cbar_max = [2,
                    2,
                    .25, 
                    .02,
                    .04,
                    .02,
                    ]
    
    react_case_info = [name, session_nr, recording, piv_method, Re_D_set, u_bulk_set, u_bulk_measured, cbar_max, nonreact_run_nr]
    
    distances_above_tube = [.25, .75, 1.25, ]
    r_range_left, poly_left_fit, r_range_right, poly_right_fit, alpha = cone_angle(distances_above_tube)
    
    cone_left_line = np.column_stack((r_range_left, poly_left_fit))
    cone_right_line = np.column_stack((r_range_right, poly_right_fit))
    
    print(f'The cone angle (radians): {np.radians(alpha)}')
    print(f'The cone angle (degrees): {alpha}')
    
    Ub_over_Uu = 7.6* np.sin(np.radians(alpha)/2)
    print(f'Ub_over_Uu: {Ub_over_Uu}')
       
    #%%% PIV file column indices
    u_r_col_index = 2
    u_x_col_index = 3
    V_abs_col_index = 4
    du_rdr_col_index = 5
    du_rdx_col_index = 6
    du_xdr_col_index = 7
    du_xdx_col_index = 8
    R_xx_col_index = 15 # RADIAL DIRECTION
    R_xy_col_index = 16
    R_yy_col_index = 17 # AXIAL DIRECTION
    TKE_col_index = 20
    
    col_indices =   [
                    u_x_col_index,
                    V_abs_col_index,
                    R_xx_col_index, 
                    R_xy_col_index, 
                    R_yy_col_index,
                    TKE_col_index,
                    ]
    
    cbar_titles =   [
                    r'$\frac{\overline{u_{x}}}{U_{b}}$',
                    r'$\frac{|\overline{V}|}{U_{b}}$',
                    r'$\frac{R_{rr}}{U_{b}^2}$', # \overline{v\'v\'
                    r'$\frac{R_{rx}}{U_{b}^2}$', # \overline{u\'v\'
                    r'$\frac{R_{xx}}{U_{b}^2}$', # \overline{v\'v\'
                    r'$\frac{k}{U_{b}^2}$', # \overline{v\'v\'
                    ]
    
    #%%% Calibration details
    D_in = flame.D_in # Inner diameter of the quartz tube, units: mm
    offset = 1  # Distance from calibrated y=0 to tube burner rim
    
    wall_center_to_origin = 2
    wall_thickness = 1.5
    offset_to_wall_center = wall_center_to_origin - wall_thickness/2
    
    #%% Start of cases loop
    
    axs = []
    
    for q in range(0, len(col_indices)):
        
        width, height = 9, 6
        fig, ax = plt.subplots(figsize=(width, height))
        
        axs.append(ax)
        
    #%%% Initiate lists where index:0 (reactive) and index:1 (non-reacive)
    
    cbar_max = react_case_info[7]
    nonreact_run_nr = react_case_info[8]
    
    values_list = [react_case_info]
    values = non_react_dict[nonreact_run_nr]
    values_list.append(values)
    
    headers_list = []
    df_piv_cropped_list = []
    df_piv_list = []
    u_bulk_measured_list = []
    Re_D_list = []
    
    pivot_u_r_norm_list = []
    pivot_u_x_norm_list = []
    pivot_u_r_norm_values_list = []
    pivot_u_x_norm_values_list = []
    pivot_V_abs_norm_values_list = []
    
    r_norm_array_list, r_norm_list, r_norm_values_list, = [], [], []
    x_norm_array_list, x_norm_list, x_norm_values_list = [], [], []
    
    r_uniform_list, x_uniform_list = [], []
    u_r_uniform_list, u_x_uniform_list = [], []
    
    #%%% Start of react + non-react loop
    for ii, values in enumerate(values_list):
        
        name = values[0]
        session_nr = values[1] 
        recording = values[2]
        piv_method = values[3]
        Re_D = values[4]
        u_bulk_set = values[5]
        u_bulk_measured = values[6]
        
        print(name)
        
        #%%%% Plot a cartoon
        if ii == 0:
            
            # image_nrs = [3167, 3169, 3171, 3173, 3175,  3177]
            # image_nrs = [2306, 2308, 2310, 2312, 2314, 2316] #[4624] #2314 #[4496]
            # image_nrs = [1737, 1738] #[4624] #2314 #[4496]
             #[4624] #2314 #[4496]
            
            # image_nrs = [2297, 2298]
            image_nrs = [1241, 1242]
            
            # image_nrs = [3172, 3174]
            # 
            # plot_cartoons(flame, image_nrs, recording, piv_method, D_in, offset_to_wall_center, offset, u_bulk_measured, cbar_titles)
            
            # fig, axs = plt.subplots(3, 2, figsize=(10, 15))
            
            # for image_i, ax_i in enumerate(axs.ravel()):
                
            #     image_nr = image_nrs[image_i]
            #     plot_cartoons(flame, ax_i, image_nr, recording, piv_method)
                
            #     if image_i in [1, 3, 5]:
            #         ax_i.set_ylabel('')
            #         ax_i.tick_params(axis='y', labelleft=False)
                
            #     if image_i in [0, 1, 2, 3]:
            #         ax_i.set_xlabel('')
            #         ax_i.tick_params(axis='x', labelbottom=False)
            
            # fig.subplots_adjust(wspace=-.75)
            
            # fig.tight_layout()
            
            # filename = f'H{flame.H2_percentage}_Re{flame.Re_D}_sequence2'
            # png_path = os.path.join('figures', f"{filename}.png")
            # fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
        
        #%%%% Read the averaged PIV file and add coordinate system translation
        # PIV directory
        Avg_Stdev_file = os.path.join(data_dir, f'session_{session_nr:03d}', recording, piv_method, 'Avg_Stdev', 'Export', 'B0001.csv')
        
        # Read the averaged PIV file and add coordinate system translation
        df_piv = pd.read_csv(Avg_Stdev_file)
        df_piv = process_df(df_piv, D_in, offset_to_wall_center, offset)
        
        # Get the column headers of the PIV file
        headers = df_piv.columns
        
        # Non-dimensional limits in r- (left, right) and x-direction (bottom, top)
        bottom_limit = -.5
        top_limit = 2.25
        left_limit = -0.575
        right_limit = 0.575
        index_name = 'y_shift_norm'
        column_name = 'x_shift_norm'
        
        # Cropped PIV dataframe based on non-dimensional limits in r- (left, right) and x-direction (bottom, top)
        df_piv_cropped = df_piv[(df_piv[index_name] > bottom_limit) & (df_piv[index_name] < top_limit) & (df_piv[column_name] > left_limit) & (df_piv[column_name] < right_limit)]
        
        # Obtain velocity fields
        pivot_u_r = pd.pivot_table(df_piv_cropped, values=headers[u_r_col_index], index=index_name, columns=column_name)
        pivot_u_x = pd.pivot_table(df_piv_cropped, values=headers[u_x_col_index], index=index_name, columns=column_name)
        pivot_V_abs = pd.pivot_table(df_piv_cropped, values=headers[V_abs_col_index], index=index_name, columns=column_name)
        
        # Create r,x PIV grid
        r_norm_array = pivot_u_r.columns
        x_norm_array = pivot_u_r.index
        r_norm, x_norm = np.meshgrid(r_norm_array, x_norm_array)
        r_norm_values = r_norm.flatten()
        x_norm_values = x_norm.flatten()
        
        # Obtain dimensionless velocity fields
        pivot_u_r_norm = pivot_u_r/u_bulk_measured
        pivot_u_x_norm = pivot_u_x/u_bulk_measured
        pivot_V_abs_norm = pivot_V_abs/u_bulk_measured
        
        pivot_u_r_norm_list.append(pivot_u_r_norm)
        pivot_u_x_norm_list.append(pivot_u_x_norm)
        
        pivot_u_r_norm_values = pivot_u_r_norm.values.flatten()
        pivot_u_x_norm_values = pivot_u_x_norm.values.flatten()
        pivot_V_abs_norm_values = pivot_V_abs_norm.values.flatten()
        
        # Create a uniform r,x PIV grid
        r_uniform = np.linspace(r_norm_values.min(), r_norm_values.max(), len(r_norm_array))
        x_uniform = np.linspace(x_norm_values.min(), x_norm_values.max(), len(x_norm_array))
        r_uniform, x_uniform = np.meshgrid(r_uniform, x_uniform)
        
        # Interpolate the velocity components to the uniform grid
        u_r_uniform = griddata((r_norm_values, x_norm_values), pivot_u_r_norm_values, (r_uniform, x_uniform), method=interpolation_method)
        u_x_uniform = griddata((r_norm_values, x_norm_values), pivot_u_x_norm_values, (r_uniform, x_uniform), method=interpolation_method)
        
        # Append to important variables to list (0: reacting, 1: non-reacting)
        headers_list.append(headers)
        df_piv_cropped_list.append(df_piv_cropped)
        df_piv_list.append(df_piv)
        u_bulk_measured_list.append(u_bulk_measured)
        Re_D_list.append(Re_D)
        
        pivot_u_r_norm_values_list.append(pivot_u_r_norm_values)
        pivot_u_x_norm_values_list.append(pivot_u_x_norm_values)
        pivot_V_abs_norm_values_list.append(pivot_V_abs_norm_values)
        
        r_norm_array_list.append(r_norm_array)
        r_norm_list.append(r_norm)
        r_norm_values_list.append(r_norm_values)
        
        x_norm_array_list.append(x_norm_array)
        x_norm_list.append(x_norm)
        x_norm_values_list.append(x_norm_values)
        
        r_uniform_list.append(r_uniform)
        x_uniform_list.append(x_uniform)
        u_r_uniform_list.append(u_r_uniform)
        u_x_uniform_list.append(u_x_uniform)
    
    #%%% Start of plots
    for i, col_index in enumerate(col_indices):
        
        ax = axs[i]
        
        pivot_var_list = []
        pivot_var_values_list = []
        
        for p in [0, 1]:
            
            df_piv = df_piv_cropped_list[p]
            headers = headers_list[p]
            
            pivot_var = pd.pivot_table(df_piv, values=headers[col_index], index=index_name, columns=column_name)
            
            # Apply correction for Davis approach for TKE 2D
            if col_index == TKE_col_index:
                pivot_var *= 2/3
                
            pivot_var_values = pivot_var.values.flatten() 
            
            pivot_var_list.append(pivot_var)
            pivot_var_values_list.append(pivot_var_values)
            
        if i == 0:
            
            nondim_value = u_bulk_measured_list[0]
            
            # Define radial locations
            r_norm_lines = [.0,]
            
            # Define x-limits
            x_norm_min, x_norm_max = bottom_limit, top_limit
            
            # Define vline stepsize
            vline_step = 0.05
            
            #%%%% Plot streamlines
            # Steamline starting locations
            r_starts = [.2,]
            # r_starts = [.1, .2, .3]
            x_starts = np.linspace(0.2, 0.2, len(r_starts))
            start_points = [(r_starts[i], x_starts[i]) for i in range(len(r_starts))]
            
            # Initialize figures
            width, height = 6, 6
            
            fig, ax6 = plt.subplots()
            
            fig, ax7 = plt.subplots(figsize=(width + 1, height))
            
            fig, ax8 = plt.subplots(figsize=(width + 1, height))
            
            fig9, ax9 = plt.subplots(figsize=(width, height))
            ax9.grid(True)
            
            # Define the region of interest for zooming
            ax9_x1, ax9_x2, ax9_y1, ax9_y2 = -.05, 2.15, -.25, .25  # for example, zoom in on this region

            # Draw a rectangle or any other shape to indicate the zoom area
            ax9.add_patch(plt.Rectangle((ax9_x1, ax9_y1), ax9_x2 - ax9_x1, ax9_y2 - ax9_y1, fill=False, color='k', linestyle='solid', lw=2))
            
            ax9_inset = inset_axes(ax9, width="100%", height="400%", loc='upper left',
                              # bbox_to_anchor=(x1, y1, x2 - x1, y2 - y1),
                              bbox_to_anchor=(ax9_x1, ax9_y1 + 7.7, ax9_x2 - ax9_x1, ax9_y2 - ax9_y1),
                              bbox_transform=ax9.transData,
                              borderpad=0)
            
            # Set the limits for the inset axes
            ax9_inset.set_xlim(ax9_x1, ax9_x2)
            ax9_inset.set_ylim(ax9_y1, ax9_y2)
            
            ax9_inset.set_xticks([])
            for spine in ax9_inset.spines.values():
                spine.set_linewidth(2)

            for p in [0, 1]:
                
                r_uniform = r_uniform_list[p]
                x_uniform = x_uniform_list[p]
                u_r_uniform = u_r_uniform_list[p]
                u_x_uniform = u_x_uniform_list[p]
                
                df_piv = df_piv_cropped_list[p]
                u_bulk_measured = u_bulk_measured_list[p]
                Re_D = Re_D_list[p]
                r_norm_values = r_norm_values_list[p]
                x_norm_values = x_norm_values_list[p]
                
                mass_cons, mom_x, mom_r, = ns_incomp_terms(df_piv, D_in, u_bulk_measured, Re_D)
                
                if p == 0:
                    streamlines, paths, flame_front_indices, colors = plot_streamlines_reacting_flow(r_uniform, x_uniform, u_r_uniform, u_x_uniform, start_points)
                    incomp_indices = plot_ns_terms(mass_cons, mom_x, mom_r, r_norm_values, x_norm_values, streamlines, flame_front_indices, colors)
                    input_indices = flame_front_indices
                else:
                    streamlines, paths, dummy_indices, colors = plot_streamlines_nonreacting_flow(r_uniform, x_uniform, u_r_uniform, u_x_uniform, start_points)
                    incomp_indices = plot_ns_terms(mass_cons, mom_x, mom_r, r_norm_values, x_norm_values, streamlines, dummy_indices, colors)
                    input_indices = dummy_indices
                
                plot_mass_cons_old(p, ax9, ax9_inset, mass_cons, r_norm_values, x_norm_values, streamlines, input_indices, colors)
                
                ax9.set_xlim([-.5, 2.2])  # replace with your desired x limits
                ax9.set_ylim(top=8.25)  # replace with your desired x limits
                
                for streamline, path, flame_front_index, color, incomp_index in zip(streamlines, paths, input_indices, colors, incomp_indices):
                    
                    # Use scipy's griddata function to interpolate the velocities along the line
                    velocities = griddata((r_norm_values_list[p], x_norm_values_list[p]), pivot_V_abs_norm_values_list[p], streamline, method=interpolation_method)
                    
                    if p == 0:
                        ls = 'solid'
                    elif p == 1:
                        ls = 'dashed'
                        
                    plot_line, = ax6.plot(path, velocities, c=color, marker='None', ls=ls)
                    
                    if p == 0:
                        ax6.plot(path[flame_front_index], velocities[flame_front_index], c=color, marker='*', mec='k', ms=ms5)
                        ax6.plot(path[incomp_index[:]], velocities[incomp_index[:]], lw=5, c=color, marker='None')
                        ax6.plot(path[incomp_index[-1]], velocities[incomp_index[-1]], ls='None', c=color, marker='s', mec='k')
                        
                    # plot_line, = ax6.plot(path, velocities, c=color, marker='None')
                                                              
                    ax6.set_xlabel(r'$s/D$', fontsize=20)
                    ax6.set_ylabel(cbar_titles[1], rotation=0, labelpad=15, fontsize=24)
                    ax6.grid(True)
                    
                    dpdx = mom_x[2] #*(u_bulk_measured**2)/(D_in*1e-3)
                    dpdr = mom_r[2] #*(u_bulk_measured**2)/(D_in*1e-3)
                    # plot_pressure_along_streamline(ax7, ax8, dpdr, dpdx, r_norm_values, x_norm_values, streamlines, incomp_indices, colors, p)
                    
                styles_react = ['solid', 'solid', 'solid']
                styles_nonreact = ['dashed', 'dashed', 'dashed']
                # make list of one line -- doesn't matter what the coordinates are
                dummy_line = [[(0, 0)]]
                # set up the proxy artist
                lc_react = mcol.LineCollection(3 * dummy_line, linestyles=styles_react, colors=colors)
                lc_nonreact = mcol.LineCollection(3 * dummy_line, linestyles=styles_nonreact, colors=colors)
                
                # # Create the legend
                # ax6.legend([lc_react, lc_nonreact], ['reacting flow', 'non reacting flow'], handler_map={type(lc_react): HandlerDashedLines()},
                #            handlelength=3, handleheight=3)
                
                ax6.set_xlim(-0.05, 2.25)
                ax6.set_ylim(0.9, 1.7)
                
                p = 0
                df_piv = df_piv_cropped_list[p]
                u_bulk_measured = u_bulk_measured_list[p]
                Re_D = Re_D_list[p]
                r_norm_values = r_norm_values_list[p]
                x_norm_values = x_norm_values_list[p]
            
        elif i == 1:
            
            nondim_value = u_bulk_measured_list[0]

        else:
            
            nondim_value = u_bulk_measured_list[0]**2
        
        # fig9.tight_layout()
        
        #%%%% Plot average 'X' field
        flow_field = ax.pcolor(r_norm_list[0], x_norm_list[0], pivot_var_list[0].values/nondim_value, cmap=colormap)
        
        ax._X_data = r_norm_list[0]
        ax._Y_data = x_norm_list[0]
        ax._Z_data = pivot_var_list[0].values/nondim_value
        ax._cmap = colormap
        ax._clim = [0, cbar_max[i]]
        
        x_limits = ax.get_xlim()
        y_limits = ax.get_ylim()
        
        if i == 1:
            skip = 4
            ax.quiver(r_norm_list[0][::skip], x_norm_list[0][::skip], pivot_u_r_norm_list[0][::skip], pivot_u_x_norm_list[0][::skip], angles='xy', scale_units='xy', scale=20, width=0.005, color='grey')
            
        
        if i == 3:
            flow_field.set_clim(-cbar_max[i], cbar_max[i])
        else:
            flow_field.set_clim(0, cbar_max[i])
        
        if i == 5:
            
            # Raw Mie-scattering directory
            image_nr = 1
            raw_dir = os.path.join(data_dir,  f'session_{flame.session_nr:03d}', flame.record_name, 'Correction', 'Resize', 'Frame0', 'Export')
            raw_file = os.path.join(raw_dir, f'B{image_nr:04d}.csv')
            
            # Read the raw Mie-scattering image and add coordinate system translation
            df_raw = pd.read_csv(raw_file)
            df_raw = process_df(df_raw, D_in, offset_to_wall_center, offset)
            
            # Get the column headers of the raw Mie-scattering image file
            headers_raw = df_raw.columns
            
            # Obtain intensity field
            pivot_intensity = pd.pivot_table(df_raw, values=headers_raw[2], index=index_name, columns=column_name)
            
            # Create r,x raw Mie scattering grid
            r_raw_array = pivot_intensity.columns
            x_raw_array = pivot_intensity.index
            r_raw, x_raw = np.meshgrid(r_raw_array, x_raw_array)
            n_windows_r_raw, n_windows_x_raw = len(r_raw_array), len(x_raw_array)
            window_size_r_raw, window_size_x_raw = np.mean(np.diff(r_raw_array)), -np.mean(np.diff(x_raw_array))
            
            # Parameters for correcting contours from pixel coordinates to physical coordinates
            r_left_raw = r_raw_array[0]
            r_right_raw = r_raw_array[-1]
            x_bottom_raw = x_raw_array[0]
            x_top_raw = x_raw_array[-1]
            
            if flame.Re_D == 3000:
                image_nrs = [10, 1112, 2018, 2259]
            
            elif flame.Re_D == 4000:
                image_nrs = [12, 1879, 2204, 2361]
            
            elif flame.Re_D == 12500:
                image_nrs = [471, 1251, 1473] # Re=12500
                
            elif flame.Re_D == 16000:
                image_nrs = [1500, 1600, 2171]
                
            for image_nr in image_nrs:
                contour_nr = image_nr - 1
                contour_corrected = contour_correction(flame, contour_nr, r_left_raw, r_right_raw, x_bottom_raw, x_top_raw, window_size_r_raw, window_size_x_raw, frame_nr=0)
                contour_x, contour_y =  contour_corrected[:,0,0], contour_corrected[:,0,1]
                ax.plot(contour_x, contour_y, color='r', ls='solid') 
        
        # Define your custom colorbar tick locations and labels
        num_ticks = 6
        custom_cbar_ticks = np.linspace(0, cbar_max[i], num_ticks) # Replace with your desired tick positions
        
        if cbar_max[i] < 1:
            custom_cbar_tick_labels = [f'{tick:.2f}' for tick in custom_cbar_ticks] # Replace with your desired tick labels
        else:
            custom_cbar_tick_labels = [f'{tick:.1f}' for tick in custom_cbar_ticks] # Replace with your desired tick labels
        
        # Set the colorbar ticks and labels
        if flame.Re_D in [3000, 12500]: 
            cbar = ax.figure.colorbar(flow_field)
            cbar.set_ticks(custom_cbar_ticks)
            cbar.set_ticklabels(custom_cbar_tick_labels)
            cbar.set_label(cbar_titles[i], rotation=0, labelpad=25, fontsize=28) 
            cbar.ax.tick_params(labelsize=fontsize)
        
        ax.set_aspect('equal')
        ax.set_xlabel(r'$r/D$', fontsize=fontsize)
        ax.set_ylabel(r'$x/D$', fontsize=fontsize)
        
        ax.set_xlim(left=-.55, right=.55)
        ax.set_ylim(bottom=0, top=2.2)
        
        # ax.set_xlim(x_limits)  # replace with your desired x limits
        # ax.set_ylim(y_limits)  # replace with your desired y limits
        
        # Cone angle
        ax.plot(r_range_left, poly_left_fit, c='k', ls='dashed') 
        ax.plot(r_range_right, poly_right_fit, c='k', ls='dashed')
        
        ax.tick_params(axis='both', labelsize=fontsize)
        ax.tick_params(axis='both', labelsize=fontsize)
        
        # if i == 1:
            
        #     # Determine shared colorbar limits
        #     vmin = 0 
        #     vmax = 2
        #     # vmax= 1.25
            
        #     # Create a figure and a 3D axis
        #     fig0 = plt.figure()

        #     # =============
        #     # set up the axes for the first plot
        #     z_bottom = 0.
        #     z = pivot_var_list[1].values/u_bulk_measured_list[1]
        #     z[z < z_bottom] = np.nan
            
        #     ax = fig0.add_subplot(1, 1, 1, projection='3d')
        #     surf = ax.plot_surface(r_norm_list[1], x_norm_list[1], z, cmap=colormap, vmin=vmin, vmax=vmax, edgecolors='k', lw=0.1)
            
        #     # Adjust the limits, add labels, title, etc.
        #     ax.set_xlabel(r'$r/D$', labelpad=5)
        #     ax.set_ylabel(r'$x/D$', labelpad=5)
        #     # ax.set_zlabel('Z Label')
        #     ax.set_title('Non-reactive flow')
        #     ax.set_aspect('equal')
        #     ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # x-axis
        #     ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # y-axis
        #     ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # z-axis
        #     ax.set_zlim(z_bottom, vmax)
        #     cbar = fig0.colorbar(surf, pad=0.01)
        #     cbar.set_label(cbar_titles[i], rotation=0, labelpad=15, fontsize=14)
        #     ax.set_xticks([-.5, 0., .5])
        #     ax.set_yticks([0, 1, 2])
        #     ax.set_zticks([])
        #     fig0.tight_layout()
            
        #     # Create a figure and a 3D axis
        #     fig1 = plt.figure()
        #     z = pivot_var_list[0].values/nondim_value
        #     z[z < z_bottom] = np.nan
            
        #     z_values = z.flatten()
            
        #     cone_left_line_3d = griddata((r_norm_values, x_norm_values), z_values, cone_left_line, method=interpolation_method)
        #     cone_right_line_3d = griddata((r_norm_values, x_norm_values), z_values, cone_right_line, method=interpolation_method)
            
        #     ax = fig1.add_subplot(1, 1, 1, projection='3d')
        #     ax.plot(cone_left_line[:,0], cone_left_line[:,1], cone_left_line_3d, c='k', ls='dashed', zorder=10)
        #     ax.plot(cone_right_line[:,0], cone_right_line[:,1], cone_right_line_3d, c='k', ls='dashed', zorder=10)
            
        #     surf = ax.plot_surface(r_norm_list[0], x_norm_list[0], z, cmap=colormap, vmin=vmin, vmax=vmax, edgecolors='k', lw=0.1, zorder=-1)
            
        #     # Adjust the limits, add labels, title, etc.
        #     ax.set_xlabel(r'$r/D$', labelpad=5)
        #     ax.set_ylabel(r'$x/D$', labelpad=5)
        #     # ax.set_zlabel(cbar_titles[i])
        #     ax.set_title('Reactive flow')
        #     ax.set_aspect('equal')
        #     ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # x-axis
        #     ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # y-axis
        #     ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # z-axis
        #     ax.set_zlim(z_bottom, vmax)
        #     cbar = fig1.colorbar(surf, pad=0.01)
        #     cbar.set_label(cbar_titles[i], rotation=0, labelpad=15, fontsize=14)
        #     ax.set_xticks([-.5, 0., .5])
        #     ax.set_yticks([0, 1, 2])
        #     ax.set_zticks([])
        #     fig1.tight_layout()


# ax1.legend()
# ax2.legend()
# ax3.legend()
# ax4.legend()

#%% Save images
# Get a list of all currently opened figures
# figure_ids = plt.get_fignums()
# figure_ids = [8]

# if 'ls' in flame.name:
#     folder = 'ls'
# else:
#     folder = 'hs'

# figures_subfolder = os.path.join(figures_folder, folder)
# if not os.path.exists(figures_subfolder):
#         os.makedirs(figures_subfolder)

# pickles_subfolder = os.path.join(pickles_folder, folder)
# if not os.path.exists(pickles_subfolder):
#         os.makedirs(pickles_subfolder)

# # Apply tight_layout to each figure
# for fid in figure_ids:
#     fig = plt.figure(fid)
#     fig.tight_layout()
#     filename = f'H{flame.H2_percentage}_Re{Re_D_list[0]}_fig{fid}'
    
#     # Constructing the paths
#     if fid == 1:
        
#         png_path = os.path.join('figures', f'{folder}', f'{filename}.png')
#         pkl_path = os.path.join('pickles', f'{folder}', f'{filename}.pkl')
        
#         # Saving the figure in EPS format
#         fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
        
#     else:
        
#         eps_path = os.path.join('figures', f'{folder}', f'{filename}.eps')
#         pkl_path = os.path.join('pickles', f'{folder}', f'{filename}.pkl')
        
#         # Saving the figure in EPS format
#         fig.savefig(eps_path, format='eps', dpi=300, bbox_inches='tight')
    
    
#     # Pickling the figure
#     with open(pkl_path, 'wb') as f:
#         pickle.dump(fig, f)
        













    
    