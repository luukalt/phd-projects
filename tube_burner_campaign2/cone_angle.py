# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 15:51:03 2023

@author: laaltenburg
"""
#%% IMPORT PACKAGES
import os
import pickle
import socket
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.interpolate import griddata
from custom_colormaps import parula

#%% FIGURE SETTINGS

# mpl.rcParams.update(mpl.rcParamsDefault)

# def set_mpl_params():
    
#     # Use Latex font in plots
#     plt.rcParams.update({
#         'text.usetex': False,
#         'font.family': 'serif',
#         'font.family': 'Arial',
        
#         'font.serif': ['Computer Modern Roman'],
#         'font.serif': ['Times New Roman'],
#         'font.size': 14.0})
    
#     # Shading of pcolor plot
#     plt.rcParams['pcolor.shading'] = 'auto'

#%%% COLORS
jet = mpl.colormaps['jet']
tab10 = mpl.colormaps['tab10']
tab20 = mpl.colormaps['tab20']

# Google hex-colors
google_red = '#db3236'
google_green = '#3cba54'
google_blue = '#4885ed'
google_yellow = '#f4c20d'

#%% IMPORTANT: SET NORMALIZATION
normalized = False

#%% IMPORTANT: TOGGLE PLOT
toggle_plot = False

#%% VARIABLES DICTIONARY
variables_dict = {
    
    'slope' : (r's_{i}', r''),
    'slope_change' : (r'a_{i}', r''),
    'V_abs' : (r'|V|', r'[ms^{-1}]'),
    'V_t' : (r'V_{t}', r'[ms^{-1}]'),
    'V_n' : (r'V_{n}', r'[ms^{-1}]'),
    'S_a' : ('S_{a} ', '[ms^{-1}]'),
    'S_d' : ('S_{d}', r'[ms^{-1}]'),
    '|S_d|' : (r'|S_{d}|', r'[ms^{-1}]'),
    'curvature' : (r'\nabla \cdot \vec n', r'[mm^{-1}]'),
    'stretch_tangential' : (r'\nabla_t \cdot \vec u', r'[s^{-1}]'),
    'stretch_tangential_abs' : (r'| \nabla_t \cdot \vec u |', r'[s^{-1}]'),
    'stretch_tangential_norm': (r'(\nabla_t \cdot \vec u) \times \frac{D}{U_{b}}', r''),
    'stretch_tangential_abs_norm' : (r'(| \nabla_t \cdot \vec u |) \times \frac{D}{U_{b}}', r''),
    'stretch_curvature' : (r'S_d \nabla \cdot \vec n', r'[s^{-1}]'),
    'stretch_curvature_abs' : (r'| S_{d} \nabla \cdot \vec n |', r'[s^{-1}]'),
    'stretch_curvature_norm': (r'(S_d \nabla \cdot \vec n) \times \frac{D}{U_{b}}', r''),
    'stretch_curvature_abs_norm': (r'(| S_d \nabla \cdot \vec n |) \times \frac{D}{U_{b}}', r'')
}

#%% IMPORTANT: SET FLAME INFO
pc_name = socket.gethostname()

if pc_name == 'A7' or 'DESKTOP-KA87T30':
    
    data_dir = 'U:\\staff-umbrella\\High hydrogen\\laaltenburg\\data\\tube_burner_campaign2\\selected_runs\\'

else:

    data_dir = 'U:\\High hydrogen\\laaltenburg\\data\\tube_burner_campaign2\\selected_runs\\'


def find_intersection_and_angle_from_arrays(x1, y1, x2, y2):
    # Fit lines to the point arrays
    m1, b1 = np.polyfit(x1, y1, 1)
    m2, b2 = np.polyfit(x2, y2, 1)
    
    # Check if lines are parallel
    if m1 == m2:
        return None, None

    # Find intersection
    x_intersection = (b2 - b1) / (m1 - m2)
    y_intersection = m1 * x_intersection + b1

    # Find the angle between the lines using arctangent
    angle_rad = np.arctan(abs((m2 - m1) / (1 + m1 * m2)))
    angle_deg = np.degrees(angle_rad)

    return (x_intersection, y_intersection), angle_deg


def cone_angle(spydata_dir, name, distances_above_tube=[.75, 1., 1.25]):

    segment_length_mm = 1 # units: mm
    window_size = 31 # units: pixels
    
    #%% READ FLAME INFO [DO NOT TOUCH]
    
    # spydata_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spydata')
    # spydata_dir = os.path.join(data_dir, 'spydata')
    
    fname = f'{name}_segment_length_{segment_length_mm}mm_wsize_{window_size}pixels'
    
    with open(os.path.join(spydata_dir, fname + '.pkl'), 'rb') as f:
        flame = pickle.load(f)
    
    #%% NON_DIMENSIONALIZE TOGGLE [DO NOT TOUCH]
    
    # Set if plot is normalized or non-dimensionalized
    if normalized:
        D = flame.D_in
        U_bulk = flame.Re_D*flame.properties.nu_u/(flame.D_in*1e-3)
        scale_vector = 20
    else:
        D = 1
        U_bulk = 1
        scale_vector = 0.5*flame.Re_D*flame.properties.nu_u/(flame.D_in*1e-3)
    
    #%% READ RAW IMAGE DATA [DO NOT TOUCH]
    
    # Construct file path
    raw_dir = os.path.join(data_dir,  f'session_{flame.session_nr:03d}', flame.record_name, 'Correction', 'Resize', 'Frame0', 'Export')
    raw_file = os.path.join(raw_dir, 'B0001.csv')
    
    df_raw = pd.read_csv(raw_file)
    
    offset = 1
    wall_center_to_origin = 2
    wall_thickness = 1.5
    offset_to_wall_center = wall_center_to_origin - wall_thickness/2
    
    # df_raw['x_norm'] = (df_raw['x [mm]'] - (flame.D_in/2 - offset_to_wall_center))/flame.D_in
    # df_raw['y_norm'] = (df_raw['y [mm]'] - offset)/flame.D_in
    
    df_raw['x_shift [mm]'] = df_raw['x [mm]'] - (flame.D_in/2 - offset_to_wall_center)
    df_raw['y_shift [mm]'] = df_raw['y [mm]'] + offset
    
    df_raw['x_shift_norm'] = df_raw['x_shift [mm]']/flame.D_in
    df_raw['y_shift_norm'] = df_raw['y_shift [mm]']/flame.D_in
    
    df_raw['x_shift [m]'] = df_raw['x_shift [mm]']*1e-3
    df_raw['y_shift [m]'] = df_raw['y_shift [mm]']*1e-3
    
    headers_raw = df_raw.columns
    
    # Read intensity
    pivot_intensity = pd.pivot_table(df_raw, values=headers_raw[2], index='y_shift_norm', columns='x_shift_norm')
    
    r_raw_norm_array, x_raw_norm_array = pivot_intensity.columns, pivot_intensity.index
    r_raw_norm, x_raw_norm = np.meshgrid(r_raw_norm_array, x_raw_norm_array)
    r_raw_norm_values = r_raw_norm.flatten()
    x_raw_norm_values = x_raw_norm.flatten()
    
    n_windows_r_raw, n_windows_x_raw = len(r_raw_norm_array), len(x_raw_norm_array)
    window_size_r_raw, window_size_x_raw = np.mean(np.diff(r_raw_norm_array)), -np.mean(np.diff(x_raw_norm_array))
    window_size_r_raw_abs, window_size_x_raw_abs = np.abs(window_size_r_raw), np.abs(window_size_x_raw)
    
    extent_raw =  np.array([
                            r_raw_norm.min() - window_size_r_raw_abs / 2,
                            r_raw_norm.max() + window_size_r_raw_abs / 2,
                            x_raw_norm.min() - window_size_x_raw_abs / 2,
                            x_raw_norm.max() + window_size_x_raw_abs / 2
                            ])
    
    r_left_raw = r_raw_norm_array[0]
    r_right_raw = r_raw_norm_array[-1]
    x_bottom_raw = x_raw_norm_array[0]
    x_top_raw = x_raw_norm_array[-1]
    
    contour_dist = flame.frames[0].contour_data.contour_distribution
    
    x_raw_array_reversed = pd.Index(x_raw_norm_array.tolist()[::-1], name='y_norm')
    
    contour_dist_df = pd.DataFrame(contour_dist, index=x_raw_array_reversed, columns=r_raw_norm_array)
    # contour_dist_values = np.flip(contour_dist_df.values, axis=0).flatten()
    
    contour_dist_df /= contour_dist_df.values.sum()
    fig1, ax1 = plt.subplots()
    
    flow_field = ax1.pcolor(contour_dist_df.columns, contour_dist_df.index, contour_dist_df, cmap=parula)
    cbar = ax1.figure.colorbar(flow_field)
    
    cbar_max = 1.5e-5 #contour_dist_df.values.max()/8
    
    flow_field.set_clim(0, cbar_max)
    
    # Define your custom colorbar tick locations and labels
    num_ticks = 4
    custom_cbar_ticks = np.linspace(0, cbar_max, num_ticks) # Replace with your desired tick positions
    
    # Set the colorbar ticks and labels
    cbar.set_ticks(custom_cbar_ticks)
    # cbar.set_ticklabels(custom_cbar_tick_labels)
    
    fontsize = 16
    cbar.set_label('probability density of flame front contour', fontsize=16) 
    cbar.ax.tick_params(labelsize=fontsize)
    
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    cbar.ax.yaxis.set_major_formatter(formatter)
    cbar.ax.yaxis.get_offset_text().set_fontsize(fontsize)
    
    ax1.set_aspect('equal')
    x_limits = ax1.get_xlim()
    y_limits = ax1.get_ylim()
    
    # flame_contour = ax1.imshow(contour_dist, extent=extent_raw)
    # flame_contour.set_clim(0, contour_dist.max()/8)
    # colorbar = fig1.colorbar(flame_contour)
    # colorbar.ax.yaxis.set_offset_position('left')  
    # colorbar.set_label('probability density')
    
    ax1.set_xlabel('$r/D$', fontsize=fontsize)
    ax1.set_ylabel('$x/D$', fontsize=fontsize)
    
    
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    # ax2.set_title('Probability density for different axial locations')
    ax2.set_xlabel('$r/D$', fontsize=fontsize)
    ax2.set_ylabel('probability density', fontsize=fontsize)
    
    # distances_above_tube = [.75, 1., 1.25]
    
    peaks_left = []
    peaks_right = []
    
    # colors = cm.viridis(np.linspace(0, 1, len(distances_above_tube)))
    
    r = contour_dist_df.columns
    r_begin_index = 0
    r_end_index = len(contour_dist_df.columns) - 1
    r_mid_index = len(contour_dist_df.columns)//2
    r_indices = np.linspace(r_begin_index, r_end_index, len(contour_dist_df.columns))
    r_mid = contour_dist_df.columns[r_mid_index]
    
    # colors = ['#800080', '#00FFFF', '#FF6600']
    colors = ['orange', 'cyan', 'magenta']
    

    for i, distance_above_tube in enumerate(distances_above_tube):
    
        # Find the two closest indices to the given index
        distance_above_tube_below = contour_dist_df.index[contour_dist_df.index <= distance_above_tube].max()
        distance_above_tube_above = contour_dist_df.index[contour_dist_df.index >= distance_above_tube].min()
        
        contour_dist_df_interp = contour_dist_df.loc[[distance_above_tube_below, distance_above_tube_above]]
        contour_dist_df_interp.loc[distance_above_tube] = np.nan
        contour_dist_df_interp.sort_index(inplace=True)
        contour_dist_df_interp.interpolate(method='index', inplace=True)
        profile_contour_dist = contour_dist_df_interp.loc[distance_above_tube]
        
        z = profile_contour_dist
        
        # Define your integration limits
        r_min, r_max = x_limits[0],  x_limits[1]
        
        # Mask to select the data within your integration range
        mask = (r >= r_min) & (r <= r_max)
        
        # Select the x and y data within your integration range
        r_range = r[mask]
        z_range = z[mask]
        
        r_range_left = r_range[z_range.index < 0]
        r_range_right = r_range[z_range.index > 0]
        
        z_range_left = z_range[z_range.index < 0]
        z_range_right = z_range[z_range.index > 0]
        
        # hline_step = 0.01
        # hline_r = np.arange(r_min, 0 + hline_step, hline_step)
        # hline = np.column_stack((hline_r, np.full(len(hline_r), distance_above_tube)))
        # pivot_var_along_vline = griddata((r_raw_norm_values, x_raw_norm_values), contour_dist_values, hline, method='linear')
        # scatter_plot = ax2.scatter(hline_r, pivot_var_along_vline, marker='o', ls='None')
        
        scatter_plot = ax2.scatter(r_range, z_range, c=colors[i], marker='o', edgecolors='k', ls='None')
        data_point_color = scatter_plot.get_facecolor()
        
        mean_r_left = 0
        mean_r_right = 0
        
        mean_r_left = np.sum(r_range_left * z_range_left) / np.sum(z_range_left)
        mean_r_right = np.sum(r_range_right * z_range_right) / np.sum(z_range_right)
        
        # print(mean_r_left, mean_r_right)
        peaks_left.append((mean_r_left, distance_above_tube))
        peaks_right.append((mean_r_right, distance_above_tube))
        
        ax1.plot(mean_r_left, distance_above_tube, color=data_point_color, marker='o', ms=10, mec='k', ls='--')
        ax1.plot(mean_r_right, distance_above_tube, color=data_point_color, marker='o', ms=10, mec='k', ls='--')
        
        ax2.axvline(x=mean_r_left, color=data_point_color, ls='--')
        ax2.axvline(x=mean_r_right, color=data_point_color, ls='--')
        
        ax1.hlines(y=distance_above_tube, xmin=r_min, xmax=r_max, color=data_point_color, ls='--')
        
    peaks = peaks_left + peaks_right
    
    poly_order = 1
    
    peaks_left_r = [r for r,x in peaks_left]
    peaks_left_x = [x for r,x in peaks_left]
    coefs_left = np.polyfit(peaks_left_r, peaks_left_x, poly_order)
    poly_left = np.poly1d(coefs_left)
    poly_left_fit = poly_left(r_range_left)
    alpha_left = np.degrees(np.arctan(abs(1/coefs_left[0])))
    
    peaks_right_r = [r for r,x in peaks_right]
    peaks_right_x = [x for r,x in peaks_right]
    coefs_right = np.polyfit(peaks_right_r, peaks_right_x, poly_order)
    poly_right = np.poly1d(coefs_right)
    poly_right_fit = poly_right(r_range_right)
    alpha_right = np.degrees(np.arctan(abs(1/coefs_right[0])))
    
    intersection, angle = find_intersection_and_angle_from_arrays(peaks_left_r, peaks_left_x, peaks_right_r, peaks_right_x)
    # print(f"Intersection: {intersection}")
    
    # Average flame angle
    alpha = alpha_left + alpha_right
    
    min_poly_fit = max([poly_left_fit.min(), poly_right_fit.min()])
    
    r_range_left = r_range_left[(poly_left_fit > min_poly_fit) & (poly_left_fit < x_top_raw)]
    r_range_right = r_range_right[(poly_right_fit > min_poly_fit) & (poly_right_fit < x_top_raw)]
    poly_left_fit = poly_left_fit[(poly_left_fit > min_poly_fit) & (poly_left_fit < x_top_raw)]
    poly_right_fit = poly_right_fit[(poly_right_fit > min_poly_fit) & (poly_right_fit < x_top_raw)]
    
    ax1.text(-.09, 1.75, r'$\alpha$', fontsize=40)

    ax1.plot(r_range_left, poly_left_fit, color='k', marker='None', ls='--', zorder=1)
    ax1.plot(r_range_right, poly_right_fit, color='k', marker='None', ls='--', zorder=1)
    
    ax1.set_xlim(x_limits)  # replace with your desired x limits
    ax1.set_ylim(y_limits[0], 2.25)  # replace with your desired y limits
    
    custom_x_ticks = [-.5, .0, .5]
    custom_x_tick_labels =  [f'{tick:.1f}' for tick in custom_x_ticks] # Replace with your desired tick labels
    ax1.set_xticks(custom_x_ticks)
    ax1.set_xticklabels(custom_x_tick_labels)  # Use this line to set custom tick labels
    
    custom_y_ticks = [.5, 1.0, 1.5, 2.]
    custom_y_tick_labels =  [f'{tick:.1f}' for tick in custom_y_ticks] # Replace with your desired tick labels
    ax1.set_yticks(custom_y_ticks)
    ax1.set_yticklabels(custom_y_tick_labels)  # Use this line to set custom tick labels
    
    
    custom_x_ticks = [-.5, .0, .5]
    custom_x_tick_labels =  [f'{tick:.1f}' for tick in custom_x_ticks] # Replace with your desired tick labels
    ax2.set_xticks(custom_x_ticks)
    ax2.set_xticklabels(custom_x_tick_labels)  # Use this line to set custom tick labels
    
    num_ticks = 5
    custom_y_ticks = np.linspace(0, 4e-5, num_ticks) # Replace with your desired tick positions
    # custom_y_tick_labels =  [f'{tick:.1f}' for tick in custom_y_ticks] # Replace with your desired tick labels
    ax2.set_yticks(custom_y_ticks)
    # ax2.set_yticklabels(custom_y_tick_labels)  # Use this line to set custom tick labels
    
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
    ax2.yaxis.get_offset_text().set_fontsize(fontsize)
    
    ax1.tick_params(axis='both', labelsize=fontsize)
    ax2.tick_params(axis='both', labelsize=fontsize)
    
    ax2.grid(True)
    ax2.set_axisbelow(True)
    
    return r_range_left.values, poly_left_fit, r_range_right.values, poly_right_fit, alpha










