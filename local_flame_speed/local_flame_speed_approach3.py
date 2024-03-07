# -*- coding: utf-8 -*-
'''
Created on Mon Mar 20 23:31:58 2023

@author: luuka
'''

#%% IMPORT PACKAGES
import os
import numpy as np
import pickle
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from scipy.interpolate import interp2d
from contour_properties import *
from parameters import *
    
#%% CLOSE ALL FIGURES
plt.close('all')

#%% SET MATPLOTLIB PARAMETERS
# set_mpl_params()

#%% FUNCTIONS
#%%% MAIN

def determine_unburnt_region(contour_corrected, toggle_plot=False):
    
    contour_x, contour_y = contour_corrected[:,0,0], contour_corrected[:,0,1]
    
    # create closed contour to define coordinates (of the interrogation windows) in unburned and burned region 
    contour_open = list(zip(contour_x, contour_y))
    contour_open.append((contour_x[0], contour_y[0]))
    contour_closed_path = mpl.path.Path(np.array(contour_open))
    
    # Initialize the mask as all false (all points burnt)
    unburnt_region = np.full((n_windows_y, n_windows_x), False)
    
    for j, i in np.ndindex((n_windows_y, n_windows_x)):
        
        piv_coord = (x[j][i], y[j][i])
        
        if contour_closed_path.contains_point(piv_coord):
        
            for i_segment in range(0, len(contour_x) - 1):
                
                unburnt_region[j, i] = True
                
                x_A, y_A, x_B, y_B  = contour_x[i_segment], contour_y[i_segment], contour_x[i_segment+1], contour_y[i_segment+1]
                
                segment_start, segment_end = (x_A, y_A), (x_B, y_B)
                distance_threshold = 0.5*np.sqrt(window_size_x**2 + window_size_y**2)
                distance_to_segment = distance_from_flame_front_check(piv_coord, segment_start, segment_end, distance_threshold)
                
                if distance_to_segment < distance_threshold:
                
                    unburnt_region[j, i] = False
                    break
                
            
    if toggle_plot:        
        # Create a figure and axes
        fig, ax = plt.subplots()
        
        # Plot the matrix as a heatmap
        # unburnt_map = ax.imshow(unburnt_region, cmap='binary')
        
        # unburnt_map = ax.imshow(unburnt_region, extent=extent_piv, cmap='binary')
        unburnt_map = ax.pcolor(x, y, unburnt_region, cmap='viridis')
        # Customize the plot
        # ax.set_xticks(np.arange(unburnt_region.shape[1]))
        # ax.set_yticks(np.arange(unburnt_region.shape[0]))
        # ax.set_xticklabels(np.arange(unburnt_region.shape[1]))
        # ax.set_yticklabels(np.arange(unburnt_region.shape[0]))
        # ax.set_xlabel('Column')
        # ax.set_ylabel('Row')
        ax.set_title('unburnt_region')
        #
        ax.set_aspect('equal')
        
        # Show the colorbar
        cbar = ax.figure.colorbar(unburnt_map)
    
    return unburnt_region

def compute_strain_rate_tensor(u, v, unburnt_region):

    dx, dy = window_size_x*1e-3, window_size_y*1e-3 # convert dx to m
    
    # dx, dy = 1, -1 # convert dx to m

    S = np.full((*u.shape, 2, 2), np.nan)
    S_scheme = np.full((*u.shape, 2, 2), 'NaN', dtype=np.dtype('object'))

    for j in range(1, u.shape[0]-1):
        for i in range(1, u.shape[1]-1):
            if not unburnt_region[j, i]:
                continue
            
            # Create variables to hold the derivatives
            dudx = dvdy = cross_deriv = np.nan
            dudx_scheme = dvdy_scheme = cross_deriv_scheme = 'NaN'

            # Central difference scheme if points on both sides are unburnt
            if unburnt_region[j, i+1] and unburnt_region[j, i-1]:
                dudx = (u[j, i+1] - u[j, i-1]) / (2 * dx)
                dudx_scheme = 'central'
            elif unburnt_region[j, i+1]:
                dudx = (u[j, i+1] - u[j, i]) / dx
                dudx_scheme = 'forward'
            elif unburnt_region[j, i-1]:
                dudx = (u[j, i] - u[j, i-1]) / dx
                dudx_scheme = 'backward'

            if unburnt_region[j+1, i] and unburnt_region[j-1, i]:
                dvdy = (v[j+1, i] - v[j-1, i]) / (2 * dy)
                dvdy_scheme = 'central'
            elif unburnt_region[j+1, i]:
                dvdy = (v[j+1, i] - v[j, i]) / dy
                dvdy_scheme = 'forward'
            elif unburnt_region[j-1, i]:
                dvdy = (v[j, i] - v[j-1, i]) / dy
                dvdy_scheme = 'backward'
            
            # Create variables to hold the derivatives
            dudy = dvdx = np.nan
            dudy_scheme = dvdx_scheme = 'NaN'

            if unburnt_region[j+1, i] and unburnt_region[j-1, i]:
                dudy = (u[j+1, i] - u[j-1, i]) / (2 * dy)
                dudy_scheme = 'central'
            elif unburnt_region[j+1, i]:
                dudy = (u[j+1, i] - u[j, i]) / dy
                dudy_scheme = 'forward'
            elif unburnt_region[j-1, i]:
                dudy = (u[j, i] - u[j-1, i]) / dy
                dudy_scheme = 'backward'

            if unburnt_region[j, i+1] and unburnt_region[j, i-1]:
                dvdx = (v[j, i+1] - v[j, i-1]) / (2 * dx)
                dvdx_scheme = 'central'
            elif unburnt_region[j, i+1]:
                dvdx = (v[j, i+1] - v[j, i]) / dx
                dvdx_scheme = 'forward'
            elif unburnt_region[j, i-1]:
                dvdx = (v[j, i] - v[j, i-1]) / dx
                dvdx_scheme = 'backward'

            if np.isfinite(dudy) and np.isfinite(dvdx):
                cross_deriv = 0.5 * (dudy + dvdx)
                cross_deriv_scheme = dudy_scheme if dudy_scheme == dvdx_scheme else 'mismatch'
            
            # If we were able to compute all derivatives, store the strain rate tensor
            if np.isfinite(dudx) and np.isfinite(dvdy) and np.isfinite(cross_deriv):
                S[j, i] = [[dudx, cross_deriv], 
                           [cross_deriv, dvdy]]
                S_scheme[j, i] = [[dudx_scheme, cross_deriv_scheme], 
                                  [cross_deriv_scheme, dvdy_scheme]]
        
        
    divergence_u = np.zeros((u.shape[0], u.shape[1]))
    for j in range(u.shape[0]):
        for i in range(u.shape[1]):
            
            # Extract the strain tensor at the current position
            strain_tensor = S[j, i]
            
            # Calculate the trace of the strain tensor
            trace = np.trace(strain_tensor)
            
            # Store the trace value in the trace field array
            divergence_u[j, i] = trace

    return S, S_scheme, divergence_u
         

def read_piv_data(image_nr):
    
    # Obtain contour number
    contour_nr = image_nr - 1
    
    # Transient file name and scaling parameters from headers of file
    # xyuv_file = os.path.join(piv_dir, f'B{image_nr:04d}.txt')
    
    piv_dir = os.path.join(main_dir,  f'session_{flame.session_nr:03d}', flame.record_name, piv_folder, 'Export')
    piv_file = os.path.join(piv_dir, f'B{image_nr:04d}.csv')

    df_piv = pd.read_csv(piv_file)

    headers = df_piv.columns
    
    # Read u velocity
    pivot_u = pd.pivot_table(df_piv, values=headers[2], index=headers[1], columns=headers[0])
    x_array = pivot_u.columns
    y_array = pivot_u.index
    u = pivot_u.values
    
    # Read v velocity
    pivot_v = pd.pivot_table(df_piv, values=headers[3], index=headers[1], columns=headers[0])
    v = pivot_v.values
    
    # Read absolute velocity
    pivot_V_abs = pd.pivot_table(df_piv, values=headers[4], index=headers[1], columns=headers[0])
    V_abs = pivot_V_abs.values
    
    # Read divergence of u 
    pivot_divergence_u_davis = pd.pivot_table(df_piv, values=headers[11], index=headers[1], columns=headers[0])
    divergence_u_davis = pivot_divergence_u_davis.values
    
    x, y = np.meshgrid(x_array, y_array)
    
    n_windows_x, n_windows_y = len(x_array), len(y_array)
    
    return contour_nr, n_windows_x, n_windows_y, x/D, y/D, u/U_bulk, v/U_bulk, V_abs/U_bulk, divergence_u_davis


def contour_correction(contour_nr, frame_nr=0):
    
    segmented_contour =  flame.frames[frame_nr].contour_data.segmented_contours[contour_nr]
    
    segmented_contour_x = segmented_contour[:,0,0]
    segmented_contour_y = segmented_contour[:,0,1]
    
    # x and y coordinates of the discretized (segmented) flame front 
    contour_x_corrected = segmented_contour_x*window_size_x_raw + x_left_raw
    contour_y_corrected = segmented_contour_y*window_size_y_raw + y_top_raw
    
    # Non-dimensionalize coordinates by pipe diameter
    contour_x_corrected /= D
    contour_y_corrected /= D
    
    contour_x_corrected_array = np.array(contour_x_corrected)
    contour_y_corrected_array = np.array(contour_y_corrected)
    
    # Combine the x and y coordinates into a single array of shape (n_coords, 2)
    contour_corrected_coords = np.array([contour_x_corrected_array, contour_y_corrected_array]).T

    # Create a new array of shape (n_coords, 1, 2) and assign the coordinates to it
    contour_corrected = np.zeros((len(contour_x_corrected_array), 1, 2))
    contour_corrected[:, 0, :] = contour_corrected_coords
    
    return contour_corrected

def determine_local_flame_speed(selected_segments_2, final_segments, colors, ax=None):
    
    # Please note that:
    
    # - final_segment = [selected_segment, x_intersect_ref_advected, y_intersect_ref_advected, x_intersect, y_intersect, i_segment, contour_corrected, flame_front_displacement, S_a]
    # - selected_segment = [x_A, y_A, x_B, y_B, 
    #                        x_mid, y_mid, x_piv, y_piv, 
    #                        V_nx_select, V_ny_select, np.abs(V_n), 
    #                        V_tx_select, V_ty_select, V_t, 
    #                        V_x_select, V_y_select, 
    #                        i_segment, contour_corrected, rotation_matrix,
    #                        x_intersect, y_intersect]
    
    tab20_loop = ListedColormap(tab20.colors*10)
    
    n_digits = 5
    
    for t, selected_segments in enumerate(selected_segments_2):
        
        color_cycle = iter(tab20_loop(np.linspace(0, 1, len(selected_segments))))
        
        for selected_segment in selected_segments:
            
            x_mid = selected_segment[0][4]
            y_mid = selected_segment[0][5]
            
            x_piv = selected_segment[0][6]
            y_piv = selected_segment[0][7]
            
            V_nx = selected_segment[0][8]
            V_ny = selected_segment[0][9]
            V_n = selected_segment[0][10]
            
            x_intersect_ref = selected_segment[1]
            y_intersect_ref = selected_segment[2]
            x_intersect = selected_segment[3]
            y_intersect = selected_segment[4]
            
            S_a = selected_segment[8]
            S_a_dx, S_a_dy = x_intersect-x_intersect_ref, y_intersect-y_intersect_ref
            S_a_angle = np.arctan2(S_a_dy, S_a_dx)
            
            V_n_angle = np.arctan2(V_ny, V_nx)
            
            S_a = np.round(S_a, n_digits)
            S_a_dx = np.round(S_a_dx, n_digits)
            S_a_dy = np.round(S_a_dy, n_digits)
            S_a_angle = np.round(S_a_angle, n_digits)
            
            V_n = np.round(V_n, n_digits)
            V_n_angle = np.round(V_n_angle, n_digits)
            
            # print(V_n_angle, S_a_angle)
            
            if (V_n_angle >= 0) != (S_a_angle >= 0) or V_n_angle != S_a_angle:
                
                S_d = S_a # + V_n
                    
            elif V_n_angle == S_a_angle:
                
                S_d = S_a # - V_n
            
            else:
                
                S_d = np.nan
                print("Local flame speed could not be detected, check this image!")
                
                
            selected_segment.append(S_d)
            
            final_segments.append(selected_segment)
            
            if toggle_plot:
                
                color = next(color_cycle)    
                ax.plot(x_piv, y_piv, color=color, marker='s')
                ax.plot(x_mid, y_mid, color=color, marker='s')
                ax.plot([x_piv, x_intersect_ref], [y_piv, y_intersect_ref], color=color, ls='--',marker='None')
                ax.plot([x_intersect_ref, x_intersect], [y_intersect_ref, y_intersect], color=color, ls='--',marker='None')
                
                plot_local_flame_speed(ax, selected_segment, color) 
    
    return final_segments
    
def local_flame_speed_from_double_image_single_frame(image_nr, n_nearest_coords=1, threshold_angle=180):
    
    # Set color map
    colormap = jet(np.linspace(0, 1, 2))
    
    # Create color iterator for plotting the data
    colormap_iter_time = iter(colormap)
    
    n_time_steps = 1
    image_nr_t0 = image_nr
    image_nr_t1 = image_nr_t0 + n_time_steps
    
    image_nrs = [image_nr_t0, image_nr_t1]
    
    dt = n_time_steps*(1/flame.image_rate)
    
    selected_segments_1, selected_segments_2, final_segments = [], [], []
    
    colors = []
    
    for i, image_nr in enumerate(image_nrs):
        
        # Read PIV data
        contour_nr_t0, n_windows_x, n_windows_y, x_t0, y_t0, u_t0, v_t0, V_abs_t0, _ = read_piv_data(image_nrs[0])
        contour_nr_t1, n_windows_x, n_windows_y, x_t1, y_t1, u_t1, v_t1, V_abs_t1, _ = read_piv_data(image_nrs[1])
        
        # Contour correction RAW --> (non-dimensionalized) WORLD
        contour_corrected_t0 = contour_correction(contour_nr_t0)
        contour_corrected_t1 = contour_correction(contour_nr_t1)
        
        # First selection round: Checks if a velocity vector crosses a contour segment at the reference time step. 
        # The amount of velocity vectors considered for selection is set with n_nearest_coords.
        selected_segments_1_t0 = first_segment_selection_procedure(contour_corrected_t0, n_windows_x, n_windows_y, x_t0, y_t0, u_t0, v_t0, n_nearest_coords)
        selected_segments_1.append(selected_segments_1_t0)
        
        # if i == 0:
        #     print(selected_coords)
        # Second selection round: Checks if the normal component of the velocity vector found at the reference time step crosses 
        # the segment at the reference time step and a segment in the 'other' time step. Another restriction is that between to selected 
        # segments (segments at t0 and t1) may not be greater than threshold_angle (in degrees)
        selected_segments_2_t0 = second_segment_selection_procedure(contour_corrected_t1, selected_segments_1_t0, dt, threshold_angle)
        selected_segments_2.append(selected_segments_2_t0)
        
        # print(y[2])
        # print(len(selected_segments_1_ref))
        # print(len(selected_segments_2_ref))
        
        # tab20_loop = ListedColormap(tab20.colors*10)
        
        # color_cycle = iter(tab20_loop(np.linspace(0, 1, len(selected_coords))))
        
        if toggle_plot:
            
            if i == 0:
                
                # Create figure for velocity field + frame front contour
                fig, ax = plt.subplots()
                
                # for j, selected_coord in enumerate(selected_coords):
                    
                #     (_, _, _, x_piv_ij, y_piv_ij, _, _, i_segment, x_mid, y_mid, _, _) = selected_coord
                    
                #     color = next(color_cycle)    
                #     ax.plot(x_piv_ij, y_piv_ij, color=color, marker='s')
                #     ax.plot(x_mid, y_mid, color=color, marker='o')
                    
                    # if j == 23:
                        
                    #     print(x_piv_ij, y_piv_ij, i_segment)
                    
                
                # ax.set_title('{}\n $D_{{in}}$={} mm \n $\phi$={}, $H_2\%$={}, $Re_{{D_{{in}}}}$={}\nImage: {} - {}, Frame: {}'.format(
                #             name, flame.D_in, flame.phi, flame.H2_percentage, flame.Re_D, image_nr_t0, image_nr_t1, frame_nr))

                # for element in selected_segments_1_ref:
                    # ax.plot(element[4], element[5], marker='s', color='m')
                # print('---------------------')
                
                if normalized:
                    
                    ax.set_xlabel(r'$r/D$')
                    ax.set_ylabel(r'$x/D$')
                    
                else:
                    
                    ax.set_xlabel(r'$r$ [mm]')
                    ax.set_ylabel(r'$x$ [mm]')
                    
                    # x_left, x_right = 5, 14
                    # y_left, y_right = 3, 9
                    # zoom = [x_left, x_right, y_left, y_right]
                    # ax.set_xlim(np.array([zoom[0], zoom[1]]))
                    # ax.set_ylim(np.array([zoom[2], zoom[3]]))
                
                # Plot velocity vectors
                # plot_velocity_vectors(fig, ax, x_ref, y_ref, u_ref, v_ref, scale_vector)
                
                # Plot velocity field
                quantity = V_abs_t0
                
                # Choose 'imshow' or 'pcolor' by uncommenting the correct line
                plot_velocity_field_pcolor(fig, ax, x_t0, y_t0, quantity)
                plot_velocity_vectors(fig, ax, x_t0, y_t0, u_t0, v_t0, scale_vector)
    
            # Create timestamp for plot
            timestamp = ((image_nrs[0]-image_nr_t0)/flame.image_rate)*1e3
            
            # Plot flame front contour
            c = next(colormap_iter_time)
            colors.append(c)
            ax = plot_contour_single_color(ax, contour_corrected_t0, timestamp, c)
            
            # Turn on legend
            ax.legend()
        
            # Tighten figure
            # fig.tight_layout()
        
        # Important: This operation reverses the image_nrs, so that the reference time step changes from image_t0 to image_t1
        image_nrs.reverse()
    
    selected_segments_2 = [selected_segments_2[0]]
    final_segments = determine_local_flame_speed(selected_segments_2, final_segments, colors, ax=ax if toggle_plot else None)
        
    return final_segments

def local_flame_speed_from_time_resolved_single_frame(image_nr, n_nearest_coords=1, threshold_angle=180):
    
    # set color map
    colormap = jet(np.linspace(0, 1, 2))
    
    # create color iterator for plotting the data
    colormap_iter_time = iter(colormap)
    
    image_nr_t0 = image_nr
    image_nr_t1 = image_nr_t0 + 1
    
    dt = (1/flame.image_rate)
    
    selected_segments_1, selected_segments_2, final_segments = [], [], []
    
    colors = []
    
    # read PIV data
    contour_nr_t0, n_windows_x, n_windows_y, x_ref, y_ref, u_ref, v_ref, V_abs_ref, _ = read_piv_data(image_nr_t0)
    contour_nr_t1, n_windows_x, n_windows_y, x, y, u, v, V_abs, _ = read_piv_data(image_nr_t1)
    
    contour_nrs = [contour_nr_t0, contour_nr_t1]
    
    for i, contour_nr in enumerate(contour_nrs):
        
        # contour correction RAW --> (non-dimensionalized) WORLD
        contour_corrected_ref = contour_correction(contour_nrs[0])
        contour_corrected = contour_correction(contour_nrs[1])
        
        # first selection round: Checks if a velocity vector crosses a contour segment at the reference time step. 
        # The amount of velocity vectors considered for selection is set with n_nearest_coords.
        selected_segments_1_ref, selected_coords = first_segment_selection_procedure(contour_corrected_ref, n_windows_x, n_windows_y, x_ref, y_ref, u_ref, v_ref, n_nearest_coords)
        selected_segments_1.append(selected_segments_1_ref)
        
        # second selection round: Checks if the normal component of the velocity vector found at the reference time step crosses 
        # the segment at the reference time step and a segment in the 'other' time step. Another restriction is that between to selected 
        # segments (segments at t0 and t1) may not be greater than threshold_angle (in degrees)
        selected_segments_2_ref= second_segment_selection_procedure(contour_corrected, selected_segments_1_ref, dt, threshold_angle)
        selected_segments_2.append(selected_segments_2_ref)
        
        if toggle_plot:
            
            if i == 0:
                
                # create figure for velocity field + frame front contour
                fig, ax = plt.subplots()
                
                ax.set_title('Flame ' + str(flame_nr) + ': ' + '$\phi$=' + str(flame.phi) + ', $H_{2}\%$=' + str(flame.H2_percentage)+ '\n' +
                             '$D_{in}$=' + str(flame.D_in) + ' mm, $Re_{D_{in}}$=' + str(flame.Re_D) + '\n' + 
                             'Image: ' + str(image_nr_t0) + ' - ' + str(image_nr_t1) + ', time-resolved')
                
                if normalized:
                    
                    ax.set_xlabel(r'$r/D$')
                    ax.set_ylabel(r'$x/D$')
                    
                else:
                    
                    ax.set_xlabel(r'$r$ [mm]')
                    ax.set_ylabel(r'$x$ [mm]')
                
                # plot velocity vectors
                plot_velocity_vectors(fig, ax, x_ref, y_ref, u_ref, v_ref, scale_vector)
                
                # plot velocity field
                quantity = V_abs_ref
                
                # choose 'imshow' or 'pcolor' by uncommenting the correct line
                plot_velocity_field_pcolor(fig, ax, x_t0, y_t0, quantity)
    
            # create timestamp for plot
            timestamp = ((contour_nrs[0]-contour_nr_t0)/flame.image_rate)*1e3
            
            # plot flame front contour
            c = next(colormap_iter_time)
            colors.append(c)
            ax = plot_contour_single_color(ax, contour_corrected_ref, timestamp, c)
            
            # turn on legend
            ax.legend()
            
            # tighten figure
            fig.tight_layout()
        
        # Important: This operation reverses the image_nrs, so that the reference time step changes from image_t0 to image_t1
        contour_nrs.reverse()
    
    selected_segments_2 = [selected_segments_2[0]]
    final_segments = determine_local_flame_speed(selected_segments_2, final_segments, colors, ax=ax if toggle_plot else None)
        
    return final_segments

def calculate_kl(x_piv, y_piv, vec_x, vec_y, x_A, y_A, dx, dy):
    
    if (vec_y*dx - vec_x*dy) == 0:
        k = 0
    else:
        k = (vec_x*(y_A-y_piv) - vec_y*(x_A-x_piv))/(vec_y*dx - vec_x*dy)

    if vec_x == 0:
        l = 0
    else:
        l = (k*dx + x_A - x_piv)/vec_x
        
    return k, l

def check_intersection_vector_and_segment(p1, p2, p3, p4):
    # Nested function to calculate the cross product of two points
    def cross_product(p1, p2):
        return p1[0] * p2[1] - p2[0] * p1[1]

    # Nested function to determine the direction of three points
    def direction(p1, p2, p3):
        # The direction is determined by calculating the cross product of the vector from p1 to p2, and from p1 to p3
        return cross_product((p3[0]-p1[0], p3[1]-p1[1]), (p2[0]-p1[0], p2[1]-p1[1]))

    # Nested function to check if a point is on a line segment
    def on_segment(p1, p2, p3):
        # p3 is on the line segment p1p2 if its x and y coordinates are between the x and y coordinates of p1 and p2
        return min(p1[0], p2[0]) <= p3[0] <= max(p1[0], p2[0]) and min(p1[1], p2[1]) <= p3[1] <= max(p1[1], p2[1])

    # Calculate the direction of the four points relative to each other
    d1 = direction(p3, p4, p1)
    d2 = direction(p3, p4, p2)
    d3 = direction(p1, p2, p3)
    d4 = direction(p1, p2, p4)

    # Check if the line segments intersect
    intersect = False
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        intersect = True
    elif d1 == 0 and on_segment(p3, p4, p1):
        intersect = True
    elif d2 == 0 and on_segment(p3, p4, p2):
        intersect = True
    elif d3 == 0 and on_segment(p1, p2, p3):
        intersect = True
    elif d4 == 0 and on_segment(p1, p2, p4):
        intersect = True

    if intersect:
        # Calculate the intersection point using the formula derived from the equations of the lines
        t = ((p1[0]-p3[0]) * (p3[1]-p4[1]) - (p1[1]-p3[1]) * (p3[0]-p4[0])) / ((p1[0]-p2[0]) * (p3[1]-p4[1]) - (p1[1]-p2[1]) * (p3[0]-p4[0]))
        intersection_point = (p1[0] + t * (p2[0]-p1[0]), p1[1] + t * (p2[1]-p1[1]))
        return intersect, (intersection_point[0], intersection_point[1]) 

    return intersect, (None, None)


def distance_from_flame_front_check(point, segment_start, segment_end, distance_threshold):
    # Calculate the vector representing the segment
    segment_vector = [segment_end[0] - segment_start[0], segment_end[1] - segment_start[1]]
    
    # Calculate the vector from the segment start point to the given point
    start_to_point_vector = [point[0] - segment_start[0], point[1] - segment_start[1]]
    
    # Calculate the dot product of the two vectors
    dot_product = start_to_point_vector[0] * segment_vector[0] + start_to_point_vector[1] * segment_vector[1]
    
    # Calculate the squared length of the segment
    segment_length_squared = segment_vector[0] ** 2 + segment_vector[1] ** 2
    
    # Calculate the normalized distance from the start point to the projected point
    normalized_distance = dot_product / segment_length_squared
    
    if normalized_distance < 0 or normalized_distance > 1:
        # The projected point is outside the segment range
        distance_to_start = np.sqrt((point[0] - segment_start[0]) ** 2 + (point[1] - segment_start[1]) ** 2)
        distance_to_end = np.sqrt((point[0] - segment_end[0]) ** 2 + (point[1] - segment_end[1]) ** 2)
        return min(distance_to_start, distance_to_end)
    
    # Calculate the coordinates of the projected point on the segment
    projection_point = [segment_start[0] + normalized_distance * segment_vector[0],
                        segment_start[1] + normalized_distance * segment_vector[1]]
    
    # Calculate the distance from the given point to the projected point
    distance_to_segment = np.sqrt((point[0] - projection_point[0]) ** 2 + (point[1] - projection_point[1]) ** 2)
    
    return distance_to_segment


def find_intersections(line_point, line_dir, contour):
    
    intersections, distances = [], []
    
    for i in range(contour.shape[0]):
        # Get the next index in a cyclic way (to close the contour)
        j = (i + 1) % contour.shape[0]
        
        # Define the line segment
        segment_start = contour[i, 0]
        segment_end = contour[j, 0]
        segment_dir = segment_end - segment_start

        # Matrix
        mat = np.array([line_dir, -segment_dir]).T

        # If the determinant is zero, the lines are parallel (or coincident, if overlapping)
        if np.linalg.det(mat) == 0:
            continue

        # Solve the system of equations
        try:
            kl = np.linalg.solve(mat, segment_start - line_point)
        except np.linalg.LinAlgError:
            continue
        
        # Check if the solution is within the bounds of the line segment (0 <= s <= 1) and on the ray (k >= 0)
        if 0 <= kl[1] <= 1 and kl[0] >= 0:
            # Calculate the intersection point
            intersection = line_point + kl[0]*line_dir
            intersections.append(intersection)
            
            # Calculate the distance between line_point and the intersection point
            distance = np.linalg.norm(line_point - intersection)
            distances.append(distance)
            
    return intersections, distances

def first_segment_selection_procedure(contour_corrected, n_windows_x, n_windows_y, x, y, u, v, n_nearest_coords):
    
    contour_x, contour_y = contour_corrected[:,0,0], contour_corrected[:,0,1]
    
    unburnt_region = determine_unburnt_region(contour_corrected)
    
    S, S_scheme, divergence_u = compute_strain_rate_tensor(u, v, unburnt_region)
    
    selected_segments_1 = []
    
    for i_segment in range(0, len(contour_x) - 1):
        
        candidate_coords_u = []
        
        x_A, y_A, x_B, y_B  = contour_x[i_segment], contour_y[i_segment], contour_x[i_segment+1], contour_y[i_segment+1]
        x_mid, y_mid = (x_A + x_B)/2, (y_A + y_B)/2
        dx, dy = x_B-x_A, y_B-y_A
        segment_start, segment_end = (x_A, y_A), (x_B, y_B)
        
        # Segment length and angle
        L, segment_angle = segment_properties(dy, dx)
        
        for j, i in np.ndindex((n_windows_y, n_windows_x)):
            
            x_piv_ij, y_piv_ij = x[j][i], y[j][i]
            u_ij, v_ij = u[j][i], v[j][i]
            
            if np.isfinite(divergence_u[j, i]):
                
                distance_to_segment_midpoint = np.sqrt((x_mid - x_piv_ij)**2 + (y_mid - y_piv_ij)**2)

                V_direction = np.array([u_ij, v_ij])
                V_magnitude = np.linalg.norm(V_direction)
                V_extension = 100*V_direction/V_magnitude
                V_coord0 = (x_piv_ij, y_piv_ij)
                V_coord1 = (V_coord0[0] + V_extension[0], V_coord0[1] +  V_extension[1])
                
                intersect, (x_intersect, y_intersect) = check_intersection_vector_and_segment(segment_start, segment_end, V_coord0, V_coord1)
                
                coord_u = [j, i, distance_to_segment_midpoint, x_piv_ij, y_piv_ij, u_ij, v_ij, i_segment, x_mid, y_mid, x_intersect, y_intersect]
                
                # check if velocity vector normal to segment intersects with the segment of the chosen time step
                # if distance_A <= L and distance_B <= L:
                if intersect:
                    
                    distance_to_segment_intersection = np.sqrt((x_intersect - x_piv_ij)**2 + (y_intersect - y_piv_ij)**2)
                    
                    if distance_to_segment_midpoint < 4*L:
                        
                        line_point = np.array([x_piv_ij, y_piv_ij])  # Adjust these values as necessary
                        line_dir = np.array([u_ij, v_ij])  # Adjust these values as necessary
                        all_intersections, all_distances = find_intersections(line_point, line_dir, contour_corrected)
                        
                        # print("-"*32)
                        # print(x_intersect, y_intersect, distance_to_segment_intersection)
                        # print(all_intersections, all_distances)
                        
                        if any(np.round(distance, 3) < np.round(distance_to_segment_intersection, 3) for distance in all_distances):
                        
                            pass
                        
                        else:
                    
                            candidate_coords_u.append(coord_u)
        
        candidate_coords_u.sort(key = lambda i: i[2])
        
        # If candidate_coords_u is not empty
        if candidate_coords_u:
            
            selected_coord_u = candidate_coords_u[0]
            # selected_coords.append(selected_coord_u)
        
            # Define the rotation matrix (clockwise)
            c, s = np.cos(segment_angle), np.sin(segment_angle)
            
            rotation_matrix = np.array([[c, s], 
                                        [-s, c]])
            
            x_piv, y_piv = selected_coord_u[3], selected_coord_u[4]
            x_intersect, y_intersect = selected_coord_u[10], selected_coord_u[11]
            
            # Stack Vx and Vy into a velocity vector
            V_x_select, V_y_select = selected_coord_u[5], selected_coord_u[6]
            V = np.array([V_x_select, V_y_select])
            
            # Compute the normal and tangential velocities
            V_t, V_n = np.dot(rotation_matrix, V)
            
            # if i_segment == 0 or i_segment == 4:
            #     print(selected_coord_u, V, V_t, V_n)
            
            # Compute the Cartesian components of the normal and tangential velocities
            V_tx_select, V_ty_select = np.dot(rotation_matrix.T, np.array([V_t, 0]))
            V_nx_select, V_ny_select = np.dot(rotation_matrix.T, np.array([0, V_n]))
            
            # Segment may not touch the top boundary of the image
            if (y_A or y_B) >= y_top_raw:
               
                pass
            
            else:
                
                selected_segment_1 = [x_A, y_A, x_B, y_B, 
                                      x_mid, y_mid, x_piv, y_piv, 
                                      V_nx_select, V_ny_select, np.abs(V_n), 
                                      V_tx_select, V_ty_select, V_t, 
                                      V_x_select, V_y_select, 
                                      i_segment, contour_corrected, rotation_matrix,
                                      x_intersect, y_intersect]
                
                selected_segments_1.append(selected_segment_1)
                    
    return selected_segments_1


def second_segment_selection_procedure(contour_corrected, selected_segments_1, dt, threshold_angle):
    
    contour_x, contour_y = contour_corrected[:,0,0], contour_corrected[:,0,1]
    
    selected_segments_2 = []
    
    for selected_segment in selected_segments_1:
        
        # we select the segment that is closest to a select segment in selected_segments_1
        counter = 0
        
        flame_front_displacement_ref = 1000
        
        # check if velocity vector normal to the segment of the reference time step intersects the segment itself 
        x_A, y_A, x_B, y_B  = selected_segment[0], selected_segment[1], selected_segment[2], selected_segment[3]
        x_mid_ref, y_mid_ref = selected_segment[4], selected_segment[5] #(x_A + x_B)/2, (y_A + y_B)/2
        dx_ref, dy_ref = x_B-x_A, y_B-y_A
        
        # segment length and angle
        L_ref, segment_angle_ref = segment_properties(dy_ref, dx_ref)
        
        # velocity data of reference time step
        x_piv = selected_segment[6]
        y_piv = selected_segment[7]
        V_nx = selected_segment[8]
        V_ny = selected_segment[9]
        V_n = selected_segment[10]
        V_x = selected_segment[14]
        V_y = selected_segment[15]
        i_segment_ref = selected_segment[16]
        x_intersect_ref = selected_segment[19]
        y_intersect_ref = selected_segment[20]
        
        x_intersect_ref_advected = x_intersect_ref + (V_x*1e3)*dt
        y_intersect_ref_advected = y_intersect_ref + (V_y*1e3)*dt
            
        for i_segment in range(0, len(contour_x) - 1):
            
            x_A, y_A, x_B, y_B  = contour_x[i_segment], contour_y[i_segment], contour_x[i_segment+1], contour_y[i_segment+1]
            x_mid, y_mid = (x_A + x_B)/2, (y_A + y_B)/2
            dx, dy = x_B-x_A, y_B-y_A
            
            # segment length and angle
            L, segment_angle = segment_properties(dy, dx)
            
            k, l = calculate_kl(x_intersect_ref_advected, y_intersect_ref_advected, dy_ref, -dx_ref, x_A, y_A, dx, dy)
            
            x_intersect = x_A + k*dx
            y_intersect = y_A + k*dy
            
            distance_A = np.sqrt((x_intersect - x_A)**2 + (y_intersect - y_A)**2)
            distance_B = np.sqrt((x_intersect - x_B)**2 + (y_intersect - y_B)**2)
            
            # check if velocity vector normal to segment intersects with the segment of the chosen time step
            if distance_A <= L and distance_B <= L:
                
                flame_front_displacement = np.sqrt((x_intersect_ref_advected - x_intersect)**2 + (y_intersect_ref_advected - y_intersect)**2)
                
                if (flame_front_displacement < flame_front_displacement_ref and (flame_front_displacement < 3*L)):
                    
                    flame_front_displacement_ref = flame_front_displacement
                    
                    if (np.abs(segment_angle - segment_angle_ref)) < np.deg2rad(threshold_angle):
                        
                        S_a = flame_front_displacement*(D*1e-3)/dt
                        
                        S_a /= U_bulk
                             
                        selected_segment_2 = [selected_segment, x_intersect_ref_advected, y_intersect_ref_advected, x_intersect, y_intersect, i_segment, contour_corrected, flame_front_displacement, S_a]
                        
                        if counter < 1:
                            
                            # print(x_intersect_ref, y_intersect_ref, x_intersect_ref_advected, y_intersect_ref_advected)
                            selected_segments_2.append(selected_segment_2)
                            counter += 1
                            
                        else:
                            
                            selected_segments_2[-1] = selected_segment_2
                           
    return selected_segments_2           

#%%% PLOT

def plot_velocity_vectors(fig, ax, x, y, u, v, scale_vector):
    ax.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=scale_vector, ls='-', fc='k', ec='k')

def plot_streamlines(x, y, u, v):
    """Plot streamlines from a structured grid with possibly unequal spacing.

    Args:
    x, y: 2D arrays of grid point coordinates.
    u, v: 2D arrays of vector components.
    """
    
    # Regularly spaced grid spanning the domain of x and y 
    xi = np.linspace(x.min(), x.max(), x.shape[1])
    yi = np.linspace(y.min(), y.max(), y.shape[0])
    
    # Bicubic interpolation
    ui = interp2d(x, y, u)(xi, yi)
    vi = interp2d(x, y, v)(xi, yi)
    
    ax.streamplot(xi, yi, ui, vi, broken_streamlines=False)

def plot_field(fig, ax, x, y, quantity):
    
    # dx, dy = window_size_x*1e-3, window_size_y*1e-3 # convert dx to m
    
    dx, dy = 1, 1
    
    # ax.imshow(quantity, extent=extent_piv, cmap='viridis')
    
    ax.pcolor(x, y, quantity, cmap='viridis')
    
    j, i = yi_piv_special, xi_piv_special 
    
    # print((u[j,i+1] - u[j,i-1]) / (2*window_size_x)) 
    # print((v[j+1,i] - v[j-1,i]) / (2*window_size_y)) 
    
    # Compute the coordinates for the box
    left = x[yi_piv_special, xi_piv_special] - 0.5 * window_size_x_abs
    bottom = y[yi_piv_special, xi_piv_special] - 0.5 * window_size_y_abs

    # Create a rectangle patch
    rect = patches.Rectangle((left, bottom), window_size_x_abs, window_size_y_abs, linewidth=1, edgecolor='r', facecolor='none')
    
    # Add the rectangle patch to the plot
    ax.add_patch(rect)

    ax.text(x[j,i-1], y[j,i-1], str(np.round(u[j,i-1]/dx, 2)), color='r')
    ax.text(x[j,i+1], y[j,i+1], str(np.round(u[j,i+1]/dx, 2)), color='r')
    
    # ax.text(x[j,i-1], y[j,i-1], str(np.round(u[i,i-1]/window_size_x, 2)), color='r')
    # ax.text(x[j,i+1], y[j,i+1], str(np.round(u[i,i+1]/window_size_x, 2)), color='r')
    
    ax.text(x[j-1,i], y[j-1,i], str(np.round(v[j-1,i]/dy, 2)), color='b')
    ax.text(x[j+1,i], y[j+1,i], str(np.round(v[j+1,i]/dy, 2)), color='b')
    
    # ax.set_xlim(np.array([10, 18]))
    # ax.set_ylim(np.array([-10, 0]))
    
    # ax.set_xlim(np.array([x[yi_piv_special, xi_piv_special] - 2, x[yi_piv_special, xi_piv_special] + 2]))
    # ax.set_ylim(np.array([y[yi_piv_special, xi_piv_special] - 2, y[yi_piv_special, xi_piv_special] + 2]))
    
    ax.plot(x[yi_piv_special, xi_piv_special], y[yi_piv_special, xi_piv_special], marker='o', c='r')

    # ax.text(x[i-1,i], y[i-1,i], str(np.round(u[i-1,i]/window_size_y, 2)), color='r')
    # ax.text(x[i+1,i], y[i+1,i], str(np.round(u[i+1,i]/window_size_y, 2)), color='r')
    
    # ax.text(x[i,i-1], y[i,i-1], str(np.round(v[i,i-1]/window_size_x, 2)), color='b')
    # ax.text(x[i,i+1], y[i,i+1], str(np.round(v[i,i+1]/window_size_x, 2)), color='b')

    
    # ax.text(x[3,2], y[3,2], str(i), color='k')
    
    # ax.plot(1,1,'ro')
    # set aspect ratio of plot
    ax.set_aspect('equal')
    
    
def plot_velocity_field_pcolor(fig, ax, x, y, quantity):
    
    quantity_plot = ax.pcolor(x, y, quantity, cmap='viridis')
    
    quantity_plot.set_clim(0, quantity.max())
    # quantity_plot.set_clim(0, 10)
    
    # set aspect ratio of plot
    ax.set_aspect('equal')
    
    bar = fig.colorbar(quantity_plot)
    bar.set_label(r'$|V|$ [ms$^-1$]') #('$u$/U$_{b}$ [-]')


def plot_contour_single_color(ax, contour, timestamp=0, c='b'):
    
    contour_x, contour_y =  contour[:,0,0], contour[:,0,1]
    
    # plot flame front contour
    # label = r'$t_0$ = ' + str(timestamp) + ' $ms$'
    label = r'$t=' + str(timestamp) + '$' + ' $ms$'
    ax.plot(contour_x, contour_y, color=c, marker='o', ls='-', lw=2, label=label)
    
    # set aspect ratio of plot 
    ax.set_aspect('equal')
    
    return ax


def plot_segmented_contour_multi_color(ax, contour, timestamp, colormap):
    
    contour_x, contour_y =  contour[:,0,0], contour[:,0,1]
    
    for i in range(0, len(contour_x) - 1):
        
        c = next(colormap_iter_space)

        x_A, y_A, x_B, y_B  = contour_x[i], contour_y[i], contour_x[i+1], contour_y[i+1]
        
        ax.plot((x_A, x_B), (y_A, y_B), color=c, marker='None', linestyle='-', linewidth=2)
                            
    # set aspect ratio of plot 
    ax.set_aspect('equal')
        
    return ax


def plot_contour(contour_nr):
    
    contour_corrected = contour_correction(contour_nr)
    
    fig, ax = plt.subplots()
    ax = plot_contour_single_color(ax, contour_corrected, c=jet(0))
    
    # ax.set_xlabel('$r/D$')
    # ax.set_ylabel('$x/D$')
    # ax.set_title('Flame ' + str(flame_nr) + ': ' + '$\phi$=' + str(flame.phi) + ', $H_{2}\%$=' + str(flame.H2_percentage)+ '\n' +
    #               '$D_{in}$=' + str(flame.D_in) + ' mm, $Re_{D_{in}}$=' + str(flame.Re_D) + '\n' + 
    #               'Image \#: ' + str(image_nr))
    
    fig.tight_layout()
    
    return fig, ax

def plot_segment_nrs(ax, contour_nr):
    
    contour_corrected = contour_correction(contour_nr)
    contour_segments = flame.frames[frame_nr].contour_data.slopes_of_segmented_contours[contour_nr]
    
    for i, segment in enumerate(contour_segments):
        
        x_coord_text = (contour_corrected[i,0,0] + contour_corrected[i+1,0,0])/2
        y_coord_text = (contour_corrected[i,0,1] + contour_corrected[i+1,0,1])/2
        ax.text(x_coord_text, y_coord_text, str(i), color='k')
    
    
def plot_slopes(ax, contour_nr):
    
    contour_corrected = contour_correction(contour_nr)
    contour_slopes = flame.frames[frame_nr].contour_data.slopes_of_segmented_contours[contour_nr]
    
    for i, slope in enumerate(contour_slopes):
        
        # this is because slopes_of_segmented_contours was calculated with the origin of the image,
        # which is top left instead of the conventional bottom left
        
        slope = np.abs(slope)
        
        x_coord_text = (contour_corrected[i,0,0] + contour_corrected[i+1,0,0])/2
        y_coord_text = (contour_corrected[i,0,1] + contour_corrected[i+1,0,1])/2
        ax.text(x_coord_text, y_coord_text, str(np.round(slope, 3)), color='k')
    
    
def plot_slope_changes(ax, contour_nr):
    
    contour_corrected = contour_correction(contour_nr)
    contour_slope_changes = flame.frames[frame_nr].contour_data.slope_changes_of_segmented_contours[contour_nr]
    
    for i, slope_change in enumerate(contour_slope_changes):
        
        # this is because slopes_of_segmented_contours was calculated with the origin of the image,
        # which is top left instead of the conventional bottom left
        # slope_change = -slope_change
        
        x_coord_text = contour_corrected[i+1,0,0] 
        y_coord_text = contour_corrected[i+1,0,1]
        # ax.text(x_coord_text, y_coord_text, "{:.3f}_{}".format(slope_change, i), color='k')
        ax.text(x_coord_text, y_coord_text, str(np.round(slope_change, 3)), color='k')
         
        
def plot_local_flame_speed(ax, final_segment, color):
    
    segment = final_segment[0]
    
    x_piv = segment[6]
    y_piv = segment[7]
    V_nx = segment[8]
    V_ny = segment[9]
    V_n = segment[10]
    V_tx = segment[11]
    V_ty = segment[12]
    V_t = segment[13]
    V_x = segment[14]
    V_y = segment[15]
    x_intersect_ref = segment[19]
    y_intersect_ref = segment[20]
    x_intersect_ref_advected = final_segment[1]
    y_intersect_ref_advected = final_segment[2]
    x_intersect = final_segment[3]
    y_intersect = final_segment[4]
    flame_front_displacement = final_segment[7]
    S_a = final_segment[8]
    S_d = final_segment[9]
    
    # ax.text(x_intersect+0.2, y_intersect-0.1, r'$\vec{V_{n}}$' , color=color, fontsize=18)
    # ax.text(x_intersect+0.05, y_intersect+0.35, r'$\vec{V}$' , color='k', fontsize=18)
    
    V_dir = np.array([V_x, V_y])
    V_abs = np.linalg.norm(V_dir)
    
    ax.text(x_piv, y_piv, np.round(V_abs, 3), c='k')
    
    # ax.text(x_intersect, y_intersect, str(np.round(S_a, 3)), color='k')
    
    ax.text((x_intersect_ref + x_intersect)/2 , (y_intersect_ref + y_intersect)/2, np.round(S_d, 3), color='w')
    
    markersize = 8
    mew = 1
    ax.plot(x_intersect_ref, y_intersect_ref, c=color, marker='^', ms=markersize, mew=mew, mec='k')
    ax.plot(x_intersect_ref_advected, y_intersect_ref_advected, c='r', marker='^', ms=markersize, mew=mew, mec='k')
    ax.plot(x_intersect, y_intersect, c=color, marker='^', ms=markersize, mew=mew, mec='k')
    ax.plot(x_piv, y_piv, c=color, marker='o', ms=markersize, mec='k')
    
    # ax.quiver(x_piv, y_piv, V_tx, V_ty, angles='xy', scale_units='xy', scale=scale_vector, ls='-', fc='g', ec='g', width=0.015, headwidth=3, headlength=5)
    # ax.quiver(x_piv, y_piv, V_nx, V_ny, angles='xy', scale_units='xy', scale=scale_vector, ls='-', fc=color, ec=color, width=0.015, headwidth=3, headlength=5)
    ax.quiver(x_piv, y_piv, V_x, V_y, angles='xy', scale_units='xy', scale=scale_vector, ls='-', fc=color, ec=color, width=0.015, headwidth=3, headlength=5)
    # ax.quiver(x_piv, y_piv, V_x, V_y, ls='-', fc=color, ec=color, width=0.015, headwidth=3, headlength=5)
    
    # m = V_ny/V_nx
    # x_working_line = range(4, 10)
    # y_working_line = [m * (x_i - x_piv) + y_piv for x_i in x_working_line]
    
    # ax.plot(x_working_line, y_working_line, color=color, ls='--')
    
    # markersize = 10
    # mew = 2
    # ax.plot(x_intersect_ref, y_intersect_ref, c='r', marker='x', ms=markersize, mew=mew)
    # ax.plot(x_intersect, y_intersect, c='r', marker='x', ms=markersize, mew=mew)
    # ax.plot(x_piv, y_piv, c='k', marker='o', ms=8)
    
    # distance = np.sqrt((x_intersect - x_intersect_ref)**2 + (y_intersect - y_intersect_ref)**2)
    
    # dx, dy = x_intersect_ref - x_intersect, y_intersect_ref - y_intersect 
    # alpha = np.arctan(dy/dx)
    # ax.quiver(x_intersect_ref, y_intersect_ref, -distance*np.cos(alpha), -distance*np.sin(alpha), angles='xy', scale_units='xy', scale=1, ls='-', fc='w', ec='w', width=0.015, headwidth=3, headlength=5)
    # ax.text(x_intersect+0.2, y_intersect-0.1, r'$\vec{S_{d}}$' , color='w', fontsize=18)

#%%% AXUILIARY PLOT
def save_animation(camera, filename, interval=500, repeat_delay=1000):
    
    animation = camera.animate(interval = interval, repeat = True, repeat_delay = repeat_delay)
    animation.save(filename + '.gif', writer='pillow')

#%% LOCAL FLAME SPEED METHODS DICTIONARY
local_flame_speed_methods = {
    
    0 : (local_flame_speed_from_double_image_single_frame),
    # 1 : (local_flame_speed_from_single_image_double_frame),
    2 : (local_flame_speed_from_time_resolved_single_frame)
}

    
#%% START OF CODE
# Before executing code, Python interpreter reads source file and define few special variables/global variables. 
# If the python interpreter is running that module (the source file) as the main program, it sets the special __name__ variable to have a value “__main__”. If this file is being imported from another module, __name__ will be set to the module’s name. Module’s name is available as value to __name__ global variable. 
if __name__ == '__main__':
    
    #%%% SET PARAMETERS OF INTEREST
    
    # !!! SET THE CORRECT PARAMETERS IN PARAMETERS.PY !!!
    
    # name = 'react_h0_c3000_hs_record1'
    # flame, piv_folder, D, U_bulk, \
    # window_size_x_raw, window_size_y_raw, x_left_raw, y_top_raw, \
    # n_windows_x, n_windows_y, window_size_x, window_size_y, \
    # x, y = parameters(name)
            
    # Toggle plot
    toggle_plot = True
    
    # Set local flame speed method
    method_key = 0
    
    # Image number
    image_nr = 104
    
    # Set amount of vector considered
    n_nearest_coords = 1
    
    # Threshold angle between segments of consecutive timesteps
    threshold_angle = 180
    
    # Obtain local flame speeds based on recording parameters
    local_flame_speed_method = local_flame_speed_methods[method_key]
    final_segments = local_flame_speed_method(image_nr, n_nearest_coords, threshold_angle)
    
    # Read PIV data
    contour_nr, n_windows_x, n_windows_y, x, y, u, v, V_abs, divergence_u_davis = read_piv_data(image_nr)
    
    # Contour correction RAW --> (non-dimensionalized) WORLD [reference]
    contour_corrected = contour_correction(contour_nr)
    
    unburnt_region = determine_unburnt_region(contour_corrected, True)
    
    S, S_scheme, divergence_u = compute_strain_rate_tensor(u, v, unburnt_region)
    
    # fig, ax = plt.subplots()        
    # ax.set_title(r'$\nabla \cdot \vec u$')
    
    for index, final_segment in enumerate(final_segments):
        
        x_piv = final_segment[0][6]
        y_piv = final_segment[0][7]
        
        S_a = final_segment[8]
        
        rotation_matrix = final_segment[0][18]
        
        # Calculate the distance of each grid point from the target coordinate
        dist = np.sqrt((x - x_piv)**2 + (y - y_piv)**2)
        
        # Find the index of the grid point closest to the target coordinate
        yi_piv, xi_piv  = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
        
        # ax.plot(x[yi_piv, xi_piv], y[yi_piv, xi_piv], marker='o', color='r')
        
        if index == 0 : #len(final_segments) - 9:
            
            yi_piv_special, xi_piv_special = yi_piv, xi_piv
            S_ji = S[yi_piv, xi_piv]
            S_tn = rotation_matrix @ S_ji @ rotation_matrix.T
            
            # print((yi_piv,xi_piv), x[yi_piv, xi_piv], y[yi_piv, xi_piv]) 
            # print(S_ji)
            # print(S_tn)
            # print(divergence_u_davis[yi_piv, xi_piv])
            # print('-'*32)
    
    fig, ax = plt.subplots()
    divergence_u_davis_map = ax.pcolor(x, y, divergence_u_davis*unburnt_region.astype(float), cmap='viridis')
    cbar = ax.figure.colorbar(divergence_u_davis_map)
    divergence_u_davis_map.set_clim(-1000, 1500)
    ax.set_aspect('equal')
    
    fig, ax = plt.subplots()
    divergence_u_map = ax.pcolor(x, y, divergence_u, cmap='viridis')
    cbar = ax.figure.colorbar(divergence_u_map)
    divergence_u_map.set_clim(-1000, 1500)
    ax.set_aspect('equal')
    
    
    contour = flame.frames[frame_nr].contour_data.segmented_contours[contour_nr]
    contour_slopes = flame.frames[frame_nr].contour_data.slopes_of_segmented_contours[contour_nr]
    contour_slope_changes = flame.frames[frame_nr].contour_data.slope_changes_of_segmented_contours[contour_nr]
    
    #%%% CHECK SEGMENT NRS OF CONTOUR
    fig, ax = plot_contour(contour_nr)
    plot_segment_nrs(ax, contour_nr)
    title = ax.get_title()
    ax.set_title('Segments nrs \n' + title)
    fig.tight_layout()
    
    #%%% CHECK SLOPE OF CONTOUR
    fig, ax = plot_contour(contour_nr)
    plot_slopes(ax, contour_nr)
    title = ax.get_title()
    ax.set_title('Slopes \n' + title)
    fig.tight_layout()
    
    #%%% CHECK CHANGE OF SLOPE OF CONTOUR
    fig, ax = plot_contour(contour_nr)
    plot_slope_changes(ax, contour_nr)
    title = ax.get_title()
    ax.set_title('Change of slopes \n' + title)
    fig.tight_layout()
    
    # p1 = [contour_corrected[i,0,0], contour_corrected[i,0,1]]
    # p2 = [contour_corrected[i+1,0,0], contour_corrected[i+1,0,1]]
    # p3 = [contour_corrected[i+2,0,0], contour_corrected[i+2,0,1]]
    # xc, yc, radius, angle = radius_of_curvature(p1, p2, p3)
    
    # c = next(colormap_iter)
    # # if radius < 3:
    # ax.plot(p1[0], p1[1], c=c, marker='o')
    # ax.plot(p3[0], p3[1], c=c, marker='o')
    # ax.plot(xc, yc, c=c, marker='o')
    # draw_radius_of_curvature(ax, c, xc, yc, radius)
        
    #%%% CHECK CURVATURE OF CONTOUR
    # contour_curvature = curvature(contour_corrected)
    
    # for i, curv in enumerate(contour_curvature):
        
    #     x_coord_text = contour_corrected[i+1,0,0] 
    #     y_coord_text = contour_corrected[i+1,0,1]
        
    #     # R = 1/curv
    #     # print(R)
        
    #     ax.text(x_coord_text, y_coord_text, str(np.round(curv, 3)), color='k')

    # # # cx, cy, radius = define_circle(p1, p2, p3)
    # # contour_tortuosity = flame.frames[frame_nr].contour_data.tortuosity_of_segmented_contours[contour_nr]
    
    # # tort = -contour_tortuosity[0]
    # # radius = radius_of_curvature(p1, p2, p3)
    
    # draw_radius_of_curvature(ax, center_x2, center_y2, radius)
    # xc, yc, r, k = circle_from_two_segments(np.array(p1), np.array(p2), np.array(p3))
    # draw_radius_of_curvature(ax, xc, yc, r)
    
    #%%% CONTOUR DISTRIBUTION
    # contour_distribution = flame.frames[frame_nr].contour_data.contour_distribution
    
    # fig, ax = plt.subplots()
    # ax.imshow(contour_distribution, extent=extent_raw, cmap='viridis')
    
    # # quantity_plot = ax.pcolor(contour_distribution, cmap='viridis')
    
    # # quantity_plot.set_clim(0, contour_distribution.max())
    # # quantity_plot.set_clim(0, 20)
    
    # # Set aspect ratio of plot
    # ax.set_aspect('equal')
    
    

