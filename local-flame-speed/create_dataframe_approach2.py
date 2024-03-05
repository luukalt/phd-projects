# -*- coding: utf-8 -*-
'''
Created on Mon Mar 27 15:04:13 2023

@author: laaltenburg
'''

'''
final_segment: [selected_segment, x_intersect_ref, y_intersect_ref, x_intersect, y_intersect, flame_front_displacement, S_a, S_d]
selected_segment: 0
x_intersect_ref: 1
y_intersect_ref: 2
x_intersect: 3
y_intersect: 4
i_segment: 5
contour_correction: 6
flame_front_displacement: 7
S_a: 8
S_d: 9

selected_segment: [x_A, y_A, x_B, y_B, x_mid, y_mid, x_loc[side], y_loc[side], V_nx_select, V_ny_select, V_n[side], V_tx_select, V_ty_select, V_t[side], V_x_select, V_y_select, i_segment, contour_correction]
x_A: 0
y_A: 1
x_B: 2
y_B: 3
x_mid: 4
y_mid: 5
x_loc[side]: 6
y_loc[side]: 7
V_nx_select: 8
V_ny_select: 9
V_n[side]: 10
V_tx_select: 11
V_ty_select: 12
V_t[side]: 13
V_x_select: 14
V_y_select: 15
i_segment: 16
contour_correction: 17
rotation_matrix: 18

'''
#%% IMPORT PACKAGES
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy import optimize
from scipy.interpolate import UnivariateSpline, CubicSpline
from tqdm import tqdm
from local_flame_speed_approach2 import *
from premixed_flame_properties import *
import math

#%% CLOSE ALL FIGURES
plt.close('all')

#%% READ DATAFRAME
def read_csv(fname):
    
    spydata_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spydata')
    
    # Read a CSV file into a pandas dataframe
    df = pd.read_csv(os.path.join(spydata_dir, fname + '.csv'), index_col=0)
    
    return df

#%% CREATE DATAFRAME

def create_dataframe(method_key=0, last_image_nr=1000, n_nearest_coords=1, threshold_angle=180, spline_skip_length_mm=4):
    
    image_nrs = [*range(1, last_image_nr + 1, 1)]
        
    # Initialize an empty DataFrame with columns
    df = pd.DataFrame(columns=['image_nr', 
                               'contour_nr', 
                               'segment_index',
                               'x_mid',
                               'y_mid',
                               'x_intersect',
                               'y_intersect',
                               'slope', 
                               'slope_change',
                               'V_abs',
                               'V_t', 
                               'V_n', 
                               'S_a', 
                               'S_d', 
                               '|S_d|', 
                               'curvature',
                               'stretch_tangential',
                               'stretch_tangential_abs',
                               'stretch_tangential_norm',
                               'stretch_tangential_abs_norm',
                               'stretch_curvature',
                               'stretch_curvature_abs',
                               'stretch_curvature_norm',
                               'stretch_curvature_abs_norm',
                               ])
    
    for image_nr in tqdm(image_nrs):
        
        local_flame_speed_method = local_flame_speed_methods[method_key]
        final_segments = local_flame_speed_method(image_nr)
        
        # Read PIV data
        contour_nr, n_windows_x, n_windows_y, x, y, u, v, velocity_abs, divergence_u_davis = read_piv_data(image_nr)
        
        # Contour correction RAW --> (non-dimensionalized) WORLD [reference]
        contour_corrected = contour_correction(contour_nr)
        contour_x, contour_y =  contour_corrected[:,0,0], contour_corrected[:,0,1]
        
        # Compute strain rate tensor S and the divergence of velocity field u
        unburnt_region = determine_unburnt_region(contour_corrected, False)
        S, S_scheme, trace_field = compute_strain_rate_tensor(u, v, unburnt_region)
        
        # Obtain information regarding the flame front contour
        contour = flame.frames[frame_nr].contour_data.segmented_contours[contour_nr]
        contour_slopes = flame.frames[frame_nr].contour_data.slopes_of_segmented_contours[contour_nr]
        contour_slope_changes = flame.frames[frame_nr].contour_data.slope_changes_of_segmented_contours[contour_nr]
        
        # Flame front fitting
        
        # Choose whether to skip flame front coordinates
        skip_coords = int(spline_skip_length_mm/flame.segment_length_mm)
        
        # Construct spline coordinates
        contour_spline_coords = np.concatenate((contour_corrected[::skip_coords], [contour_corrected[-1]]))
        contour_spline_coords = np.squeeze(contour_spline_coords)
        
        # Fit cubic spline
        s_interp, x_interp, y_interp, x_spline, y_spline, arc_length = fit_parametric_cubic_spline(contour_spline_coords)
        
        # Calculate curvature of fitted cubic spline
        curvature = signed_curvature_cubic_spline(s_interp, x_spline, y_spline)
        
        for final_segment in final_segments:
            
            x_mid = final_segment[0][4]
            y_mid = final_segment[0][5]
            x_piv = final_segment[0][6]
            y_piv = final_segment[0][7]
            V_n = final_segment[0][10]
            V_t = final_segment[0][13]
            V_x = final_segment[0][14]
            V_y = final_segment[0][15]
            i_segment = final_segment[0][16]
            rotation_matrix = final_segment[0][18]
            
            x_intersect_t0 = final_segment[1]
            y_intersect_t0 = final_segment[2]
            S_a = final_segment[8]
            S_d = final_segment[9]
            
            V_dir = np.array([V_x, V_y])
            V_abs = np.linalg.norm(V_dir)
            
            S_d_abs = np.abs(S_d)
            
            # Determine slope
            slope = contour_slopes[i_segment]
            slope = np.abs(slope)
                
            # Determine slope change
            if i_segment == 0:
                
                slope_change = contour_slope_changes[i_segment]
                
            elif i_segment == len(contour) - 2:
                
                slope_change = contour_slope_changes[-1]
                
            else:
                
                slope_change = (contour_slope_changes[i_segment] + contour_slope_changes[i_segment-1])/2
                
            # slope_change = -slope_change
            
            # Flame stretch
            # Calculate the distance of each grid point from the target coordinate
            dist = np.sqrt((x - x_piv)**2 + (y - y_piv)**2)
            
            # Find the index of the grid point closest to the target coordinate
            yi_piv, xi_piv  = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
            
            S_ji = S[yi_piv, xi_piv]
            S_tn = rotation_matrix @ S_ji @ rotation_matrix.T
            
            (interp_index, nearest_x_interp_coord, nearest_y_interp_coord, nearest_curvature) = find_closest_point_and_curvature(s_interp, x_interp, y_interp, curvature, x_intersect_t0, y_intersect_t0)
            
            stretch_norm = flame.D_in*1e-3/flame.u_bulk_measured
            
            stretch_tangential = S_tn[0,0]
            stretch_tangential_abs = np.abs(stretch_tangential)
            
            stretch_curvature = S_d*(nearest_curvature*1e3)
            stretch_curvature_abs = np.abs(stretch_curvature)
            
            # If curvature > 0, then flame front is convex () towards reactants (which is a flame bulge), else if
            # curvature < 0, then flame front is concave () towards reactants (which is a flame cusp)
            # if nearest_curvature > 0:
                
            row_data = {'image_nr': image_nr,
                        'contour_nr': contour_nr, 
                        'segment_index': i_segment,
                        'x_mid': x_mid,
                        'y_mid': y_mid,
                        'x_intersect' : x_intersect_t0,
                        'y_intersect': y_intersect_t0,
                        'slope': slope,
                        'slope_change': slope_change,
                        'V_abs' : V_abs,
                        'V_t': V_t, 
                        'V_n': V_n, 
                        'S_a': S_a, 
                        'S_d': S_d, 
                        '|S_d|': S_d_abs, 
                        'curvature':nearest_curvature, 
                        'stretch_tangential': stretch_tangential,
                        'stretch_tangential_abs': stretch_tangential_abs,
                        'stretch_tangential_norm': stretch_tangential*stretch_norm,
                        'stretch_tangential_abs_norm': stretch_tangential_abs*stretch_norm,
                        'stretch_curvature': stretch_curvature,
                        'stretch_curvature_abs': stretch_curvature_abs,
                        'stretch_curvature_norm': stretch_curvature*stretch_norm,
                        'stretch_curvature_abs_norm': stretch_curvature_abs*stretch_norm,
                        
                        }
               
            # Append the dictionary as a new row to the DataFrame
            df = pd.concat([df, pd.DataFrame(row_data, index=[0])], ignore_index=True)
            
    # Write the DataFrame to a CSV file
    fname = f"conven_segm_len_{flame.segment_length_mm}mm_wsize_{flame.window_size}pxl_spline_skip_len_{skip_coords}mm_{last_image_nr}"
            
    spydata_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spydata', flame.name)
    
    if not os.path.isdir(spydata_dir):
        # os.mkdirs(spydata_dir)
        os.makedirs(spydata_dir, exist_ok=True)
            
    df.to_csv(os.path.join(spydata_dir, fname + '.csv'), index=True, index_label='index')
    
    return df
                
#%% FIT CONTINOUS CURVE TO FLAME SHAPE

def fit_parametric_cubic_spline(coords):
    coords = np.array(coords)
    s = np.arange(len(coords))
    s_interp = np.linspace(s.min(), s.max(), num=1000)
    
    x_spline = CubicSpline(s, coords[:, 0], bc_type='natural')
    y_spline = CubicSpline(s, coords[:, 1], bc_type='natural')

    x_interp = x_spline(s_interp)
    y_interp = y_spline(s_interp)
    
    # Calculate the arc length
    dx_interp = np.diff(x_interp)
    dy_interp = np.diff(y_interp)
    arc_length_interp = np.sqrt(dx_interp**2 + dy_interp**2)
    
    # Insert zero
    first_value = 0
    arc_length_interp = np.insert(arc_length_interp, 0, first_value)
    arc_length = np.cumsum(arc_length_interp)
    
    return s_interp, x_interp, y_interp, x_spline, y_spline, arc_length


def signed_curvature_cubic_spline(s_interp, x_spline, y_spline):

    x_spline_d1 = x_spline.derivative()
    y_spline_d1 = y_spline.derivative()

    x_spline_d2 = x_spline_d1.derivative()
    y_spline_d2 = y_spline_d1.derivative()
    
    x_d1 = x_spline_d1(s_interp)
    y_d1 = y_spline_d1(s_interp)

    x_d2 = x_spline_d2(s_interp)
    y_d2 = y_spline_d2(s_interp)
    
    signed_curvature = (x_d1 * y_d2 - x_d2 * y_d1) / (x_d1**2 + y_d1**2)**(3/2)
    
    # Flame front convex towards reactants is positive, hence the (-) minus sign in the return statement.
    # The reason for this is that the flame front starts right and ends left.
    return -signed_curvature


def find_closest_point_and_curvature(s_interp, x_interp, y_interp, curvature, x_segment, y_segment):
    
    # Calculate the squared distance for each point in the interpolation
    squared_distances = (x_interp - x_segment)**2 + (y_interp - y_segment)**2

    # Get the index of the closest point in the interpolation
    index = np.argmin(squared_distances)
    
    # # Get the t value for the closest point
    # s = s_interp[index]

    # Get the curvature value for the closest point
    # curvature = np.interp(s, s_interp, curvature)
    
    nearest_curvature = curvature[index]
    
    return (index, x_interp[index], y_interp[index], nearest_curvature)


def get_label(var_name):
    
    icon = variables_dict[var_name][0]
    unit = variables_dict[var_name][1]
    label = r'${}\ {}$'.format(icon, unit)
    return icon, unit, label


#%% MAIN

if __name__ == '__main__':
    
    # !!! SET THE CORRECT PARAMETERS IN PARAMETERS.PY !!!
    
    print(flame.name)
    
    spline_skip_length_mm_list = [2]
    
    for spline_skip_length_mm in spline_skip_length_mm_list:
        
        df = create_dataframe(method_key=0, last_image_nr=1000, n_nearest_coords=1, threshold_angle=180, spline_skip_length_mm=spline_skip_length_mm)
    
    # Image number
    image_nrs = [65]
    
    fig1, ax1 = plt.subplots()
    
    image_colors = colors = tab10.colors[:len(image_nrs)]
     
    label_dict = {}
    
    for image_index, image_nr in enumerate(image_nrs):
        
        #%%% READ FLAME AND VELOCITY INFO
        label = f'Image {image_nr}'
        
        print('image nr:', image_nr)
        method_key = 0
        spline_skip_length_mm = 2
        local_flame_speed_method = local_flame_speed_methods[method_key]
        final_segments = local_flame_speed_method(image_nr)
        
        # Read PIV data
        contour_nr, n_windows_x, n_windows_y, x, y, u, v, velocity_abs, divergence_u_davis = read_piv_data(image_nr)
        
        # Contour correction RAW --> (non-dimensionalized) WORLD [reference]
        contour_corrected = contour_correction(contour_nr)
        contour_x, contour_y =  contour_corrected[:,0,0], contour_corrected[:,0,1]
        
        # Compute strain rate tensor S and the divergence of velocity field u
        unburnt_region = determine_unburnt_region(contour_corrected, False)
        S, S_scheme, trace_field = compute_strain_rate_tensor(u, v, unburnt_region)
        
        # Obtain information regarding the flame front contour
        contour = flame.frames[frame_nr].contour_data.segmented_contours[contour_nr]
        contour_slopes = flame.frames[frame_nr].contour_data.slopes_of_segmented_contours[contour_nr]
        contour_slope_changes = flame.frames[frame_nr].contour_data.slope_changes_of_segmented_contours[contour_nr]
        
        # #%%% CHECK SEGMENT NRS OF CONTOUR
        # fig, ax = plot_contour(contour_nr)
        # plot_segment_nrs(ax, contour_nr)
        # title = ax.get_title()
        # ax.set_title('Segments nrs \n' + title)
        # fig.tight_layout()
        
        #%%% FIT SPLINE TO FLAME FRONT
        fig2, ax2 = plt.subplots()
        
        shape_colors = jet(np.linspace(0, 1, len(contour_slopes)))
           
        skip_coords = int(spline_skip_length_mm/flame.segment_length_mm)
        
        contour_spline_coords = np.concatenate((contour_corrected[::skip_coords], [contour_corrected[-1]]))
        contour_spline_coords = np.squeeze(contour_spline_coords)
        
        # Fit cubic spline
        s_interp, x_interp, y_interp, x_spline, y_spline, arc_length = fit_parametric_cubic_spline(contour_spline_coords)
        curvature = signed_curvature_cubic_spline(s_interp, x_spline, y_spline)
        
        # Plot segmented flame front
        ax2.plot(contour_x, contour_y, color='k', ls='--', lw=1)
        
        # Plot cubic spline
        ax2.plot(x_interp, y_interp, color='k', ls='-', lw=3)
        
        fig3, ax3 = plt.subplots()
        
        ax3.plot(arc_length, curvature, color='k')
        
        # list_i = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950]
        
        # for i, j in enumerate(list_i):
        
        #     ax2.plot(x_interp[j], y_interp[j], color=shape_colors[i], marker='o')
        #     ax3.plot(arc_length[j], curvature[j], color=shape_colors[i], marker='o')

        
        mask = curvature > 0
        
        # Find indices of True values in the mask
        true_indices = np.nonzero(mask)[0]
        
        # Find indices where the difference between consecutive indices is not 1
        diff_indices = np.where(np.diff(true_indices) != 1)[0] + 1
        
        # Split indices into sequences of consecutive True values
        sequence_indices = np.split(true_indices, diff_indices)
        
        # Filter out sequences with length 1
        sequence_indices = [seq for seq in sequence_indices if len(seq) > 1]
        
        # Create an iterator for the color palette
        color_cycle = iter(jet(np.linspace(0, 1, len(sequence_indices))))

        for i, seq in enumerate(sequence_indices):
            
            color = next(color_cycle)
            
            filtered_x = x_interp[seq]
            filtered_y = y_interp[seq]
            ax2.plot(filtered_x, filtered_y, color=color, ls='-', lw=3)
        
            filtered_arc_length = arc_length[seq]
            filtered_curvature = curvature[seq]
            ax3.plot(filtered_arc_length, filtered_curvature, color=color, ls='-', lw=3)
                
        for segment_index, final_segment in enumerate(final_segments):
            
            x_piv = final_segment[0][6]
            y_piv = final_segment[0][7]
            V_n = final_segment[0][10]
            V_t = final_segment[0][13]
            i_segment = final_segment[0][16]
            rotation_matrix = final_segment[0][18]
            
            x_intersect_t0 = final_segment[1]
            y_intersect_t0 = final_segment[2]
            S_a = final_segment[8]
            S_d = final_segment[9]
            
            # Direction of S_d
            if S_d > 0:
                
                S_a_direction = S_a
                
            elif S_d < 0:
                
                S_a_direction = -S_a
                
            # Flame stretch
            # Calculate the distance of each grid point from the target coordinate
            dist = np.sqrt((x - x_piv)**2 + (y - y_piv)**2)
            
            # Find the index of the grid point closest to the target coordinate
            yi_piv, xi_piv  = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
            
            S_ji = S[yi_piv, xi_piv]
            S_tn = rotation_matrix @ S_ji @ rotation_matrix.T
            
            segment_endpoint_coords = contour_corrected[i_segment:i_segment+2, 0, :2].T
            segment_x_mid, segment_y_mid = np.sum(segment_endpoint_coords, axis=1)/2
            
            # ax2.plot(*segment_endpoint_coords, color=shape_colors[segment_index], ls='-', lw=3)
            
            (interp_index, nearest_x_interp_coord, nearest_y_interp_coord, nearest_curvature) = find_closest_point_and_curvature(s_interp, x_interp, y_interp, curvature, x_intersect_t0, y_intersect_t0)
            
            if nearest_curvature > 0:
                
                ax2.text(segment_x_mid, segment_y_mid, str(np.round(S_d, 3)), color='k')
                
                ax2.plot(x_intersect_t0, y_intersect_t0, color=shape_colors[segment_index], marker='x')
                
                ax2.plot(nearest_x_interp_coord, nearest_y_interp_coord, color=shape_colors[segment_index], marker='o')
                
                ax3.plot(arc_length[interp_index], nearest_curvature, color=shape_colors[segment_index], marker='o')
            
                # print(S_ji, S_tn, S_d, curvature, S_tn[0,0], S_d*curvature*1e3)
                
                var_x, var_x_name = nearest_curvature, 'curvature'
                var_y, var_y_name = S_d, 'S_d'
                # var_y, var_y_name = S_a_direction, 'S_a_direction'
                # var_y, var_y_name = V_n, 'V_n'
                
                _,_, x_label = get_label(var_x_name)
                _,_, y_label = get_label(var_y_name)
                
                if label not in label_dict:
                    label_dict[label] = True
                    ax1.plot(var_x, var_y, ls='', marker='o', color=shape_colors[segment_index], label=label) #image_colors[image_index]
                    
                else:
                    ax1.plot(var_x, var_y, ls='', marker='o', color=shape_colors[segment_index])
                
                ax1.set_xlabel(x_label)
                ax1.set_ylabel(y_label)
                        
        
        # Figure 1 settings
        ax1.set_title(label)
        ax1.grid(True)
        ax1.legend(title='Image \#')
        ax1.axvline(x=0, color='k', linewidth=2) # plot origin
        ax1.axhline(y=0, color='k', linewidth=2) # plot origin
        
        # Figure 2 settings
        ax2.set_title(label)
        ax2.set_xlim(x_left_raw, x_right_raw)
        ax2.set_ylim(y_bottom_raw, y_top_raw)
        ax2.grid(True)
        ax2.set_aspect('equal', adjustable='box')
        
        # Figure 3 settings
        ax3.set_title(label + '\n' + r'$\kappa$ vs. arc length'+ f' with Cubic Spline fitting')
        ax3.set_xlabel(r"Arc length [mm]")
        ax3.set_ylabel("Curvature [mm$^{-1}$]")
        ax3.set_ylim(-2, 2)
        ax3.grid(True)
        
        # Tighten figure layouts
        fig1.tight_layout()
        fig2.tight_layout()
        fig3.tight_layout()
        
        print('-'*32)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    