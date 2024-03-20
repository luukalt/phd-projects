# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 12:50:26 2023

@author: laaltenburg

Davis 10 has an export bug: Column header: ' [counts]' instead of 'Intensity [counts]'

"""

#%% IMPORT STANDARD PACKAGES
import os
import sys
import pickle
import pandas as pd
from matplotlib.path import Path
from multiprocessing import Pool, cpu_count
import numpy as np

#%% ADD SYS PATHS
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
flame_front_detection_directory = os.path.abspath(os.path.join(parent_folder, 'flame_front_detection'))
flame_simulations_directory = os.path.abspath(os.path.join(parent_folder, 'flame_simulations'))
plot_parameters_directory = os.path.abspath(os.path.join(parent_folder, 'plot_parameters'))

# Add to sys.path
sys.path.append(flame_simulations_directory)

#%% IMPORT USER DEFINED PACKAGES
from premixed_flame_properties import PremixedFlame
from functions import process_df, contour_correction

#%% FUNCTIONS

def safe_divide(a, b):
    if b == 0:
        return 0  # Return 0 (or any other value you prefer) when division by zero is attempted
    else:
        return a / b

def get_values_in_interrogation_window(flame, raw_grid_df, piv_grid_df, int_window_size):
    """
    For each coordinate in nearest_coords_df (derived from the small grid),
    find all coordinates in raw_grid_df that fall within a box of specified size centered on the coordinate.
    Retrieve the corresponding values from the specified column of raw_grid_df.

    :param raw_grid_df: DataFrame with coordinates and other data of the big grid.
    :param nearest_coords_df: DataFrame with nearest coordinates in the big grid.
    :param box_size: Size of the box centered on each coordinate (width and height).
    :param column: The column from which to retrieve values.
    :return: DataFrame with values from the specified column for coordinates within the box.
    """
    
    df_favre = piv_grid_df.copy()
    
    # Half size for easy calculations
    half_size = int_window_size / 2
    
    # Lists to hold column data
    raw_indices_in_int_window = []
    
    states_in_int_window = []
    sum_states_in_int_window = []
    mean_states_in_int_window = []
    
    counts_in_int_window = []
    sum_counts_in_int_window = []
    mean_counts_in_int_window = []
    
    counts_norm_in_int_window = []
    sum_counts_norm_in_int_window = []
    mean_counts_norm_in_int_window = []
    
    Wsum_u_r = []
    Wsum_u_x = []
    Wmean_u_r = []
    Wmean_u_x = []
    
    Wsum_u_r_counts = []
    Wsum_u_x_counts = []
    Wmean_u_r_counts = []
    Wmean_u_x_counts = []
    
    u_r_col = 'Velocity u [m/s]' #df_favre.columns[u_r_col_index]
    u_x_col = 'Velocity v [m/s]' #df_favre.columns[u_x_col_index]
    
    for i, (index, row) in enumerate(df_favre.iterrows()):
        center_x, center_y = row['x_shift_norm'], row['y_shift_norm']

        # Find coordinates within the box
        in_int_window = raw_grid_df[
            (raw_grid_df['x_shift_norm'] >= center_x - half_size) &
            (raw_grid_df['x_shift_norm'] <= center_x + half_size) &
            (raw_grid_df['y_shift_norm'] >= center_y - half_size) &
            (raw_grid_df['y_shift_norm'] <= center_y + half_size)
        ]
        
        # Append the values and indices
        raw_indices = in_int_window.index.tolist()
        raw_indices_in_int_window.append(raw_indices)
        
        # STATES
        states = in_int_window['state'].tolist()
        states_in_int_window.append(states)
        
        # Calculate and append sum of state
        sum_states = sum(states)
        sum_states_in_int_window.append(sum_states)
        
        # Calculate and append mean of state
        mean_states = sum_states / len(states) if states else 0
        mean_states_in_int_window.append(mean_states)
        
        # COUNTS
        counts = in_int_window[intensity_count_column_header].tolist()
        counts_in_int_window.append(counts)
        
        # Calculate and append sum
        sum_counts = sum(counts)
        sum_counts_in_int_window.append(sum_counts)
        
        # Calculate and append mean
        mean_counts = sum_counts / len(counts) if counts else 0
        mean_counts_in_int_window.append(mean_counts)
        
        # COUNTS NORM
        # min_counts = np.min(counts)
        # max_counts = np.max(counts)
        
        if len(counts) == 0:
            min_counts = 0
            max_counts = 0
        else:
            min_counts = np.min(counts)
            max_counts = np.max(counts)
            
        counts_norm = [safe_divide(count - min_counts, max_counts - min_counts) for count in counts]
        counts_norm_in_int_window.append(counts_norm)
        
        # Calculate and append sum
        sum_counts_norm = sum(counts_norm)
        sum_counts_norm_in_int_window.append(sum_counts_norm)
        
        # Calculate and append mean
        mean_counts_norm = sum_counts_norm / len(counts_norm) if counts_norm else 0
        mean_counts_norm_in_int_window.append(mean_counts_norm)
        
        # Calculate and append multiplied sum
        Wsum_u_r.append(sum_states * row[u_r_col])
        Wsum_u_x.append(sum_states * row[u_x_col])
        Wmean_u_r.append(mean_states * row[u_r_col])
        Wmean_u_x.append(mean_states * row[u_x_col])
        
        # THIS IS JUST FOR CHECKING
        Wsum_u_r_counts.append(sum_counts * row[u_r_col])
        Wsum_u_x_counts.append(sum_counts * row[u_x_col])
        Wmean_u_r_counts.append(mean_counts * row[u_r_col])
        Wmean_u_x_counts.append(mean_counts * row[u_x_col])
        
    # Add the new columns to nearest_coords_df
    df_favre['raw_indices_in_int_window'] = raw_indices_in_int_window
    
    df_favre['states_in_int_window_for_state'] = states_in_int_window
    df_favre['Wsum [states]'] = sum_states_in_int_window
    df_favre['Wmean [states]'] = mean_states_in_int_window
    
    df_favre['counts_in_int_window_for_counts'] = counts_in_int_window
    df_favre['Wsum [counts]'] = sum_counts_in_int_window
    df_favre['Wmean [counts]'] = mean_counts_in_int_window
    
    df_favre['counts_in_int_window_for_counts_norm'] = counts_norm_in_int_window
    df_favre['Wsum_norm [counts]'] = sum_counts_norm_in_int_window
    df_favre['Wmean_norm [counts]'] = mean_counts_norm_in_int_window
    
    df_favre['Wsum*u'] = Wsum_u_r
    df_favre['Wsum*v'] = Wsum_u_x
    df_favre['Wmean*u'] = Wmean_u_r
    df_favre['Wmean*v'] = Wmean_u_x
    
    df_favre['Wsum*u [counts]'] = Wsum_u_r_counts
    df_favre['Wsum*v [counts]'] = Wsum_u_x_counts
    df_favre['Wmean*u [counts]'] = Wmean_u_r_counts
    df_favre['Wmean*v [counts]'] = Wmean_u_x_counts
    
    df_favre['rho [kg/m^3]'] = flame.properties.rho_b + (flame.properties.rho_u - flame.properties.rho_b) * df_favre['Wmean [states]']
    df_favre['rho*u'] = df_favre['rho [kg/m^3]'] * df_favre[u_r_col]
    df_favre['rho*v'] = df_favre['rho [kg/m^3]'] * df_favre[u_x_col]
    
    return df_favre

def process_file(file_number, flame, D_in, offset_to_wall_center, offset, interrogation_window_size_norm, columns_to_average, bottom_limit, top_limit, left_limit, right_limit, index_name, column_name, data_dir, session_nr, recording, piv_method):
    """
    Process a single file. This function is executed by each worker.
    """
    # PIV file processing 
    piv_file = os.path.join(data_dir, f'session_{session_nr:03d}', recording, piv_method, 'Export', f'B{file_number:04d}.csv')
    df_piv = pd.read_csv(piv_file)
    df_piv = process_df(df_piv, D_in, offset_to_wall_center, offset)
    df_piv_cropped = df_piv[(df_piv[index_name] > bottom_limit) & (df_piv[index_name] < top_limit) & (df_piv[column_name] > left_limit) & (df_piv[column_name] < right_limit)]
    
    # Raw Mie scattering file processing
    raw_file = os.path.join(data_dir, f'session_{flame.session_nr:03d}', recording, 'Correction', 'Resize', 'Frame0', 'Export', f'B{file_number:04d}.csv')
    # raw_file = os.path.join(data_dir, f'session_{flame.session_nr:03d}', recording, 'Correction', 'Resize', 'Frame0', 'Norm by Avg Stddev Intensity Normalization', 'AboveBelow', 'Export', f'B{file_number:04d}.csv')
    df_raw = pd.read_csv(raw_file)
    df_raw = process_df(df_raw, D_in, offset_to_wall_center, offset)
    
    # Read intensity of the raw Mie scattering image
    pivot_intensity = pd.pivot_table(df_raw, values=intensity_count_column_header, index='y_shift_norm', columns='x_shift_norm')
    r_raw_array = pivot_intensity.columns
    x_raw_array = pivot_intensity.index
    r_raw, x_raw = np.meshgrid(r_raw_array, x_raw_array)
    # n_windows_r_raw, n_windows_x_raw = len(r_raw_array), len(x_raw_array)
    window_size_r_raw, window_size_x_raw = np.mean(np.diff(r_raw_array)), -np.mean(np.diff(x_raw_array))
    
    r_left_raw = r_raw_array[0]
    r_right_raw = r_raw_array[-1]
    x_bottom_raw = x_raw_array[0]
    x_top_raw = x_raw_array[-1]
    
    contour_nr = file_number - 1
    contour_corrected = contour_correction(flame, contour_nr, r_left_raw, r_right_raw, x_bottom_raw, x_top_raw, window_size_r_raw, window_size_x_raw, frame_nr=0)
    contour_x, contour_y =  contour_corrected[:,0,0], contour_corrected[:,0,1]
    
    # Close the contour by appending the first point at the end
    contour_closed = np.vstack([np.column_stack((contour_x, contour_y)), 
                                [contour_x[-1], contour_y[-1] - .2], 
                                [contour_x[0], contour_y[-1] - .2], 
                                [contour_x[0], contour_y[0]]
                                ])
    
    # Create a Path object
    path = Path(contour_closed)
    
    states = []
    
    for i, (index, row) in enumerate(df_raw.iterrows()):
        r_coord, x_coord = row['x_shift_norm'], row['y_shift_norm']
        
        point = np.array([(r_coord, x_coord)])  # Replace with your coordinates
        is_inside = path.contains_points(point)
        
        if is_inside:
            states.append(1)
        else:
            states.append(0)
    
    # Add the 'state' column to df_raw
    df_raw['state'] = states

    # Your processing function
    df_favre = get_values_in_interrogation_window(flame, df_raw, df_piv_cropped, interrogation_window_size_norm)

    # Save the DataFrame to CSV
    favre_file = os.path.join('spydata', flame.name, 'favre_csv', f'B{file_number:04d}.csv')
    # favre_file = os.path.join('spydata', flame.name, 'favre_csv', 'Below150', 'udf_3x3_max_cross', f'B{file_number:04d}.csv')
    os.makedirs(os.path.dirname(favre_file), exist_ok=True)
    df_favre.to_csv(favre_file, index=True, index_label='index')

    return file_number

def update_file(file_number, flame):
    
    favre_file = os.path.join('spydata', flame.name, 'favre_csv', f'B{file_number:04d}.csv')
    df_favre = pd.read_csv(favre_file)
    
    favre_avg_file = os.path.join('spydata', flame.name, 'AvgFavre.csv')
    df_favre_avg = pd.read_csv(favre_avg_file)
    
    # Check if the column exists in both DataFrames
    u_r_col = 'Velocity u [m/s]'
    u_x_col = 'Velocity v [m/s]'
    Wmean_col = 'Wmean [states]'
    rho_col = 'rho [kg/m^3]'

    u_r_fluc = df_favre[u_r_col] - df_favre_avg[u_r_col]
    u_x_fluc = df_favre[u_x_col] - df_favre_avg[u_x_col]
    
    # u_r_tilde = df_favre_avg['Wmean*u'].div(df_favre_avg['Wmean']).fillna(0)
    # u_x_tilde = df_favre_avg['Wmean*v'].div(df_favre_avg['Wmean']).fillna(0)
    
    u_r_tilde = df_favre_avg['rho*u'].div(df_favre_avg['rho [kg/m^3]']).fillna(0)
    u_x_tilde = df_favre_avg['rho*v'].div(df_favre_avg['rho [kg/m^3]']).fillna(0)
    
    Wmean = df_favre[Wmean_col]
    rho = df_favre[rho_col]
    
    u_r_fluc_favre = df_favre[u_r_col] - u_r_tilde
    u_x_fluc_favre = df_favre[u_x_col] - u_x_tilde
    
    # Add the result as a new column
    df_favre_avg['u_favre [m/s]'] = u_r_tilde
    df_favre_avg['v_favre [m/s]'] = u_x_tilde
    
    df_favre['u_fluc [m/s]'] = u_r_fluc
    df_favre['v_fluc [m/s]'] = u_x_fluc
    
    df_favre['u_fluc*u_fluc'] = u_r_fluc*u_r_fluc
    df_favre['v_fluc*v_fluc'] = u_x_fluc*u_x_fluc
    df_favre['u_fluc*v_fluc'] = u_r_fluc*u_x_fluc
    
    df_favre['u_fluc_favre [m/s]'] = u_r_fluc_favre
    df_favre['v_fluc_favre [m/s]'] = u_x_fluc_favre
    
    df_favre['u_fluc_favre*u_fluc_favre'] = u_r_fluc_favre*u_r_fluc_favre
    df_favre['v_fluc_favre*v_fluc_favre'] = u_x_fluc_favre*u_x_fluc_favre
    df_favre['u_fluc_favre*v_fluc_favre'] = u_r_fluc_favre*u_x_fluc_favre
    
    df_favre['Wmean*u_fluc_favre'] = Wmean*u_r_fluc_favre
    df_favre['Wmean*v_fluc_favre'] = Wmean*u_x_fluc_favre
    
    df_favre['Wmean*u_fluc_favre*u_fluc_favre'] = Wmean*u_r_fluc_favre*u_r_fluc_favre
    df_favre['Wmean*v_fluc_favre*v_fluc_favre'] = Wmean*u_x_fluc_favre*u_x_fluc_favre
    df_favre['Wmean*u_fluc_favre*v_fluc_favre'] = Wmean*u_r_fluc_favre*u_x_fluc_favre
    
    df_favre['rho*u_fluc_favre'] = rho*u_r_fluc_favre
    df_favre['rho*v_fluc_favre'] = rho*u_x_fluc_favre
    
    df_favre['rho*u_fluc_favre*u_fluc_favre'] = rho*u_r_fluc_favre*u_r_fluc_favre
    df_favre['rho*v_fluc_favre*v_fluc_favre'] = rho*u_x_fluc_favre*u_x_fluc_favre
    df_favre['rho*u_fluc_favre*v_fluc_favre'] = rho*u_r_fluc_favre*u_x_fluc_favre
    

    # Save the updated DataFrames back to files
    df_favre.to_csv(favre_file, index=False)
    if file_number == 1:
        df_favre_avg.to_csv(favre_avg_file, index=False)
        
    return file_number

def process_raw_file(image_nr, flame, offset_to_wall_center, offset, index_name, column_name, data_dir, session_nr, recording):
    
    raw_file = os.path.join(data_dir, f'session_{flame.session_nr:03d}', recording, 'Correction', 'Resize', 'Frame0', 'Export', f'B{image_nr:04d}.csv')
    
    df_raw = pd.read_csv(raw_file)
    df_raw = process_df(df_raw, flame.D_in, offset_to_wall_center, offset)
    
    bottom_limit = -100
    top_limit = 100
    left_limit = -100
    right_limit = 100
    
    df_raw = df_raw[(df_raw[index_name] > bottom_limit) & (df_raw[index_name] < top_limit) & (df_raw[column_name] > left_limit) & (df_raw[column_name] < right_limit)]

    # Read intensity
    pivot_intensity = pd.pivot_table(df_raw, values=intensity_count_column_header, index='y_shift_norm', columns='x_shift_norm')
    r_raw_array = pivot_intensity.columns
    x_raw_array = pivot_intensity.index
    r_raw, x_raw = np.meshgrid(r_raw_array, x_raw_array)
    # n_windows_r_raw, n_windows_x_raw = len(r_raw_array), len(x_raw_array)
    window_size_r_raw, window_size_x_raw = np.mean(np.diff(r_raw_array)), -np.mean(np.diff(x_raw_array))
    
    r_left_raw = r_raw_array[0]
    r_right_raw = r_raw_array[-1]
    x_bottom_raw = x_raw_array[0]
    x_top_raw = x_raw_array[-1]
    
    contour_nr = image_nr - 1
    contour_corrected = contour_correction(flame, contour_nr, r_left_raw, r_right_raw, x_bottom_raw, x_top_raw, window_size_r_raw, window_size_x_raw, frame_nr=0)
    contour_x, contour_y =  contour_corrected[:,0,0], contour_corrected[:,0,1]
    
    # Close the contour by appending the first point at the end
    contour_closed = np.vstack([np.column_stack((contour_x, contour_y)), [contour_x[0], contour_y[0]]])

    # Create a Path object
    path = Path(contour_closed)
    
    states = []
    
    for i, (index, row) in enumerate(df_raw.iterrows()):
        r_coord, x_coord = row['x_shift_norm'], row['y_shift_norm']
        
        point = np.array([(r_coord, x_coord)])  # Replace with your coordinates
        is_inside = path.contains_points(point)
        
        if is_inside:
            states.append(1)
        else:
            states.append(0)
    
    # Add the 'state' column to df_raw
    df_raw['state'] = states

    return df_raw

#%% MAIN
if __name__ == "__main__":
    
    data_dir = 'U:\\staff-umbrella\\High hydrogen\\laaltenburg\\data\\tube_burner_campaign2\\selected_runs\\'
    
    intensity_count_column_header = ' [counts]'
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
        spydata_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spydata\\udf')
    elif react_names_hs:
        spydata_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spydata')
        
        
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
        # run_nr = flame.run_nr
        # Re_D_set = flame.Re_D
        # u_bulk_set = flame.u_bulk_measured
        # u_bulk_measured = flame.u_bulk_measured
        
        mixture = PremixedFlame(flame.phi, flame.H2_percentage, flame.T_lab, flame.p_lab)
        mixture.solve_equations()
    
        flame.properties.rho_b = mixture.rho_b
        
    #%%% Calibration details
    D_in = flame.D_in # Inner diameter of the quartz tube, units: mm
    offset = 1  # Distance from calibrated y=0 to tube burner rim
    
    wall_center_to_origin = 2
    wall_thickness = 1.5
    offset_to_wall_center = wall_center_to_origin - wall_thickness/2
    
    #%%% Read the averaged PIV file and add coordinate system translation
    # PIV directory    
    Avg_Stdev_file = os.path.join(data_dir, f'session_{session_nr:03d}', recording, piv_method, 'Avg_Stdev', 'Export', 'B0001.csv')
    
    # Read the averaged PIV file and add coordinate system translation
    df_piv = pd.read_csv(Avg_Stdev_file)
    df_piv = process_df(df_piv, D_in, offset_to_wall_center, offset)
    
    # Get the column headers of the PIV file
    headers = df_piv.columns

    # Non-dimensional limits in r- (left, right) and x-direction (bottom, top)
    # bottom_limit = -.5
    # top_limit = 2.25
    # left_limit = -0.575
    # right_limit = 0.575
    
    bottom_limit = -100
    top_limit = 100
    left_limit = -100
    right_limit = 100
    index_name = 'y_shift_norm'
    column_name = 'x_shift_norm'
    
    # Cropped PIV dataframe based on Non-dimensional limits in r- (left, right) and x-direction (bottom, top)
    df_reynolds_avg = df_piv[(df_piv[index_name] > bottom_limit) & (df_piv[index_name] < top_limit) & (df_piv[column_name] > left_limit) & (df_piv[column_name] < right_limit)]
    
    # Set the size of the interrogation window and calucate its non-dimensionless value
    interrogation_window_size = 15 # pixels
    interrogation_window_size_norm = (interrogation_window_size/flame.scale) / D_in # Define the size of the box
    
    # Read the first instantenous PIV file to determine the DataFrame structure
    first_piv_file_file = os.path.join(data_dir, f'session_{session_nr:03d}', recording, piv_method, 'Export', 'B0001.csv')
    df_piv_first = pd.read_csv(first_piv_file_file)
    df_piv_first = process_df(df_piv_first, D_in, offset_to_wall_center, offset)
    
    # Cropped first instantenous PIV file dataframe based on non-dimensional limits in r- (left, right) and x-direction (bottom, top)
    df_piv_first_cropped = df_piv_first[(df_piv_first[index_name] > bottom_limit) & (df_piv_first[index_name] < top_limit) & (df_piv_first[column_name] > left_limit) & (df_piv_first[column_name] < right_limit)]
    
    #%%% Set amount of images to average over
    image_nrs = flame.n_images
    
    #%%% Step 1: Obtain time averages of mean part
    # Specify the columns for which you want to calculate the average
    columns_to_average = ['x_shift [m]', 'y_shift [m]', 'x_shift [mm]', 'y_shift [mm]', 'x_shift_norm', 'y_shift_norm', 
                          'Velocity u [m/s]', 'Velocity v [m/s]', 
                          'Wsum [states]', 'Wmean [states]',
                          'Wsum [counts]', 'Wmean [counts]',
                          'Wsum_norm [counts]', 'Wmean_norm [counts]',
                          'Wsum*u', 'Wsum*v', 'Wmean*u', 'Wmean*v',
                          'Wsum*u [counts]', 'Wsum*v [counts]', 'Wmean*u [counts]', 'Wmean*v [counts]',
                          'rho [kg/m^3]', 'rho*u', 'rho*v'
                          ]

    # Initialize df_lists with the same indices and columns as the cropped first instantenous PIV file
    df_favre_lists = pd.DataFrame(index=df_piv_first_cropped.index)
    for col in columns_to_average:  # df_piv_first_cropped.columns:
        df_favre_lists[col] = [[] for _ in range(len(df_piv_first_cropped))]

    # Bundle all arguments into a tuple
    args = (flame, D_in, offset_to_wall_center, offset, interrogation_window_size_norm, columns_to_average, bottom_limit, top_limit, left_limit, right_limit, index_name, column_name, data_dir, session_nr, recording, piv_method)
    
    # Set number of cpus for parallel computing
    n_cpus = cpu_count()
    
    with Pool(n_cpus - 1) as pool:
        # Use starmap to unpack arguments
        for i in pool.starmap(process_file, [(num, *args) for num in range(1, image_nrs + 1)]):
            # Load the processed data
            favre_file = os.path.join('spydata', flame.name, 'favre_csv', f'B{i:04d}.csv')
            df_favre = pd.read_csv(favre_file, index_col='index')
            
            # Append values from df_favre to the corresponding lists in df_lists
            for col in df_favre_lists.columns:
                for idx in df_favre_lists.index:
                    if col in df_favre.columns and idx in df_favre.index:
                        df_favre_lists.at[idx, col].append(df_favre.at[idx, col])
                    else:
                        df_favre_lists.at[idx, col].append(None)
         
    print('Processing files completed')

    df_favre_avg = df_favre_lists.applymap(lambda lst: sum(lst) / len(lst) if lst else None)
    
    #%%%% Save the Favre average of Step 1 to a CSV file
    output_file = os.path.join('spydata', flame.name, 'AvgFavre.csv')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_favre_avg.to_csv(output_file, index=True, index_label='index')
    
    #%%% Step 2: Obtain time averages of fluctuating part
    columns_to_average += ['u_fluc [m/s]', 'v_fluc [m/s]', 
                          'u_fluc*u_fluc', 'v_fluc*v_fluc', 'u_fluc*v_fluc',
                          'u_fluc_favre [m/s]', 'v_fluc_favre [m/s]',
                          'u_fluc_favre*u_fluc_favre', 'v_fluc_favre*v_fluc_favre', 'u_fluc_favre*v_fluc_favre',
                          'Wmean*u_fluc_favre', 'Wmean*v_fluc_favre',
                          'Wmean*u_fluc_favre*u_fluc_favre', 'Wmean*v_fluc_favre*v_fluc_favre', 'Wmean*u_fluc_favre*v_fluc_favre',
                          'rho*u_fluc_favre', 'rho*v_fluc_favre',
                          'rho*u_fluc_favre*u_fluc_favre', 'rho*v_fluc_favre*v_fluc_favre', 'rho*u_fluc_favre*v_fluc_favre'
                          ]
    
    # Initialize df_lists with the same indices and columns as df_first
    df_favre_lists2 = pd.DataFrame(index=df_piv_first_cropped.index)
    
    for col in columns_to_average:  # df_piv_first_cropped.columns:
        df_favre_lists2[col] = [[] for _ in range(len(df_piv_first_cropped))]
        
    with Pool(n_cpus - 1) as pool:
        # Use starmap to unpack arguments
        for i in pool.starmap(update_file, [(num, flame) for num in range(1, image_nrs + 1)]):
            
            # Load the processed data
            favre_file = os.path.join('spydata', flame.name, 'favre_csv', f'B{i:04d}.csv')
            df_favre = pd.read_csv(favre_file, index_col='index')
            
            # Append values from df_favre to the corresponding lists in df_lists
            for col in df_favre_lists2.columns:
                for idx in df_favre_lists2.index:
                    if col in df_favre.columns and idx in df_favre.index:
                        df_favre_lists2.at[idx, col].append(df_favre.at[idx, col])
                    else:
                        df_favre_lists2.at[idx, col].append(None)
                        
    print('Updating files completed')
    
    df_favre_avg2 = df_favre_lists2.applymap(lambda lst: sum(lst) / len(lst) if lst else None)
    
    # df_favre_avg_file_path = os.path.join('spydata', flame.name, 'favre_csv', f'B{i:04d}.csv')
    # df_favre_avg = pd.read_csv(df_favre_avg_file_path, index_col='index')
    
    favre_avg_file = os.path.join('spydata', flame.name, 'AvgFavre.csv')
    df_favre_avg = pd.read_csv(favre_avg_file, index_col='index')
    
    df_favre_avg2['u_favre [m/s]'] = df_favre_avg['u_favre [m/s]']
    df_favre_avg2['v_favre [m/s]'] = df_favre_avg['v_favre [m/s]']
    
    df_favre_avg2['R_uu [m^2/s^2]'] = df_favre_avg2['rho*u_fluc_favre*u_fluc_favre'].div(df_favre_avg2['rho [kg/m^3]']).fillna(0)
    df_favre_avg2['R_uv [m^2/s^2]'] = df_favre_avg2['rho*u_fluc_favre*v_fluc_favre'].div(df_favre_avg2['rho [kg/m^3]']).fillna(0)
    df_favre_avg2['R_vv [m^2/s^2]'] = df_favre_avg2['rho*v_fluc_favre*v_fluc_favre'].div(df_favre_avg2['rho [kg/m^3]']).fillna(0)
    
    df_favre_avg2['0.5*(R_uu + R_vv) [m^2/s^2]'] = 0.5 * (df_favre_avg2['R_uu [m^2/s^2]'] + df_favre_avg2['R_vv [m^2/s^2]'])
    df_favre_avg2['0.75*(R_uu + R_vv) [m^2/s^2]'] = 0.75 * (df_favre_avg2['R_uu [m^2/s^2]'] + df_favre_avg2['R_vv [m^2/s^2]'])
    
    #%%%% Save the Favre average of Step 2 to a CSV file
    output_file = os.path.join('spydata', flame.name, 'AvgFavreFinal.csv')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_favre_avg2.to_csv(output_file, index=True, index_label='index')





