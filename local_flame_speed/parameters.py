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

#%% FIGURE SETTINGS

# def set_mpl_params():
    
#     # Use Latex font in plots
#     plt.rcParams.update({
#         'text.usetex': False,
#         # 'font.family': 'serif',
#         # 'font.serif': ['Computer Modern Roman'],
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

if pc_name == 'DESKTOP-B05JCBI':
    
    main_dir = 'W:\\staff-umbrella\\High hydrogen\\laaltenburg\\data\\tube_burner_campaign2\\selected_runs\\'

else:

    main_dir = 'U:\\High hydrogen\\laaltenburg\\data\\tube_burner_campaign2\\selected_runs\\'

# =============================================================================
# name = 'react_h0_c3000_hs_record1'
# frame_nr = 0
# segment_length_mm = 1 # units: mm
# window_size = 31 # units: pixels
# piv_folder = 'PIV_MP(3x16x16_0%ov_ImgCorr)'

# =============================================================================

# =============================================================================
# name = 'react_h0_s4000_hs_record1'
# frame_nr = 0
# segment_length_mm = 1 # units: mm
# window_size = 31 # units: pixels
# piv_folder = 'PIV_MP(3x16x16_0%ov_ImgCorr)'

# =============================================================================

# =============================================================================
# name = 'react_h100_c12500_hs_record1'
# frame_nr = 0
# segment_length_mm = 1 # units: mm
# window_size = 31 # units: pixels
# piv_folder = 'PIV_MP(3x16x16_0%ov_ImgCorr)'

# =============================================================================

# =============================================================================
name = 'react_h100_s16000_hs_record1'
frame_nr = 0
segment_length_mm = 1 # units: mm
window_size = 31 # units: pixels
piv_folder = 'PIV_MP(3x16x16_0%ov_ImgCorr)'

# =============================================================================

#%% READ FLAME INFO [DO NOT TOUCH]

spydata_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spydata')
# spydata_dir = os.path.join(main_dir, 'spydata')

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
raw_dir = os.path.join(main_dir,  f'session_{flame.session_nr:03d}', flame.record_name, 'Correction', 'Resize', 'Frame0', 'Export')
raw_file = os.path.join(raw_dir, 'B0001.csv')

df_raw = pd.read_csv(raw_file)

headers_raw = df_raw.columns

# Read intensity
pivot_intensity = pd.pivot_table(df_raw, values=headers_raw[2], index=headers_raw[1], columns=headers_raw[0])

x_raw_array = pivot_intensity.columns
y_raw_array = pivot_intensity.index
n_windows_x_raw, n_windows_y_raw = len(x_raw_array), len(y_raw_array)
window_size_x_raw, window_size_y_raw = np.mean(np.diff(x_raw_array)), -np.mean(np.diff(y_raw_array))

x_left_raw = x_raw_array[0]
x_right_raw = x_raw_array[-1]
y_bottom_raw = y_raw_array[0]
y_top_raw = y_raw_array[-1]

# extent_raw =  np.array([
#                         x_left_raw - window_size_x_raw / 2,
#                         x_left_raw + (n_windows_x_raw - 0.5) * window_size_x_raw,
#                         y_top_raw + (n_windows_y_raw - 0.5) * window_size_y_raw,
#                         y_top_raw - window_size_y_raw / 2
#                         ])/D

#%% READ PIV IMAGE DIMENSIONS [DO NOT TOUCH]

piv_dir = os.path.join(main_dir,  f'session_{flame.session_nr:03d}', flame.record_name, piv_folder, 'Export')
piv_file = os.path.join(piv_dir, 'B0001.csv')

df_piv = pd.read_csv(piv_file)

headers_piv = df_piv.columns

# Read absolute velocity
pivot_velocity_abs = pd.pivot_table(df_piv, values=headers_piv[4], index=headers_piv[1], columns=headers_piv[0])

x_array, y_array = pivot_velocity_abs.columns, pivot_velocity_abs.index
x, y = np.meshgrid(x_array, y_array)
n_windows_x, n_windows_y = len(x_array), len(y_array)
window_size_x, window_size_y = np.mean(np.diff(x_array)), np.mean(np.diff(y_array))
window_size_x_abs, window_size_y_abs = np.abs(window_size_x), np.abs(window_size_y)

extent_piv =  np.array([
                        x.min() - window_size_x_abs / 2,
                        x.max() + window_size_x_abs / 2,
                        y.min() - window_size_y_abs / 2,
                        y.max() + window_size_y_abs / 2
                        ])

    # return flame, piv_folder, D, U_bulk, \
    #         window_size_x_raw, window_size_y_raw, x_left_raw, y_top_raw, \
    #         n_windows_x, n_windows_y, window_size_x, window_size_y, \
    #         x, y


            
            




