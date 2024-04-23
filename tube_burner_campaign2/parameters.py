# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 00:10:22 2024

@author: luuka
"""

#%% IMPORT STANDARD PACKAGES
import os
import sys
import pickle 
import socket

#%% IMPORT USER DEFINED PACKAGES
from sys_paths import parent_directory
import rc_params_settings

#%% Define cases
react_names_ls =    [
                    # ('react_h0_c3000_ls_record1', 57),
                    # ('react_h0_s4000_ls_record1', 58),
                    # ('react_h100_c12000_ls_record1', 61),
                    ('react_h100_c12500_ls_record1', 61),
                    # ('react_h100_s16000_ls_record1', 62)
                    ]

react_names_hs =    [
                    # ('react_h0_f2700_hs_record1', 57),
                    # ('react_h0_c3000_hs_record1', 57),
                    # ('react_h0_s4000_hs_record1', 58),
                    # ('react_h100_c12500_hs_record1', 61),
                    # ('react_h100_s16000_hs_record1', 62)
                    ]

#%% Set case parameters
frame_nr = 0
segment_length_mm = 1 # units: mm
window_size = 31 # units: pixels
# piv_method = 'PIV_MP(3x16x16_75%ov_ImgCorr)'
piv_method = 'PIV_MP(3x16x16_0%ov_ImgCorr)'

# Interpolation method used to interpolate data to grid data
interpolation_method = 'linear'

#%% Define paths
# PC name 
pc_name = socket.gethostname()

# Set data directory based on PC name
if pc_name == 'A7' or 'DESKTOP-KA87T30':
    
    data_dir = os.path.join('U:', 'staff-umbrella', 'High hydrogen', 'laaltenburg', 'data', 'tube_burner_campaign2', 'selected_runs')

else:

    data_dir = os.path.join('U:', 'High hydrogen', 'laaltenburg', 'data', 'tube_burner_campaign2', 'selected_runs')

# Set spydata directory based on PC name
if react_names_ls:
    spydata_dir = os.path.join(parent_directory, 'spydata\\udf')
elif react_names_hs:
    spydata_dir = os.path.join(parent_directory, 'spydata')

react_names = react_names_ls + react_names_hs

#%% Load case 
for name, nonreact_run_nr in react_names:

    fname = f'{name}_segment_length_{segment_length_mm}mm_wsize_{window_size}pixels'

    with open(os.path.join(spydata_dir, fname + '.pkl'), 'rb') as f:
        flame = pickle.load(f)
    
