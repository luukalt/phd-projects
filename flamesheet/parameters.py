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
import numpy as np

#%% IMPORT USER DEFINED PACKAGES
# from sys_paths import parent_directory
# import rc_params_settings

#%% Define cases
main_dir = os.path.join('Y:', 'laaltenburg', 'flamesheet_2d_campaign1')
project_name = 'flamesheet_2d_day14'

#%%%% Select recording
# H2% = 0, phi = 0, Re_H = 7000, image_rate = 0.2 kHz
record_name = "Recording_Date=221118_Time=144758"
pre_record_name = "Recording_Date=221118_Time=112139"
piv_result = "PIV_MP(3x32x32_50%ov_ImgCorr)"

# H2% = 100, phi = 0.3, Re_H = 7000, image_rate = 0.2 kHz 
# record_name = 'Recording_Date=221118_Time=115220_01'
# pre_record_name = 'Recording_Date=221118_Time=112139'
# piv_result = 'PIV_MP(3x32x32_50%ov_ImgCorr)'

# H2% = 100, phi = 0.3, Re_H = 7000, image_rate = 4.5 kHz
# record_name = "Recording_Date=221118_Time=125724_01"
# pre_record_name = "Recording_Date=221118_Time=112139"
# piv_result = "PIV_MP(3x32x32_50%ov_ImgCorr)"

#%%%%  Define data directories
project_dir = os.path.join(main_dir, project_name)

calibration_csv_dir =  os.path.join(project_dir, 'Properties', 'Calibration', 'DewarpedImages1', 'Export', 'B0001.csv')
calibration_tif_dir = os.path.join(project_dir, 'Properties', 'Calibration', 'DewarpedImages1', 'Export_01', 'B0001.tif')

pre_record_raw_dir = os.path.join(project_dir, pre_record_name, 'Reorganize frames', 'Export', 'B0001.tif')
pre_record_correction_dir = os.path.join(project_dir, pre_record_name, 'Correction', 'Reorganize frames', 'Export', 'B0001.tif')

piv_avgV_dir = os.path.join(project_dir, record_name, piv_result, 'Avg_Stdev', 'Export')
piv_transV_dir = os.path.join(project_dir, record_name, piv_result, 'Export')
piv_Rstress_dir =  os.path.join(project_dir, record_name, piv_result, 'AvgVx_AvgVy_AvgV_StdevVx_StdevVy_StdevV_Rxy_Rxx_Ryy_AKE_TKE_TSS', 'Export')


 