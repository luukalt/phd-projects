# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 00:10:22 2024

@author: luuka
"""

#%% IMPORT STANDARD PACKAGES
import os

#%% IMPORT USER DEFINED PACKAGES
# from sys_paths import parent_directory
# import rc_params_settings

#%% Define cases
main_dir = os.path.join('Y:', 'laaltenburg', 'flamesheet_2d_campaign1')

#%%%% Select recording

case = 2

if case == 0:
    # test99, H2% = 100, phi = 0.35, Re_H = 5000, image_rate = 0.2 kHz
    day_nr = '23-2'
    project_name = "flamesheet_2d_day" + day_nr
    record_name = "Recording_Date=230215_Time=153433_01"
    pre_record_name = "Recording_Date=230215_Time=153306"
    u_bulk_measured = 8.57
elif case == 1:
    # test101, H2% = 100, phi = 0.35, Re_H = 7000, image_rate = 0.2 kHz
    day_nr = '23-2'
    project_name = "flamesheet_2d_day" + day_nr
    record_name = "Recording_Date=230215_Time=160055_01"
    pre_record_name = "Recording_Date=230215_Time=153306"
    u_bulk_measured = 11.88
elif case == 2:    
    # test103, H2% = 0, phi = 0, Re_H = 7000, image_rate = 0.2 kHz
    day_nr = '24-1'
    project_name = "flamesheet_2d_day" + day_nr
    record_name = "Recording_Date=230216_Time=103453_01"
    pre_record_name = "Recording_Date=230216_Time=103309"
    u_bulk_measured = 10.33
elif case == 3:    
    # test105, H2% = 0, phi = 0, Re_H = 5000, image_rate = 0.2 kHz
    day_nr = '24-1'
    record_name = "Recording_Date=230216_Time=105230_01"
    pre_record_name = "Recording_Date=230216_Time=103309"
    u_bulk_measured = 7.44

# H2% = 100, phi = 0.3, Re_H = 7000, image_rate = 0.2 kHz 
# record_name = 'Recording_Date=221118_Time=115220_01'
# pre_record_name = 'Recording_Date=221118_Time=112139'
# piv_result = 'PIV_MP(3x32x32_50%ov_ImgCorr)'

# H2% = 100, phi = 0.3, Re_H = 7000, image_rate = 4.5 kHz
# record_name = "Recording_Date=221118_Time=125724_01"
# pre_record_name = "Recording_Date=221118_Time=112139"
# piv_result = "PIV_MP(3x32x32_50%ov_ImgCorr)"

#%%%% Geometric Mask
# Without mask
# piv_result = "PIV_MP(3x32x32_50%ov_ImgCorr)" 

# With mask
piv_result = "PIV_MP(3x32x32_50%ov_ImgCorr)_01"

# Interpolation method used to interpolate data to grid data
interpolation_method = 'linear'

#%%%%  Define data directories
project_dir = os.path.join(main_dir, project_name)

calibration_csv_file =  os.path.join(project_dir, 'Properties', 'Calibration', 'DewarpedImages1', 'Export', 'B0001.csv')
calibration_tif_file = os.path.join(project_dir, 'Properties', 'Calibration', 'DewarpedImages1', 'Export_01', 'B0001.tif')

pre_record_raw_file = os.path.join(project_dir, pre_record_name, 'Reorganize frames', 'Export', 'B0001.tif')
pre_record_correction_file = os.path.join(project_dir, pre_record_name, 'ImageCorrection', 'Reorganize frames', 'Export', 'B0001.tif')

piv_avgV_dir = os.path.join(project_dir, record_name, piv_result, 'Avg_Stdev', 'Export')
piv_strain_dir = os.path.join(project_dir, record_name, piv_result, 'Avg_Stdev')
piv_transV_dir = os.path.join(project_dir, record_name, piv_result, 'Export')
# piv_Rstress_dir =  os.path.join(project_dir, record_name, piv_result, 'AvgVx_AvgVy_AvgV_StdevVx_StdevVy_StdevV_Rxy_Rxx_Ryy_AKE_TKE_TSS', 'Export')


 