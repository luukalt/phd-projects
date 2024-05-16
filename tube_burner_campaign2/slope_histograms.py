# -*- coding: utf-8 -*-
"""
Created on Wed May 15 16:10:31 2024

@author: luuka
"""

#%% IMPORT STANDARD PACKAGES
import os
import json
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
from parameters import data_dir, flame, piv_method, interpolation_method
from plot_params import colormap, fontsize
from time_averaging_terms import fans_terms
from plot_functions import plot_streamlines_reacting_flow, plot_mass_cons, plot_fans_terms, plot_pressure_along_streamline
from functions import process_df, contour_correction

figures_folder = 'figures'
if not os.path.exists(figures_folder):
        os.makedirs(figures_folder)
        
pickles_folder = 'pickles'
if not os.path.exists(pickles_folder):
        os.makedirs(pickles_folder)

#%% Plots

# # Save list to a file
# with open(os.path.join("json", f'{flame.name}_all_contour_slope_values_between_025_175.json'), 'w') as f:
#     json.dump(all_contour_slope_values, f)

figs, ax = plt.subplots()

# Load list from file
flame_names = ['react_h0_s4000_ls_record1', 'react_h0_c3000_ls_record1']
labels = ['DNG-4000', 'DNG-3000'] 
ytop_limit = 4

flame_names = ['react_h100_s16000_ls_record1', 'react_h100_c12500_ls_record1']
labels = [r'H$_{2}$-16000', r'H$_{2}$-12500']
ytop_limit = 2

colors = ['tab:blue', 'tab:red']

for flame_name, color, label in zip(flame_names, colors, labels):
    
    # file = os.path.join("json", f'{flame_name}_all_contour_slope_values_between_025_175.json')
    file = os.path.join("json", f'{flame_name}_all_contour_slope_values_between_025_125.json')
    # file = os.path.join("json", f'{flame_name}_all_contour_slope_values_all.json')
    
    with open(file, 'r') as f:
        all_contour_slope_values = json.load(f)
    
    
    n, bins, _ = ax.hist(all_contour_slope_values, color=color, alpha=0.5, bins=25, edgecolor='black', density=True, label=label)
    
    # ax.axvline(x=.5, color='red', linestyle='--', linewidth=2)
    print(vars(flame))
    print(np.mean(all_contour_slope_values))
    print(len(all_contour_slope_values))
    
    ax.grid(True)
    # ax.set_title(flame.name)
    ax.set_xlabel('slope', fontsize=fontsize)
    ax.set_ylabel('pdf', fontsize=fontsize)
    ax.set_ylim(top=ytop_limit)
    ax.legend(fontsize=fontsize)
    
    # Calculate the width of each bin
    bin_width = bins[1] - bins[0]
    
    # Calculate the sum of areas of the bins
    total_area = sum(n * bin_width)
    
    print("Sum of areas of bins after normalization:", total_area)