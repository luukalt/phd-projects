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
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import griddata

#%% IMPORT USER DEFINED PACKAGES
from sys_paths import parent_directory
import sys_paths
import rc_params_settings
from parameters import data_dir, flame, piv_method, interpolation_method
from plot_params import colormap, fontsize, fontsize_label, fontsize_fraction
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

width, height = 9, 6
fig, ax = plt.subplots(figsize=(width, height))

# Load list from file
# flame_names = ['react_h0_s4000_ls_record1', 'react_h0_c3000_ls_record1']
# labels = ['DNG-4000', 'DNG-3000'] 
# ytop_limit = .025

flame_names = ['react_h100_s16000_ls_record1', 'react_h100_c12500_ls_record1']
labels = [r'H$_{2}$-16000', r'H$_{2}$-12500']
ytop_limit = .01

colors = ['tab:blue', 'tab:red']

all_contour_slope_values_degrees_list = []

for flame_name, color, label in zip(flame_names, colors, labels):
    
    # file = os.path.join("json", f'{flame_name}_all_contour_slope_values_between_025_175.json')
    file = os.path.join("json", f'{flame_name}_all_contour_slope_values_between_025_125.json')
    # file = os.path.join("json", f'{flame_name}_all_contour_slope_values_all.json')
    
    with open(file, 'r') as f:
        all_contour_slope_values = json.load(f)
    
    # Normalization factor
    norm = np.pi
    
    # Calculate the angles in degrees for all values in the list
    all_contour_slope_values_degrees = [np.degrees(alpha_tan * norm) - 90 for alpha_tan in all_contour_slope_values]
    
    all_contour_slope_values_degrees_list.append(all_contour_slope_values_degrees)
    
    # n, bins, _ = ax.hist(all_contour_slope_values_degrees, color=color, alpha=0.5, bins=25, edgecolor='black', density=True, label=label)


    # ax.axvline(x=.5, color='red', linestyle='--', linewidth=2)
    # print(vars(flame))
    print(np.mean(all_contour_slope_values_degrees) * 2)
    print(np.mean(all_contour_slope_values))
    print(len(all_contour_slope_values_degrees))
    
    # # Calculate the width of each bin
    # bin_width = bins[1] - bins[0]
    
    # # Calculate the sum of areas of the bins
    # total_area = sum(n * bin_width)
    
    # print("Sum of areas of bins after normalization:", total_area)

bins = 25
ax.hist(all_contour_slope_values_degrees_list, bins, label=labels, edgecolor='black', density=True,)
ax.grid(True)
# ax.set_title(flame.name)
ax.set_xlabel(r'$\theta$', fontsize=fontsize_fraction)
ax.set_ylabel('probability density', fontsize=fontsize_label)

num_ticks = 7
xticks = np.linspace(-90, 90, num_ticks)
ax.set_xticks(xticks)

ax.set_ylim(top=ytop_limit)
y_min, y_max = ax.get_ylim()
num_ticks = 6
yticks = np.linspace(y_min, y_max, num_ticks)
ax.set_yticks(yticks)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

ax.legend(fontsize=fontsize)
ax.tick_params(axis='both', labelsize=fontsize)

fig.tight_layout()
   
# %%% Save images
# Get a list of all currently opened figures
figure_ids = plt.get_fignums()
figure_ids = [1]

if 'ls' in flame.name:
    folder = 'ls'
else:
    folder = 'hs'

figures_subfolder = os.path.join(figures_folder, folder)
if not os.path.exists(figures_subfolder):
        os.makedirs(figures_subfolder)

# Apply tight_layout to each figure
for fid in figure_ids:
    fig = plt.figure(fid)
    filename = f'H100_fig{fid}_slope_histogram'
    
    # Get the current width and height of the figure
    current_width, current_height = fig.get_size_inches()
    
  
    eps_path = os.path.join('figures', f'{folder}', f"{filename}.eps")
   
    # Saving the figure in EPS format
    fig.savefig(eps_path, format='eps', dpi=300, bbox_inches='tight')
    
    # Get the current width and height of the figure
    current_width, current_height = fig.get_size_inches()
    
    print("Current Width:", current_width)
    print("Current Height:", current_height)
    
