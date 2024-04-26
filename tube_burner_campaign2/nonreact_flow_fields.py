# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 10:25:15 2023

@author: laaltenburg

"""

#%% IMPORT STANDARD PACKAGES
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import scipy.io
# from custom_colormaps import parula
# from parameters import set_mpl_params

#%% IMPORT USER DEFINED PACKAGES
from sys_paths import parent_directory
import sys_paths
import rc_params_settings
from parameters import data_dir
from plot_params import colormap, fontsize, fontsize_fraction, fontsize_legend

#%% CLOSE ALL FIGURES
plt.close('all')

#%% SET MATPLOTLIB PARAMETERS

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Computer Modern Roman"],
#     "font.size": 14.0})

# set_mpl_params()
jet = mpl.colormaps['jet']
viridis = mpl.colormaps['viridis']

#%% START
# data_dir = 'U:\\High hydrogen\\laaltenburg\\data\\tube_burner_campaign2\\selected_runs\\'

# Create an empty dictionary
non_react_dict = {}

# piv_dir = 'PIV_MP(3x16x16_75%ov_ImgCorr)'
piv_dir = 'PIV_MP(3x16x16_0%ov_ImgCorr)'

# Run 57: nonreact_h0_c3000_ls_record1
name = 'nonreact_h0_c3000_ls_record1'
session_nr = 6
recording = 'Recording_Date=230626_Time=153340_01'
piv_dir = piv_dir #'PIV_MP(3x16x16_0%ov_ImgCorr)'
run_nr = 57
Re_D_set = 3000
u_bulk_set = 1.84
u_bulk_measured = 1.84
non_react_dict[run_nr] = [name, session_nr, recording, piv_dir, Re_D_set, u_bulk_set, u_bulk_measured]

# Run 58: nonreact_h0_s4000_ls_record1
name = 'nonreact_h0_s4000_ls_record1'
session_nr = session_nr
recording = 'Recording_Date=230626_Time=153826_01'
piv_dir = piv_dir #'PIV_MP(3x16x16_0%ov_ImgCorr)'
run_nr = 58
Re_D_set = 4000
u_bulk_set = 2.46
u_bulk_measured = 2.46
non_react_dict[run_nr] = [name, session_nr, recording, piv_dir, Re_D_set, u_bulk_set, u_bulk_measured]

# Run 61: nonreact_h0_s12000_ls_record1
name = 'nonreact_h0_s12000_ls_record1'
session_nr = session_nr
recording = 'Recording_Date=230626_Time=155631_01'
piv_dir = piv_dir #'PIV_MP(3x16x16_0%ov_ImgCorr)'
run_nr = 61
Re_D_set = 12000
u_bulk_set = 7.37
u_bulk_measured = 7.38
non_react_dict[run_nr] = [name, session_nr, recording, piv_dir, Re_D_set, u_bulk_set, u_bulk_measured]

# Run 62: nonreact_h0_s16000_ls_record1
name = 'nonreact_h0_s16000_ls_record1'
session_nr = session_nr
recording = 'Recording_Date=230626_Time=160234_01'
piv_dir = piv_dir #'PIV_MP(3x16x16_0%ov_ImgCorr)'
run_nr = 62
Re_D_set = 16000
u_bulk_set = 9.83
u_bulk_measured = 9.84
non_react_dict[run_nr] = [name, session_nr, recording, piv_dir, Re_D_set, u_bulk_set, u_bulk_measured]

# Run 63: nonreact_h0_s23000_ls_record1
name = 'nonreact_h0_s23000_ls_record1'
session_nr = session_nr
recording = 'Recording_Date=230626_Time=160825_01'
piv_dir = piv_dir #'PIV_MP(3x16x16_0%ov_ImgCorr)'
run_nr = 63
Re_D_set = 23000
u_bulk_set = 14.13
u_bulk_measured = 14.13
non_react_dict[run_nr] = [name, session_nr, recording, piv_dir, Re_D_set, u_bulk_set, u_bulk_measured]

#%% FUNCTIONS
def read_csv(filename):
    
    # Read a CSV file into a pandas dataframe
    df = pd.read_csv(filename)
    
    return df

def read_mat(filename):
    
    # Load the .mat file
    mat = scipy.io.loadmat(filename)
    
    return mat

#%% START OF CODE
# Before executing code, Python interpreter reads source file and define few special variables/global variables. 
# If the python interpreter is running that module (the source file) as the main program, it sets the special __name__ variable to have a value “__main__”. If this file is being imported from another module, __name__ will be set to the module’s name. Module’s name is available as value to __name__ global variable. 
if __name__ == '__main__':
    
    D_in = 25.16 # Inner diameter of the quartz tube, units: mm
    offset = 1  # Distance from calibrated y=0 to tube burner rim
    
    width, height = 9, 6
    # Initialize axial velocity plot
    fig1, ax1 = plt.subplots(figsize=(width, height))
    
    # Initialize Ruu velocity plot (RADIAL DIRECTION)
    fig2, ax2 = plt.subplots(figsize=(width, height))
    
    # Initialize Rxy velocity plot
    fig3, ax3 = plt.subplots(figsize=(width, height))
    
    # Initialize Ryy velocity plot (AXIAL DIRECTION)
    fig4, ax4 = plt.subplots(figsize=(width, height))
    
    # Set color map
    colormap = colormap(np.linspace(0, 1, len(non_react_dict)))
    # colormap = viridis(np.linspace(0, 1, len(non_react_dict)))
    
    # Create color iterator for plotting the data
    colormap_iter = iter(colormap)
    
    u_axial_col_index = 3
    R_xx_col_index = 15 # RADIAL DIRECTION
    R_xy_col_index = 16
    R_yy_col_index = 17 # AXIAL DIRECTION
    
    col_indices = [u_axial_col_index, R_xx_col_index, R_xy_col_index, R_yy_col_index]
    axs = [ax1, ax2, ax3, ax4]
    
    for key, values in non_react_dict.items():
        name = values[0]
        session_nr = values[1]
        recording = values[2]
        piv_dir = values[3]
        Re_D = values[4]
        u_bulk_set = values[5]
        u_bulk_measured = values[6]
        
        Avg_Stdev_file = os.path.join(data_dir, f'session_{session_nr:03d}', recording, piv_dir, 'Avg_Stdev', 'Export', 'B0001.csv')
    
        df = read_csv(Avg_Stdev_file)
        
        # Get the column headers
        headers = df.columns
        
        # Determine profile location
        pivot_df = pd.pivot_table(df, values=headers[4], index=headers[1], columns=headers[0])
        
        # Given index. This corresponds to a distance of 0.19*D_in above the tube
        distance_above_tube = D_in*0.19 - offset
        
        # Find the two closest indices to the given index
        distance_above_tube_below = pivot_df.index[pivot_df.index <= distance_above_tube].max()
        distance_above_tube_above = pivot_df.index[pivot_df.index >= distance_above_tube].min()
        
        c = next(colormap_iter)
        
        for i, col_index in enumerate(col_indices):
            
            # Pivot the DataFrame using pivot_table
            pivot_df = pd.pivot_table(df, values=headers[col_index], index=headers[1], columns=headers[0])
        
            pivot_df_interp = pivot_df.loc[[distance_above_tube_below, distance_above_tube_above]]
            pivot_df_interp.loc[distance_above_tube] = np.nan
            pivot_df_interp.sort_index(inplace=True)
            pivot_df_interp.interpolate(method='index', inplace=True)
            profile_variable = pivot_df_interp.loc[distance_above_tube]
            
            wall_center_to_origin = 2
            wall_thickness = 1.5
            offset_to_wall_center = wall_center_to_origin - wall_thickness/2
            r = profile_variable.index - (D_in/2 - offset_to_wall_center)
            r /= D_in
            
            ax = axs[i]
            
            if i == 0:
                
                # Define your integration limits
                r_min, r_max = 0, 0.5
                
                # Mask to select the data within your integration range
                mask = (r >= r_min) & (r <= r_max)
                
                # Select the x and y data within your integration range
                r_range = r[mask]
                profile_variable_range = profile_variable.values[mask]
                
                # Check if x_min and x_max are in x_range, if not, interpolate
                if r_min not in r_range:
                    profile_variable_min = np.interp(r_min, r, profile_variable)
                    r_range = np.insert(r_range, 0, r_min)
                    profile_variable_range = np.insert(profile_variable_range, 0, profile_variable_min)
                
                if r_max not in r_range:
                    profile_variable_max = np.interp(r_max, r, profile_variable)
                    r_range = np.append(r_range, r_max)
                    profile_variable_range = np.append(profile_variable_range, profile_variable_max)
                
                # Calculate the integral using the trapezoidal rule
                integral = np.trapz(profile_variable_range*r_range, r_range)
                u_bulk_integral = 2*integral/(0.5**2)
                
                print('U_bulk:', u_bulk_integral, ' Deviation:', (u_bulk_measured - u_bulk_integral)/ u_bulk_integral *100)

                ax.scatter(r, profile_variable/u_bulk_measured, marker='o', facecolors=c, edgecolors='k', ls='None', label=r'${:d}$'.format(Re_D))
            
            else:
                
                ax.scatter(r, profile_variable/u_bulk_measured**2, marker='o', facecolors=c, edgecolors='k', ls='None', label=r'${:d}$'.format(Re_D))
        
    
    #%% Reference data: LDA measurements Mark Tummers et al.
    filename_u_axial_ref = 'tummers_lda_u_axial.mat'
    filename_Rxx_ref = 'tummers_lda_Rxx.mat'
    filename_Rxy_ref = 'tummers_lda_Rxy.mat'
    filename_Ryy_ref = 'tummers_lda_Ryy.mat'
    
    Re_D_ref = 23000
    label_ref = r'${:d}$, LDA'.format(Re_D_ref)
    
    u_axial_ref_file = read_mat(os.path.join(parent_directory, 'ref_data', filename_u_axial_ref)) #read_mat(os.path.join('ref_data', filename_u_axial_ref))
    u_axial_ref = u_axial_ref_file['ydat'].flatten()
    r_ref = u_axial_ref_file['xdat'].flatten()
    ax1.scatter(r_ref, u_axial_ref, marker='s', color='k', ls='None', label=label_ref)
    ax1.set_xlim([-.6, .6])
    ax1.set_ylim([0, 1.6])
    ax1.grid()
    # ax1.set_aspect('equal')
    xlabel = r'$r/D$'
    ylabel = r'$\frac{\overline{u_{x}}}{U_{b}}$'
    ax1.set_xlabel(xlabel, fontsize=fontsize)
    ax1.set_ylabel(ylabel, rotation=0, labelpad=15, fontsize=fontsize_fraction)
    
    Rxx_ref_file = read_mat(os.path.join(parent_directory, 'ref_data', filename_Rxx_ref))
    Rxx_ref = Rxx_ref_file['ydat'].flatten()
    r_ref = Rxx_ref_file['xdat'].flatten()
    ax2.scatter(r_ref, Rxx_ref, marker='s', color='k', ls='None', label=label_ref)
    ax2.set_xlim([-.6, .6])
    # ax2.set_ylim([0, .06])
    ax2.grid()
    # ax2.set_aspect('equal')
    ylabel = r'$\frac{R_{rr}}{U_{b}^2}$'
    ax2.set_xlabel(xlabel, fontsize=fontsize)
    ax2.set_ylabel(ylabel, rotation=0, labelpad=15, fontsize=fontsize_fraction)
    
    Rxy_ref_file = read_mat(os.path.join(parent_directory, 'ref_data', filename_Rxy_ref))
    Rxy_ref = Rxy_ref_file['ydat'].flatten()
    r_ref = Rxy_ref_file['xdat'].flatten()
    ax3.scatter(r_ref, Rxy_ref, marker='s', color='k', ls='None', label=label_ref)
    ax3.set_xlim([-.6, .6])  
    ax3.set_ylim([-.015, .015])
    ax3.grid()
    # ax3.set_aspect('equal')
    ylabel = r'$\frac{R_{rx}}{U_{b}^2}$' # \overline{u\'v\'
    ax3.set_xlabel(xlabel, fontsize=fontsize)
    ax3.set_ylabel(ylabel, rotation=0, labelpad=15, fontsize=fontsize_fraction)
    
    Ryy_ref_file = read_mat(os.path.join(parent_directory, 'ref_data', filename_Ryy_ref))
    Ryy_ref = Ryy_ref_file['ydat'].flatten()
    r_ref = Ryy_ref_file['xdat'].flatten()
    ax4.scatter(r_ref, Ryy_ref, marker='s', color='k', ls='None', label=label_ref)
    ax4.set_xlim([-.6, .6])
    ax4.set_ylim([0, .06])
    ax4.grid()
    # ax4.set_aspect('equal')
    ylabel = r'$\frac{R_{xx}}{U_{b}^2}$' # \overline{v\'v\'
    ax4.set_xlabel(xlabel, fontsize=fontsize)
    ax4.set_ylabel(ylabel, rotation=0, labelpad=15, fontsize=fontsize_fraction)
    
    # ax1.legend()
    # ax2.legend()
    # ax3.legend()
    # ax4.legend()
    
    ax1.legend(title="$Re_D$", prop={'size': fontsize_legend})
    ax2.legend(title="$Re_D$", prop={'size': fontsize_legend})
    ax3.legend(title="$Re_D$", prop={'size': fontsize_legend})
    ax4.legend(title="$Re_D$", prop={'size': fontsize_legend})
    
    # Get a list of all currently opened figures
    figure_ids = plt.get_fignums()

    # Apply tight_layout to each figure
    for fid in figure_ids:
        fig = plt.figure(fid)
        fig.tight_layout()
        svg_path = os.path.join('figures', 'nonreacting_flow', f'nonreact_fig{fid}.svg')
        fig.savefig(svg_path, format="svg", dpi=300, bbox_inches="tight")
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    