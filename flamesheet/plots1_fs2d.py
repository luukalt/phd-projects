# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 16:11:23 2022

@author: laaltenburg
"""
#%% IMPORT PACKAGES 
import os
import pickle
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle, Polygon, Circle
import numpy as np
import scipy
import scipy.ndimage
import pandas as pd
from scipy.integrate import trapezoid
from scipy.interpolate import griddata
import progressbar

#%% IMPORT USER DEFINED PACKAGES
from sys_paths import parent_directory
import sys_paths
import rc_params_settings
from parameters import *
from plot_params import colormap, fontsize, fontsize_legend, fontsize_label, fontsize_fraction

figures_folder = 'figures'
if not os.path.exists(figures_folder):
        os.makedirs(figures_folder)
        
pickles_folder = 'pickles'
if not os.path.exists(pickles_folder):
        os.makedirs(pickles_folder)
        

if day_nr == '24-1':
    from wall_detection_day24_1 import *
if day_nr == '23-2':
    from wall_detection_day23_2 import *
    

#%% HEADERS OF Avg_Stdev (B0001.csv)

# 1. 'x [mm]'
# 2. 'y [mm]'
# 3. 'Velocity u [m/s]'
# 4. 'Velocity v [m/s]'
# 5. 'Velocity |V| [m/s]'
# 6. 'du/dx [1/s]'
# 7. 'du/dy [1/s]'
# 8. 'dv/dx [1/s]'
# 9. 'dv/dy [1/s]'
# 10. 'Vorticity w_z (dv/dx - du/dy) [1/s]'
# 11. '|Vorticity| [1/s]'
# 12. 'Divergence 2D (du/dx + dv/dy) [1/s]'
# 13. 'Swirling strength 2D (L_ci) [1/s^2]'
# 14. 'Average kinetic energy [(m/s)^2]'
# 15. 'Number of vectors [n]'
# 16. 'Reynolds stress Rxx [(m/s)^2]'
# 17. 'Reynolds stress Rxy [(m/s)^2]'
# 18. 'Reynolds stress Ryy [(m/s)^2]'
# 19. 'Standard deviation Vx [m/s]'
# 20. 'Standard deviation Vy [m/s]'
# 21. 'Turbulent kinetic energy [(m/s)^2]'
# 22. 'Turbulent shear stress [(m/s)^2]'
      
       
#%% START
plt.close("all")

#%% FIGURE SETTINGS

# Color maps
tableau = cm.tab10.colors

#%% MAIN FUNCTIONS
def read_xy_dimensions(file):
    
    df_raw = pd.read_csv(file)
    
    # Get the column headers of the file
    headers = df_raw.columns
    
    pivot_intensity = pd.pivot_table(df_raw, values=headers[2], index=headers[1], columns=headers[0])
    
    n_windows_x_raw = pivot_intensity.columns.size
    n_windows_y_raw = pivot_intensity.index.size
    
    # Create r,x raw Mie scattering grid
    x_raw_array = pivot_intensity.columns
    y_raw_array = pivot_intensity.index
    x_raw, y_raw = np.meshgrid(x_raw_array, y_raw_array)
    window_size_x_raw, window_size_y_raw = np.mean(np.diff(x_raw_array)), -np.mean(np.diff(y_raw_array))
    
    # Parameters for correcting contours from pixel coordinates to physical coordinates
    x_left_raw = x_raw_array[0]
    x_right_raw = x_raw_array[-1]
    y_bottom_raw = y_raw_array[0]
    y_top_raw = y_raw_array[-1]
    
    # print(n_windows_x_raw, n_windows_y_raw)
    # print(y_bottom_raw, y_top_raw)
    
    return x_raw, y_raw, x_left_raw, x_right_raw, y_bottom_raw, y_top_raw, window_size_x_raw, window_size_y_raw


def read_flow_data(file, normalized):
    
    # Set if plot is normalized or non-dimensionalized
    if normalized:
        u_bulk = u_bulk_measured
    else:
        u_bulk = 1
    
    print(f'Set normalized: {normalized}')
    print(f'Set u_bulk_measured: {u_bulk_measured}')
    
    # File name and scaling parameters from headers of file
    df_piv = pd.read_csv(file)
    
    # Get the column headers of the file
    headers = df_piv.columns
    
    # print(headers)
    
    pivot_U = pd.pivot_table(df_piv, values=headers[2], index=headers[1], columns=headers[0])
    pivot_V = pd.pivot_table(df_piv, values=headers[3], index=headers[1], columns=headers[0])
    pivot_absV = pd.pivot_table(df_piv, values=headers[4], index=headers[1], columns=headers[0])
    
    pivot_RXX = pd.pivot_table(df_piv, values=headers[16], index=headers[1], columns=headers[0])
    pivot_RXY = pd.pivot_table(df_piv, values=headers[17], index=headers[1], columns=headers[0])
    pivot_RYY = pd.pivot_table(df_piv, values=headers[18], index=headers[1], columns=headers[0])
    pivot_TKE = pd.pivot_table(df_piv, values=headers[21], index=headers[1], columns=headers[0])
    
    pivot_U /= u_bulk
    pivot_V /= u_bulk
    pivot_absV /= u_bulk
    pivot_RXX /= u_bulk**2
    pivot_RXY /= u_bulk**2
    pivot_RYY /= u_bulk**2
    pivot_TKE /= u_bulk**2
    
    n_windows_x = pivot_absV.columns.size
    n_windows_y = pivot_absV.index.size
    
    X_array = pivot_absV.columns
    Y_array = pivot_absV.index
    X, Y = np.meshgrid(X_array, Y_array)
    
    
    # Flow strain
    # List of strain components
    strain_components = ['Exx', 'Exy', 'Eyx', 'Eyy', '(Exy+Eyx)_DIV_2']
    df_strain = None

    for component in strain_components:
        strain_file = os.path.join(piv_strain_dir, component, 'Export', csv_file)
        df = pd.read_csv(strain_file)
        
        # Rename the component column (e.g., 'Exx' -> 'Exx')
        df.rename(columns={component: component}, inplace=True)
        
        if df_strain is None:
            df_strain = df
        else:
            # Merge based on columns 'x' and 'y'
            df_strain = pd.merge(df_strain, df, on=['x [mm]', 'y [mm]'], how='outer')
    
    headers = df_strain.columns
    
    # print(headers)
    
    pivot_EXX = pd.pivot_table(df_strain, values=headers[2], index=headers[1], columns=headers[0])
    pivot_EXY = pd.pivot_table(df_strain, values=headers[3], index=headers[1], columns=headers[0])
    pivot_EYX = pd.pivot_table(df_strain, values=headers[4], index=headers[1], columns=headers[0])
    pivot_EYY = pd.pivot_table(df_strain, values=headers[5], index=headers[1], columns=headers[0])
    pivot_EXY_EYX_div_2 = pd.pivot_table(df_strain, values=headers[6], index=headers[1], columns=headers[0]) 
    
    return n_windows_x, n_windows_y, X, Y, pivot_U.values, pivot_V.values, pivot_absV.values, pivot_RXX.values, pivot_RXY.values, pivot_RYY.values, pivot_TKE.values, pivot_EXX.values, pivot_EXY.values, pivot_EYX.values, pivot_EYY.values, pivot_EXY_EYX_div_2.values 


def plot_field(fig, ax, X, Y, quantity, label, cmin, cmax):
    
    quantity_plot = ax.pcolor(X, Y, quantity, cmap="viridis", rasterized=True)
    quantity_plot.set_clim(cmin, cmax)
    
    # Set x- and y-label
    ax.set_xlabel('$x$ [mm]', fontsize=fontsize)
    ax.set_ylabel('$y$ [mm]', fontsize=fontsize)
    
    # Set aspect ratio of plot
    ax.set_aspect('equal')
    
    # Set contour bar
    bar = fig.colorbar(quantity_plot, ax=ax)
    bar.set_label(label)
    
    # ax.set_xlim(left=-.55, right=.55)
    ax.set_ylim(bottom=-23)
    
def plot_vector_field(fig, ax, X, Y, Vx, Vy):
    scale = 1
    headwidth = 6
    color = "r"
    skip = 2
    ax.quiver(X[0::skip, 0::skip], Y[0::skip, 0::skip], Vx[0::skip, 0::skip], Vy[0::skip, 0::skip], color=color, angles='xy', scale_units='xy', scale=scale, headwidth=headwidth)
    
def plot_streamlines(fig, ax, X, Y, Vx, Vy):
    
    # Vx[Vx == -0] = 0
    # Vy[Vy == -0] = 0
    
    # XYUV[XYUV == -0] = 0
    
    # X = XYUV[:,0].reshape(n_windows_y, n_windows_x)
    # Y = XYUV[:,1].reshape(n_windows_y, n_windows_x)
    # Vx = XYUV[:,2].reshape(n_windows_y, n_windows_x) # *-1 because -> inverse x-axis
    # Vy = XYUV[:,3].reshape(n_windows_y, n_windows_x)
    Vx = AvgVx
    Vy = AvgVy
    
    # row_start, row_end = 0, 100
    # col_start, col_end = 6, 25
    
    # X = X[row_start:row_end, col_start:col_end]
    # Y = Y[row_start:row_end, col_start:col_end]
    # Vx = Vx[row_start:row_end, col_start:col_end]
    # Vy = Vy[row_start:row_end, col_start:col_end]
    
    X_values = X.flatten()
    Y_values = Y.flatten()
    
    # skip = 3
    # X = X[0::skip, 0::skip]
    # Y = Y[0::skip, 0::skip]
    # Vx = Vx[0::skip, 0::skip]
    # Vy = Vy[0::skip, 0::skip]
    
    # Regularly spaced grid spanning the domain of x and y 
    Xi = np.linspace(X.min(), X.max(), X.shape[1])
    Yi = np.linspace(Y.min(), Y.max(), Y.shape[0])
    Xi, Yi = np.meshgrid(Xi, Yi)
    
    # Bicubic interpolation
    # Vxi = interp2d(X, Y, Vx)(Xi, Yi)
    # Vyi = interp2d(X, Y, Vy)(Xi, Yi)
    
    streamline_interpolation_method = 'linear'
    Vxi = griddata((X_values, Y_values), Vx.flatten(), (Xi, Yi), method=streamline_interpolation_method)
    Vyi = griddata((X_values, Y_values), Vy.flatten(), (Xi, Yi), method=streamline_interpolation_method)
    
    # fig, ax = plt.subplots()
    
    # ax.pcolor(Xi, Yi, Vyi)
    
    # Streamlines starting points
    skip_points = 4
    # start_points_1 = [[-1, j] for j in range(-18, -14, int(skip_points/2))]
    # start_points_1 = [[i, 10] for i in range(3, 25, skip_points)]
    
    # start_points_2 = [[i, 30] for i in range(3, 25, skip_points*2)]
    
    # start_points_3 = [[-1, j] for j in range(-18, -14, int(skip_points/2))]
    
    # start_points = start_points_1 + start_points_2 # + start_points_3
    
    start_points = [[20, 20]]
    
    # StreamplotSet = ax.streamplot(xi, yi, uCi, vCi, density=0.25, color="k", minlength=0.1, arrowsize=2, broken_streamlines=False)
    streamlines = ax.streamplot(Xi, Yi, Vxi, Vyi, start_points=start_points, density=0.5, color="k", minlength=0.1, arrowsize=2, broken_streamlines=False)
    
    return streamlines

    
def plot_profile_in_field(fig, ax, coord0_mm, coord1_mm, color, num):
    
    x0_mm, y0_mm = coord0_mm
    x1_mm, y1_mm = coord1_mm
    # x0, y0 = (coord0_mm - np.array([X0, Y0]))/np.array([dx_piv, dy_piv])
    # x1, y1 = (coord1_mm - np.array([X0, Y0]))/np.array([dx_piv, dy_piv])
    
    # x_profile, y_profile = np.linspace(x0, x1, num), np.linspace(y0, y1, num)
    x_profile, y_profile = np.linspace(x0_mm, x1_mm, num), np.linspace(y0_mm, y1_mm, num)
    profile_coords = np.column_stack((x_profile, y_profile))
    
    # # Extract the values along the line, using first, second or third order interpolation
    # profile_line = np.linspace(0, np.sqrt((x1_mm - x0_mm)**2 + (y1_mm - y0_mm)**2), num)
    # profile_line = profile_line[1:-1]
    
    # Calculate cumulative distance along the profile
    distances = np.sqrt(np.diff(x_profile)**2 + np.diff(y_profile)**2)
    cumulative_distances = np.cumsum(distances)
    profile_line = np.insert(cumulative_distances, 0, 0)
    
    # Extract the values along the line, using first, second or third order interpolation
    profile_line = profile_line[1:-1]
    profile_coords = profile_coords[1:-1]
    
    ax.plot(profile_coords[:, 0], profile_coords[:, 1], c=color, ls='solid')
    
    return profile_coords, profile_line


def plot_profile(fig, ax, rotation_matrix, profile_coords, profile_line, quantity_x, quantity_y, label, cmin, cmax, color, num):

    X_values = X.flatten()
    Y_values = Y.flatten()
    quantity_x_values = quantity_x.flatten()
    quantity_y_values = quantity_y.flatten()
    
    quantity_x_profile = griddata((X_values, Y_values), quantity_x_values, profile_coords, method=interpolation_method)
    quantity_y_profile = griddata((X_values, Y_values), quantity_y_values, profile_coords, method=interpolation_method)
    
    # quantity_x_profile[0] = 0
    # quantity_y_profile[0] = 0
    # quantity_x_profile[-1] = 0
    # quantity_y_profile[-1] = 0
    
    quantity_tangent_profile, quantity_normal_profile  = np.dot(rotation_matrix, np.array([quantity_x_profile, quantity_y_profile]))
    
    # quantity_tangent_profile = quantity_x_profile*np.cos(theta) - quantity_y_profile*np.sin(theta)
    # quantity_normal_profile = quantity_x_profile*np.sin(theta) + quantity_y_profile*np.cos(theta)
    
    ax.plot(profile_line, quantity_normal_profile, c=color, ls='solid', marker='o')
    
    ax.set_xlabel("distance along line [mm]", fontsize=fontsize)
    ax.set_ylabel("$V_{n}$ [ms$^-1$]", fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    # ax[1].set_ylabel("$R_{NN}$ [ms$^-1$]")
    
    
    ax.set_xlim(np.array([0, 30]))
    # ax[1].set_ylim(np.array([cmin, cmax]))
    ax.grid(True)
    ax.axhline(y=0, color='k')
    
    u_bulk_2d = trapezoid(quantity_normal_profile, profile_line)/10
    # print(trapezoid(quantity_normal_profile, profile_line))
    # print("u_bulk_measured_2d= {0:.1f} m/s".format(u_bulk_2d))
    
    return quantity_x_profile, quantity_y_profile, quantity_tangent_profile, quantity_normal_profile


def plot_curved_profile(fig, ax, profile_coords, quantity_x, quantity_y, label, cmin, cmax, color, num):

    X_values = X.flatten()
    Y_values = Y.flatten()
    quantity_x_values = quantity_x.flatten()
    quantity_y_values = quantity_y.flatten()
    
    quantity_x_profile = griddata((X_values, Y_values), quantity_x_values, profile_coords, method=interpolation_method)
    quantity_y_profile = griddata((X_values, Y_values), quantity_y_values, profile_coords, method=interpolation_method)
    
    # quantity_x_profile[0] = 0
    # quantity_y_profile[0] = 0
    # quantity_x_profile[-1] = 0
    # quantity_y_profile[-1] = 0
    
    # Compute rotation matrices along the curved profile
    rotation_matrices, thetas = compute_rotation_matrices(profile_coords)
    
    quantity_tangent_profile = []
    quantity_normal_profile = []
    
    for i in range(len(profile_coords)):
        
        rotation_matrix = rotation_matrices[i]
        qx = quantity_x_profile[i]
        qy = quantity_y_profile[i]
        qt, qn = np.dot(rotation_matrix, np.array([qx, qy]))
        quantity_tangent_profile.append(qt)
        quantity_normal_profile.append(qn)
    
    quantity_tangent_profile = np.array(quantity_tangent_profile)
    quantity_normal_profile = np.array(quantity_normal_profile)
    
    # quantity_tangent_profile, quantity_normal_profile  = np.dot(rotation_matrix, np.array([quantity_x_profile, quantity_y_profile]))
    
    # quantity_tangent_profile = quantity_x_profile*np.cos(theta) - quantity_y_profile*np.sin(theta)
    # quantity_normal_profile = quantity_x_profile*np.sin(theta) + quantity_y_profile*np.cos(theta)
    
    x_profile, y_profile = profile_coords[:, 0], profile_coords[:, 1]
    distances = np.sqrt(np.diff(x_profile)**2 + np.diff(y_profile)**2)
    cumulative_distances = np.cumsum(distances)
    profile_line = np.insert(cumulative_distances, 0, 0)
    
    ax.plot(profile_line, quantity_tangent_profile, ls='solid', marker=marker, label= r'$V_{t}$')
    ax.plot(profile_line, quantity_normal_profile,  ls='solid', marker=marker, label= r'$V_{n}$')
    # ax.plot(profile_line, thetas, c='k', ls='dashed', marker=marker)
    # ax.plot(profile_line[250], thetas[250], c='c', ls='None', marker='x')
    
    
    ax.set_xlabel("distance along line [mm]")
    ax.set_ylabel("Velocity [ms$^-1$]")
    # ax.tick_params(axis='both', labelsize=fontsize)
    # ax[1].set_ylabel("$R_{NN}$ [ms$^-1$]")
    
    
    # ax.set_xlim(np.array([0, 30]))
    # ax[1].set_ylim(np.array([cmin, cmax]))
    ax.grid(True)
    ax.axhline(y=0, color='k')
    
    u_bulk_2d = trapezoid(quantity_normal_profile, profile_line)/10
    # print(trapezoid(quantity_normal_profile, profile_line))
    # print("u_bulk_measured_2d= {0:.1f} m/s".format(u_bulk_2d))
    
    return quantity_x_profile, quantity_y_profile, quantity_tangent_profile, quantity_normal_profile
    
    
def plot_reynolds_stress(fig, ax, rotation_matrix, theta, profile_coords, profile_line, label, cmin, cmax, color, num):
    
    X_values = X.flatten()
    Y_values = Y.flatten()
    
    RXX_values = RXX.flatten()
    RYY_values = RYY.flatten()
    RXY_values = RXY.flatten()
    
    RXX_profile = griddata((X_values, Y_values), RXX_values, profile_coords, method=interpolation_method)
    RYY_profile = griddata((X_values, Y_values), RYY_values, profile_coords, method=interpolation_method)
    RXY_profile = griddata((X_values, Y_values), RXY_values, profile_coords, method=interpolation_method)
    
    # RXX_profile = scipy.ndimage.map_coordinates(RXX, np.vstack((y_profile, x_profile)), order=order)
    # RYY_profile = scipy.ndimage.map_coordinates(RYY, np.vstack((y_profile, x_profile)), order=order)
    # RXY_profile = scipy.ndimage.map_coordinates(RXY, np.vstack((y_profile, x_profile)), order=order)
    
    # ### Approach 1: Calculate Reynolds stresses on arbirtary line "manually"
    # RTT = RXX_profile*(np.cos(theta))**2 - 2*RXY_profile*np.cos(theta)*np.sin(theta) + RYY_profile*(np.sin(theta))**2
    # RTN = (RXX_profile - RYY_profile)*np.cos(theta)*np.sin(theta) + RXY_profile*((np.cos(theta))**2 - (np.sin(theta))**2)
    # RNN = RXX_profile*(np.sin(theta))**2 + 2*RXY_profile*np.cos(theta)*np.sin(theta) + RYY_profile*(np.cos(theta))**2
    
    ### Approach 2: Calculate Reynolds stresses on arbirtary line with matrix multiplication
    T_stress = np.array(((RXX_profile, RXY_profile), (RXY_profile, RYY_profile)))
    T_stress_rotated = np.zeros_like(T_stress)
    
    for i in range(T_stress.shape[2]):
        T_stress_rotated[:, :, i] = rotation_matrix @ T_stress[:, :, i] @ rotation_matrix.T
    
    
    RTT = T_stress_rotated[0, 0, :]
    RNN = T_stress_rotated[1, 1, :]
    RTN = T_stress_rotated[0, 1, :]
    
    ### Approach 3: Calculate Reynolds stresses on arbirtary line using rotation matrix [CORRECT RESULT WITH "CORRECT" CODE]
    # R_stress_tensor_dummy = np.zeros([2, 2, num])
    # R_stress_tensor_rotated = np.zeros([2, 2, num])

    # for i in range(num):
    #     R_stress_tensor_dummy = rotation_matrix.dot(R_stress_tensor[:,:,i])
    #     R_stress_tensor_rotated[:,:,i] = R_stress_tensor_dummy.dot(rotation_matrix_T)
    
    # Rtt = R_stress_tensor_rotated[0,0,:]
    # Rnn = R_stress_tensor_rotated[1,1,:]
    # Rtn = R_stress_tensor_rotated[0,1,:]
    
    ax.plot(profile_line, RNN, c=color, ls="-", marker="o")
    # ax.set_xlim(np.array([arbitrary_line[0], arbitrary_line[-1]]))
    # ax.set_xlim(left=0, right=30]))
    
    ax.set_xlabel("distance along line [mm]", fontsize=fontsize)
    ax.set_ylabel("$R_{nn}$ [m$^2$s$^{-2}$]", fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.grid(True)
    
    min_value, min_index, max_value, max_index = find_min_max_indices(RNN)
    
    # ax.set_ylim(np.array([cmin, cmax]))
    
    return RXX_profile, RYY_profile, RXY_profile, min_index, max_index

# def plot_image(fig, ax, nx, ny):

#     XYI = np.genfromtxt(correction_dir +'B0001.csv', delimiter=",", skip_header=1)
#     X = XYI[:,0].reshape(ny, nx)
#     Y = XYI[:,1].reshape(ny, nx)
#     I = XYI[:,2].reshape(ny, nx)
#     intensity_plot = ax.pcolor(X, Y, I, cmap='gray')
    
#     intensity_plot = ax.pcolor(I, cmap='gray')
#     intensity_plot.set_clim(0, 4095)
    
#     ax.set_xlabel('$x [mm]$')
#     ax.set_ylabel('$y [mm]$')
    
#     # Set aspect ratio of plot
#     ax.set_aspect('equal')
    
#     bar = fig.colorbar(intensity_plot)
#     bar.set_label('Image intensity [counts]') #('$u$/U$_{b}$ [-]')
    
#     return X, Y, I, XYI

def plot_strain_rate_symmetric(fig, ax, rotation_matrix, theta, profile_coords, profile_line, label, cmin, cmax, color, num):
    
    X_values = X.flatten()
    Y_values = Y.flatten()
    
    EXX_values = EXX.flatten()
    EXY_values = EXY.flatten()
    EYX_values = EYX.flatten()
    EYY_values = EYY.flatten()
    EXY_EYX_div_2_values = EXY_EYX_div_2.flatten()
    
    EXX_profile = griddata((X_values, Y_values), EXX_values, profile_coords, method=interpolation_method)
    EXY_profile = griddata((X_values, Y_values), EXY_values, profile_coords, method=interpolation_method)
    EYX_profile = griddata((X_values, Y_values), EYX_values, profile_coords, method=interpolation_method)
    # EXY_EYX_div_2_profile = griddata((X_values, Y_values), (EXY_values + EYX_values)/2, profile_coords, method=interpolation_method)
    EXY_EYX_div_2_profile = griddata((X_values, Y_values), EXY_EYX_div_2_values, profile_coords, method=interpolation_method)
    EYY_profile = griddata((X_values, Y_values), EYY_values, profile_coords, method=interpolation_method)
    
    # ### Approach 1: Calculate Reynolds stresses on arbirtary line "manually"
    # ETT = EXX_profile*(np.cos(theta))**2 - 2*EXY_EYX_div_2_profile*np.cos(theta)*np.sin(theta) + EYY_profile*(np.sin(theta))**2
    # ETN = (EXX_profile - EYY_profile)*np.cos(theta)*np.sin(theta) + EXY_EYX_div_2_profile*((np.cos(theta))**2 - (np.sin(theta))**2)
    # ENN = EXX_profile*(np.sin(theta))**2 + 2*EXY_EYX_div_2_profile*np.cos(theta)*np.sin(theta) + EYY_profile*(np.cos(theta))**2

    ### Approach 2: Calculate Reynolds stresses on arbirtary line with matrix multiplication
    T_strain = np.array(((EXX_profile, EXY_EYX_div_2_profile), (EXY_EYX_div_2_profile, EYY_profile)))
    T_strain_rotated = np.zeros_like(T_strain)
    
    for i in range(T_strain.shape[2]):
        T_strain_rotated[:, :, i] = rotation_matrix @ T_strain[:, :, i] @ rotation_matrix.T
    
    
    ETT = T_strain_rotated[0, 0, :]
    ENN = T_strain_rotated[1, 1, :]
    ETN = T_strain_rotated[0, 1, :]
    
    ax.plot(profile_line, ENN, c=color, ls="-", marker="o")
    # ax.set_xlim(np.array([arbitrary_line[0], arbitrary_line[-1]]))
    # ax.set_xlim(left=0, right=30]))
    
    ax.set_xlabel("distance along line [mm]", fontsize=fontsize)
    ax.set_ylabel("$E_{nn}$ [$1/s$]", fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.grid(True)
    
    min_value, min_index, max_value, max_index = find_min_max_indices(ENN)
    
    # ax.plot(profile_line[min_index], ENN[min_index], c='r', marker="x")
    # ax.plot(profile_line[max_index], ENN[max_index], c='b', marker="x")
    
    return EXX_profile, EYY_profile, EXY_EYX_div_2_profile, min_index, max_index


def plot_strain_rate(fig, ax, rotation_matrix, theta, profile_coords, profile_line, label, cmin, cmax, color, num):
    
    X_values = X.flatten()
    Y_values = Y.flatten()
    
    EXX_values = EXX.flatten()
    EXY_values = EXY.flatten()
    EYX_values = EYX.flatten()
    EYY_values = EYY.flatten()
    EXY_EYX_div_2_values = EXY_EYX_div_2.flatten()
    
    EXX_profile = griddata((X_values, Y_values), EXX_values, profile_coords, method=interpolation_method)
    EXY_profile = griddata((X_values, Y_values), EXY_values, profile_coords, method=interpolation_method)
    EYX_profile = griddata((X_values, Y_values), EYX_values, profile_coords, method=interpolation_method)
    # EXY_EYX_div_2_profile = griddata((X_values, Y_values), (EXY_values + EYX_values)/2, profile_coords, method=interpolation_method)
    EXY_EYX_div_2_profile = griddata((X_values, Y_values), EXY_EYX_div_2_values, profile_coords, method=interpolation_method)
    EYY_profile = griddata((X_values, Y_values), EYY_values, profile_coords, method=interpolation_method)
    
    # ### Approach 1: Calculate Reynolds stresses on arbirtary line "manually"
    # ETT = EXX_profile*(np.cos(theta))**2 - 2*EXY_EYX_div_2_profile*np.cos(theta)*np.sin(theta) + EYY_profile*(np.sin(theta))**2
    # ETN = (EXX_profile - EYY_profile)*np.cos(theta)*np.sin(theta) + EXY_EYX_div_2_profile*((np.cos(theta))**2 - (np.sin(theta))**2)
    # ENN = EXX_profile*(np.sin(theta))**2 + 2*EXY_EYX_div_2_profile*np.cos(theta)*np.sin(theta) + EYY_profile*(np.cos(theta))**2

    ### Approach 2: Calculate Reynolds stresses on arbirtary line with matrix multiplication
    T_strain = np.array(((EXX_profile, EXY_profile), (EYX_profile, EYY_profile)))
    T_strain_rotated = np.zeros_like(T_strain)
    
    for i in range(T_strain.shape[2]):
        T_strain_rotated[:, :, i] = rotation_matrix @ T_strain[:, :, i] @ rotation_matrix.T
    
    ETT = T_strain_rotated[0, 0, :]
    ENN = T_strain_rotated[1, 1, :]
    ETN = T_strain_rotated[0, 1, :]
    ENT = T_strain_rotated[1, 0, :]
    
    ax.plot(profile_line, ENN, c=color, ls="-", marker="o")
    # ax.set_xlim(np.array([arbitrary_line[0], arbitrary_line[-1]]))
    # ax.set_xlim(left=0, right=30]))
    
    ax.set_xlabel("distance along line [mm]", fontsize=fontsize)
    ax.set_ylabel("$E_{nn}$ [$1/s$]", fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.grid(True)
    
    min_value, min_index, max_value, max_index = find_min_max_indices(ENN)
    
    # ax.plot(profile_line[min_index], ENN[min_index], c='r', marker="x")
    # ax.plot(profile_line[max_index], ENN[max_index], c='b', marker="x")
    
    return EXX_profile, EYY_profile, EXY_EYX_div_2_profile, min_index, max_index


def plot_tke(fig, ax, profile_coords, profile_line, label, cmin, cmax, color, num):

    X_values = X.flatten()
    Y_values = Y.flatten()
    TKE_values = TKE.flatten()
    
    TKE_profile = griddata((X_values, Y_values), TKE_values, profile_coords, method=interpolation_method)
    
    # quantity_tangent_profile = quantity_x_profile*np.cos(theta) - quantity_y_profile*np.sin(theta)
    # quantity_normal_profile = quantity_x_profile*np.sin(theta) + quantity_y_profile*np.cos(theta)
    
    ax.plot(profile_line, TKE_profile, c=color, ls='solid', marker='o')
    
    ax.set_xlabel("distance along line [mm]", fontsize=fontsize)
    ax.set_ylabel("TKE [m$^2$s$^{-2}$]", fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    # ax[1].set_ylabel("$R_{NN}$ [ms$^-1$]")
    
    ax.set_xlim(np.array([0, 30]))
    # ax[1].set_ylim(np.array([cmin, cmax]))
    ax.grid(True)
    ax.axhline(y=0, color='k')
    
    min_value, min_index, max_value, max_index = find_min_max_indices(TKE_profile)
    
    return TKE_profile, min_index, max_index

# def plot_curved_profile(fig, ax, profile_coords, quantity_x, quantity_y, label, cmin, cmax, color, num):

#     X_values = X.flatten()
#     Y_values = Y.flatten()
#     quantity_x_values = quantity_x.flatten()
#     quantity_y_values = quantity_y.flatten()
    
#     quantity_x_profile = griddata((X_values, Y_values), quantity_x_values, profile_coords, method=interpolation_method)
#     quantity_y_profile = griddata((X_values, Y_values), quantity_y_values, profile_coords, method=interpolation_method)
    
#     # quantity_x_profile[0] = 0
#     # quantity_y_profile[0] = 0
#     # quantity_x_profile[-1] = 0
#     # quantity_y_profile[-1] = 0
    
#     # Compute rotation matrices along the curved profile
#     rotation_matrices, thetas = compute_rotation_matrices(profile_coords)
    
#     quantity_tangent_profile = []
#     quantity_normal_profile = []
    
#     for i in range(len(profile_coords)):
        
#         rotation_matrix = rotation_matrices[i]
#         qx = quantity_x_profile[i]
#         qy = quantity_y_profile[i]
#         qt, qn = np.dot(rotation_matrix, np.array([qx, qy]))
#         quantity_tangent_profile.append(qt)
#         quantity_normal_profile.append(qn)
    
#     quantity_tangent_profile = np.array(quantity_tangent_profile)
#     quantity_normal_profile = np.array(quantity_normal_profile)
    
#     # quantity_tangent_profile, quantity_normal_profile  = np.dot(rotation_matrix, np.array([quantity_x_profile, quantity_y_profile]))
    
#     # quantity_tangent_profile = quantity_x_profile*np.cos(theta) - quantity_y_profile*np.sin(theta)
#     # quantity_normal_profile = quantity_x_profile*np.sin(theta) + quantity_y_profile*np.cos(theta)
    
#     x_profile, y_profile = profile_coords[:, 0], profile_coords[:, 1]
#     distances = np.sqrt(np.diff(x_profile)**2 + np.diff(y_profile)**2)
#     cumulative_distances = np.cumsum(distances)
#     profile_line = np.insert(cumulative_distances, 0, 0)
    
#     ax.plot(profile_line, quantity_tangent_profile, ls='solid', marker=marker, label= r'$V_{t}$')
#     ax.plot(profile_line, quantity_normal_profile, ls='solid', marker=marker, label= r'V_{n}')
#     # ax.plot(profile_line, thetas, c='k', ls='dashed', marker=marker)
#     # ax.plot(profile_line[250], thetas[250], c='c', ls='None', marker='x')
    
    
#     ax.set_xlabel("distance along line [mm]")
#     ax.set_ylabel("Velocity [ms$^-1$]")
#     # ax.tick_params(axis='both', labelsize=fontsize)
#     # ax[1].set_ylabel("$R_{NN}$ [ms$^-1$]")
    
    
#     # ax.set_xlim(np.array([0, 30]))
#     # ax[1].set_ylim(np.array([cmin, cmax]))
#     ax.grid(True)
#     ax.axhline(y=0, color='k')
    
#     u_bulk_2d = trapezoid(quantity_normal_profile, profile_line)/10
#     # print(trapezoid(quantity_normal_profile, profile_line))
#     # print("u_bulk_measured_2d= {0:.1f} m/s".format(u_bulk_2d))
    
#     return quantity_x_profile, quantity_y_profile, quantity_tangent_profile, quantity_normal_profile


def plot_curved_reynolds_stress(fig, ax, profile_coords, label, cmin, cmax, color, num):
    
    X_values = X.flatten()
    Y_values = Y.flatten()
    
    RXX_values = RXX.flatten()
    RYY_values = RYY.flatten()
    RXY_values = RXY.flatten()
    
    RXX_profile = griddata((X_values, Y_values), RXX_values, profile_coords, method=interpolation_method)
    RYY_profile = griddata((X_values, Y_values), RYY_values, profile_coords, method=interpolation_method)
    RXY_profile = griddata((X_values, Y_values), RXY_values, profile_coords, method=interpolation_method)
    
    # Compute rotation matrices along the curved profile
    rotation_matrices, thetas = compute_rotation_matrices(profile_coords)
    
    ### Approach 2: Calculate Reynolds stresses on arbirtary line with matrix multiplication
    T_stress = np.array(((RXX_profile, RXY_profile), (RXY_profile, RYY_profile)))
    T_stress_rotated = np.zeros_like(T_stress)
    
    for i in range(len(profile_coords)):
        
        rotation_matrix = rotation_matrices[i]
        
        T_stress_rotated[:, :, i] = rotation_matrix @ T_stress[:, :, i] @ rotation_matrix.T
        
    
    RTT = T_stress_rotated[0, 0, :]
    RNN = T_stress_rotated[1, 1, :]
    RTN = T_stress_rotated[0, 1, :]
    
    x_profile, y_profile = profile_coords[:, 0], profile_coords[:, 1]
    distances = np.sqrt(np.diff(x_profile)**2 + np.diff(y_profile)**2)
    cumulative_distances = np.cumsum(distances)
    profile_line = np.insert(cumulative_distances, 0, 0)
    
    ax.plot(profile_line, RTT, ls="solid", marker=marker, label= r'$R_{tt}$')
    ax.plot(profile_line, RNN, ls="solid", marker=marker, label= r'$R_{nn}$')
    # ax.plot(profile_line, thetas, c='k', ls='dashed', marker=marker)
    # ax.plot(profile_line[250], thetas[250], c='c', ls='None', marker='x')
    
    ax.set_xlabel("distance along line [mm]")
    ax.set_ylabel("Reynolds stress [m$^2$s$^{-2}$]")
    # ax.tick_params(axis='both', labelsize=fontsize)
    ax.grid(True)
    
    # ax.set_ylim(np.array([-10, 10]))
    
    return RXX_profile, RYY_profile, RXY_profile

def plot_curved_strain_rate_symmetric(fig, ax, profile_coords, label, cmin, cmax, color, num):
    
    X_values = X.flatten()
    Y_values = Y.flatten()
    
    EXX_values = EXX.flatten()
    EXY_values = EXY.flatten()
    EYX_values = EYX.flatten()
    EYY_values = EYY.flatten()
    EXY_EYX_div_2_values = EXY_EYX_div_2.flatten()
    
    EXX_profile = griddata((X_values, Y_values), EXX_values, profile_coords, method=interpolation_method)
    EXY_profile = griddata((X_values, Y_values), EXY_values, profile_coords, method=interpolation_method)
    EYX_profile = griddata((X_values, Y_values), EYX_values, profile_coords, method=interpolation_method)
    # EXY_EYX_div_2_profile = griddata((X_values, Y_values), (EXY_values + EYX_values)/2, profile_coords, method=interpolation_method)
    EXY_EYX_div_2_profile = griddata((X_values, Y_values), EXY_EYX_div_2_values, profile_coords, method=interpolation_method)
    EYY_profile = griddata((X_values, Y_values), EYY_values, profile_coords, method=interpolation_method)
    
    # Compute rotation matrices along the curved profile
    rotation_matrices, thetas = compute_rotation_matrices(profile_coords)
    
    ### Approach 2: Calculate Reynolds stresses on arbirtary line with matrix multiplication
    T_strain = np.array(((EXX_profile, EXY_EYX_div_2_profile), (EXY_EYX_div_2_profile, EYY_profile)))
    T_strain_rotated = np.zeros_like(T_strain)
    
    for i in range(len(profile_coords)):
        
        rotation_matrix = rotation_matrices[i]
        
        T_strain_rotated[:, :, i] = rotation_matrix @ T_strain[:, :, i] @ rotation_matrix.T
    
    ETT = T_strain_rotated[0, 0, :]
    ENN = T_strain_rotated[1, 1, :]
    ETN = T_strain_rotated[0, 1, :]
    ENT = T_strain_rotated[1, 0, :]
    
    x_profile, y_profile = profile_coords[:, 0], profile_coords[:, 1]
    distances = np.sqrt(np.diff(x_profile)**2 + np.diff(y_profile)**2)
    cumulative_distances = np.cumsum(distances)
    profile_line = np.insert(cumulative_distances, 0, 0)
    
    ax.plot(profile_line, ETT, ls="solid", marker=marker, label= r'$E_{tt}$')
    ax.plot(profile_line, ENN,  ls="solid", marker=marker, label= r'$E_{nn}$')
    ax.plot(profile_line, ETN,  ls="solid", marker=marker, label= r'$E_{tn}$')
    # ax.plot(profile_line, ENT,  ls="solid", marker=marker, label= r'$E_{nt}$')
    
    # ax.plot(profile_line, thetas, c='k', ls='dashed', marker='None')
    # ax.plot(profile_line[250], thetas[250], c='c', ls='None', marker='x')
    
    ax.set_xlabel("distance along line [mm]")
    ax.set_ylabel("Strain [$1/s$]")
    # ax.tick_params(axis='both', labelsize=fontsize)
    ax.grid(True)
    
    # ax.set_ylim(np.array([-10, 10]))
    
    min_value, min_index, max_value, max_index = find_min_max_indices(ETN)
    ax.plot(profile_line[min_index], ETN[min_index], c='r', marker="x")
    ax.plot(profile_line[max_index], ETN[max_index], c='b', marker="x")
    
    return EXX_profile, EYY_profile, EXY_EYX_div_2_profile, min_index, max_index

def plot_curved_strain_rate(fig, ax, profile_coords, label, cmin, cmax, color, num):
    
    X_values = X.flatten()
    Y_values = Y.flatten()
    
    EXX_values = EXX.flatten()
    EXY_values = EXY.flatten()
    EYX_values = EYX.flatten()
    EYY_values = EYY.flatten()
    EXY_EYX_div_2_values = EXY_EYX_div_2.flatten()
    
    EXX_profile = griddata((X_values, Y_values), EXX_values, profile_coords, method=interpolation_method)
    EXY_profile = griddata((X_values, Y_values), EXY_values, profile_coords, method=interpolation_method)
    EYX_profile = griddata((X_values, Y_values), EYX_values, profile_coords, method=interpolation_method)
    # EXY_EYX_div_2_profile = griddata((X_values, Y_values), (EXY_values + EYX_values)/2, profile_coords, method=interpolation_method)
    EXY_EYX_div_2_profile = griddata((X_values, Y_values), EXY_EYX_div_2_values, profile_coords, method=interpolation_method)
    EYY_profile = griddata((X_values, Y_values), EYY_values, profile_coords, method=interpolation_method)
    
    # Compute rotation matrices along the curved profile
    rotation_matrices, thetas = compute_rotation_matrices(profile_coords)
    
    ### Approach 2: Calculate Reynolds stresses on arbirtary line with matrix multiplication
    T_strain = np.array(((EXX_profile, EXY_profile), (EYX_profile, EYY_profile)))
    T_strain_rotated = np.zeros_like(T_strain)
    
    for i in range(len(profile_coords)):
        
        rotation_matrix = rotation_matrices[i]
        
        T_strain_rotated[:, :, i] = rotation_matrix @ T_strain[:, :, i] @ rotation_matrix.T
    
    ETT = T_strain_rotated[0, 0, :]
    ENN = T_strain_rotated[1, 1, :]
    ETN = T_strain_rotated[0, 1, :]
    ENT = T_strain_rotated[1, 0, :]
    
    x_profile, y_profile = profile_coords[:, 0], profile_coords[:, 1]
    distances = np.sqrt(np.diff(x_profile)**2 + np.diff(y_profile)**2)
    cumulative_distances = np.cumsum(distances)
    profile_line = np.insert(cumulative_distances, 0, 0)
    
    ax.plot(profile_line, ETT, ls="solid", marker=marker, label= r'$E_{tt}$')
    ax.plot(profile_line, ENN,  ls="solid", marker=marker, label= r'$E_{nn}$')
    ax.plot(profile_line, ETN,  ls="solid", marker=marker, label= r'$E_{tn}$')
    ax.plot(profile_line, ENT,  ls="solid", marker=marker, label= r'$E_{nt}$')
    
    ax.plot(profile_line, thetas, c='k', ls='dashed', marker='None')
    # ax.plot(profile_line[250], thetas[250], c='c', ls='None', marker='x')
    
    ax.set_xlabel("distance along line [mm]")
    ax.set_ylabel("Strain [$1/s$]")
    # ax.tick_params(axis='both', labelsize=fontsize)
    ax.grid(True)
    
    # ax.set_ylim(np.array([-10, 10]))
    
    return EXX_profile, EYY_profile, EXY_EYX_div_2_profile

def plot_curved_tke(fig, ax, profile_coords, label, cmin, cmax, color, num):

    X_values = X.flatten()
    Y_values = Y.flatten()
    TKE_values = TKE.flatten()
    
    TKE_profile = griddata((X_values, Y_values), TKE_values, profile_coords, method=interpolation_method)

    x_profile, y_profile = profile_coords[:, 0], profile_coords[:, 1]
    distances = np.sqrt(np.diff(x_profile)**2 + np.diff(y_profile)**2)
    cumulative_distances = np.cumsum(distances)
    profile_line = np.insert(cumulative_distances, 0, 0)
    
    ax.plot(profile_line, TKE_profile, ls='solid', marker=marker)
    
    
    ax.set_xlabel("distance along line [mm]")
    ax.set_ylabel("TKE [m$^2$s$^{-2}$]")
    # ax.tick_params(axis='both', labelsize=fontsize)
    # ax[1].set_ylabel("$R_{NN}$ [ms$^-1$]")
    
    # ax.set_xlim(np.array([0, 30]))
    # ax[1].set_ylim(np.array([cmin, cmax]))
    ax.grid(True)
    ax.axhline(y=0, color='k')
    
    min_value, min_index, max_value, max_index = find_min_max_indices(TKE_profile)
    
    return TKE_profile, min_index, max_index

def draw_walls(ax):

    # Liner
    pt1 = pt1_liner_mm
    pt2 = pt2_liner_mm
    pt3 = np.array([pt2_liner_mm[0] - 4, pt2_liner_mm[1]])
    pt4 = np.array([pt1_liner_mm[0] - 4, pt1_liner_mm[1]])
    ax.add_patch(Polygon([pt1, pt2, pt3, pt4], color="silver"))
    
    # Core flow plate 
    pt1 = pt1_core_left_mm
    pt2 = np.array([pt1_core_left_mm[0] + 4, pt1_core_left_mm[1]])
    pt3 = np.array([pt1_core_left_mm[0] + 4, pt2_core_left_mm[1]])
    pt4 = pt2_core_left_mm
    ax.add_patch(Polygon([pt1, pt2, pt3, pt4], color="silver"))
    
    # # Back plate
    # pt1 = coordinate_grid[:, pt1_core_right[0], pt1_core_right[1]]
    # pt2 = coordinate_grid[:, -1, 0]
    # pt3 = coordinate_grid[:, -1, -1]
    # pt4 = coordinate_grid[:, pt2_core_right[0], pt2_core_right[1]]
    # ax.add_patch(Polygon([pt1, pt2, pt3, pt4], color="silver"))
    
    num = 1000
    theta1 = np.linspace(np.pi/4, -3*np.pi/4, num)
    
    y_theta1 = cy_mm + radius_mm*np.sin(theta1)
    array = y_theta1
    value = y_raw[pt2_core_left[1], 0]
    idx1 = find_nearest(array, value)
    
    theta2 = np.linspace(theta1[idx1], -np.pi, num)
    
    x_theta2 = cx_mm + radius_mm*np.cos(theta2)
    y_theta2 = cy_mm + radius_mm*np.sin(theta2)
    
    xy_zip  = zip(x_theta2, y_theta2)
    xy_unzip_list = list(xy_zip)
    
    pt1 = x_theta2[-1], coordinate_grid[:, 0, 0][1]
    pt2 = x_theta2[-1]-7.5, coordinate_grid[:, 0, 0][1] #coordinate_grid[:, 0, 0]
    pt3 = x_theta2[-1]-7.5, coordinate_grid[:, -1, -1][1] 
    pt4 = np.array([pt1_core_left_mm[0] + 4, y_raw[-1, -1]])
    pt5 = np.array([pt1_core_left_mm[0] + 4, pt2_core_left_mm[1]])
    
    xy_unzip_list.extend([tuple(pt1), tuple(pt2), tuple(pt3), tuple(pt4), tuple(pt5)])
    
    ax.add_patch(Polygon(xy_unzip_list, color="silver"))
    
    return cx_mm, cy_mm, radius_mm
    
#%% AUXILIARY FUNCTIONS
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx #, array[idx]

def line_line_intersection(p1_start, p1_end, p2_start, p2_end):
    
    p = p1_start
    r = (p1_end-p1_start)
    
    q = p2_start
    s = (p2_end-p2_start)
    
    t = np.cross(q - p,s)/(np.cross(r,s))
    
    # This is the intersection point
    intersection = p + t*r
    return intersection
    
def circle_line_segment_intersection(circle_center, circle_radius, pt1, pt2, full_line=False, tangent_tol=1e-9):
    """ Find the points at which a circle intersects a line-segment.  This can happen at 0, 1, or 2 points.

    :param circle_center: The (x, y) location of the circle center
    :param circle_radius: The radius of the circle
    :param pt1: The (x, y) location of the first point of the segment
    :param pt2: The (x, y) location of the second point of the segment
    :param full_line: True to find intersections along full line - not just in the segment.  False will just return intersections within the segment.
    :param tangent_tol: Numerical tolerance at which we decide the intersections are close enough to consider it a tangent
    :return Sequence[Tuple[float, float]]: A list of length 0, 1, or 2, where each element is a point at which the circle intercepts a line segment.

    Note: We follow: http://mathworld.wolfram.com/Circle-LineIntersection.html
    """

    (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, circle_center
    (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
    dx, dy = (x2 - x1), (y2 - y1)
    dr = (dx ** 2 + dy ** 2)**.5
    big_d = x1 * y2 - x2 * y1
    discriminant = circle_radius ** 2 * dr ** 2 - big_d ** 2

    if discriminant < 0:  # No intersection between circle and line
        return []
    else:  # There may be 0, 1, or 2 intersections with the segment
        intersections = [
            (cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant**.5) / dr ** 2,
             cy + (-big_d * dx + sign * abs(dy) * discriminant**.5) / dr ** 2)
            for sign in ((1, -1) if dy < 0 else (-1, 1))]  # This makes sure the order along the segment is correct
        if not full_line:  # If only considering the segment, filter out intersections that do not fall within the segment
            fraction_along_segment = [(xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy for xi, yi in intersections]
            intersections = [pt for pt, frac in zip(intersections, fraction_along_segment) if 0 <= frac <= 1]
        if len(intersections) == 2 and abs(discriminant) <= tangent_tol:  # If line is tangent to circle, return just one point (as both intersections have same location)
            return [intersections[0]]
        else:
            return intersections

def contour_correction(x_left_raw, x_right_raw, y_bottom_raw, y_top_raw, window_size_x_raw, window_size_y_raw):
    
    with open(os.path.join('pickles', f'{record_name}_BfmOfAvg_wsize_{window_size}.pkl'), 'rb') as f:
            contour = pickle.load(f)
            
    segmented_contour = contour
    
    segmented_contour_x = segmented_contour[:, 0, 0]
    segmented_contour_y = segmented_contour[:, 0, 1]
    
    # x and y coordinates of the discretized (segmented) flame front 
    contour_x_corrected = segmented_contour_x*window_size_x_raw + x_left_raw
    contour_y_corrected = segmented_contour_y*window_size_y_raw + y_top_raw
    
    # Non-dimensionalize coordinates by pipe diameter
    contour_x_corrected /= 1 #D_in
    contour_y_corrected /= 1 #D_in
    
    contour_x_corrected_array = np.array(contour_x_corrected)
    contour_y_corrected_array = np.array(contour_y_corrected)
    
    # Combine the x and y coordinates into a single array of shape (n_coords, 2)
    contour_corrected_coords = np.array([contour_x_corrected_array, contour_y_corrected_array]).T

    # Create a new array of shape (n_coords, 1, 2) and assign the coordinates to it
    contour_corrected = np.zeros((len(contour_x_corrected_array), 1, 2))
    contour_corrected[:, 0, :] = contour_corrected_coords
    
    return contour_corrected  

def compute_rotation_matrices(profile_coords):
    
    tangents = np.diff(profile_coords, axis=0)
    tangents = np.vstack([tangents, tangents[-1]])  # Repeat the last tangent for the last point
    
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / norms  # Normalize tangents
    
    thetas = np.arctan2(-tangents[:, 1], tangents[:, 0])
    
    rotation_matrices = [np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) for theta in thetas]
    
    return rotation_matrices, thetas

def find_min_max_indices(data_array):

  min_value = np.min(data_array)
  min_index = np.argmin(data_array)

  max_value = np.max(data_array)
  max_index = np.argmax(data_array)

  return min_value, min_index, max_value, max_index

def find_highest_negative_index(arr):
    
    highest_row_index = -1
    corresponding_col_index = -1

    for row_idx in range(len(arr) - 1, -1, -1):
        for col_idx in range(len(arr[row_idx])):
            if arr[row_idx][col_idx] < 0:
                highest_row_index = row_idx
                corresponding_col_index = col_idx
                return highest_row_index, corresponding_col_index  # Return as soon as we find the highest negative value
    return None  # Return None if no negative value is found

def find_first_positive_index(arr):
    for row_idx in range(len(arr) - 1, -1, -1):
        for col_idx in range(len(arr[row_idx])):
            if arr[row_idx][col_idx] > 0:
                return row_idx, col_idx  # Return as soon as we find the first positive value
    return None  # Return None if no positive value is found



def find_values_in_range(A, Y, left_bound, right_bound, lower_bound, upper_bound):
    # Create a boolean mask for the condition -20 < Y < 0
    mask = (X > left_bound) & (X < right_bound) & (Y > lower_bound) & (Y < upper_bound) 
    
    # Create an output array initialized with NaNs (or any other placeholder)
    output = np.full(A.shape, np.nan)
    
    # Apply the mask to A and assign the values to the output array
    output[mask] = A[mask]
    
    # Return the output array and the mask
    return output

def find_index_closest_to_zero_derivative(arr):
    # Compute the numerical derivative
    derivative = np.diff(arr[:, 1])
    
    # Find the index where the derivative is closest to zero
    index_closest_to_zero = np.argmin(np.abs(derivative))
    
    return index_closest_to_zero

def plot_perpendicular_line(point1, point2, length):
    """
    Plot a line segment between two points and a perpendicular line through the midpoint.

    Parameters:
    point1 (array-like): The first point [x1, y1].
    point2 (array-like): The second point [x2, y2].
    length (float): The length of the perpendicular line segment (default is 2).

    Returns:
    perp_point1 (numpy array): The first endpoint of the perpendicular line segment.
    perp_point2 (numpy array): The second endpoint of the perpendicular line segment.
    """

    # Calculate the midpoint and direction vector
    midpoint = (point1 + point2) / 2
    direction = point2 - point1

    # Handle vertical line case (avoid division by zero)
    if np.allclose(direction[0], 0):
        perp_direction = np.array([-1, 0])  # Perpendicular line with unit vector along x-axis
    else:
        # Calculate normalized perpendicular direction vector
        perp_direction = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)

    # Calculate perpendicular line segment endpoints using midpoint and scaled direction vector
    perp_point1 = midpoint + length * perp_direction / 2
    perp_point2 = midpoint - length * perp_direction / 2
    
    return perp_point1, perp_point2
    
#%% MAIN

if __name__ == "__main__":

    #%%% Read info of the raw images
    x_raw, y_raw, x_left_raw, x_right_raw, y_bottom_raw, y_top_raw, window_size_x_raw, window_size_y_raw = read_xy_dimensions(calibration_csv_file)
    
    # Invert the y-axis
    y_raw = y_raw[::-1]
    # xv, yv = np.meshgrid(x, y, indexing='ij')
    coordinate_grid = np.array([x_raw, y_raw])

    #%%% Normalize data?
    normalized = False
    
    marker = 'o'
    #%%% Read and save average velocity data
    # File name and scaling parameters from headers of file
    csv_file = 'B0001.csv'
    piv_avgV_file = os.path.join(piv_avgV_dir, csv_file)
    
    n_windows_x, n_windows_y, X, Y, AvgVx, AvgVy, AvgAbsV, RXY, RXX, RYY, TKE, EXX, EXY, EYX, EYY, EXY_EYX_div_2 = read_flow_data(piv_avgV_file, normalized) 
    
    contour_correction = contour_correction(x_left_raw, x_right_raw, y_bottom_raw, y_top_raw, window_size_x_raw, window_size_y_raw)
    
    contour_correction2 = contour_correction[:, 0, :]
    contour_correction2 = contour_correction2[::-1]
    contour_correction2 = contour_correction2[(contour_correction2[:, 0] > 0) & (contour_correction2[:, 1] < 15)]
    
    # contour_correction2[:, 1] -= -3
    
    #%%% Read and save transient velocity data
    # n_images = 2500
    # U_transient = np.zeros([n_windows_y, n_windows_x, n_images])
    # V_transient = np.zeros([n_windows_y, n_windows_x, n_images])
    
    # for image_nr in progressbar.progressbar(range(1, n_images + 1)):
        
    #     n_windows_x, n_windows_y, X, Y, U, V, velocity_abs = read_flow_data(piv_transV_dir, image_nr, normalized)
        
    #     U_transient[:,:,image_nr-1] = U
    #     V_transient[:,:,image_nr-1] = V
    
    #%%% Read and save vector statistics    
    # EXX, EYY, EXY, EYX
    scalars =  [AvgVx, 
                AvgVy, 
                AvgAbsV, 
                RXY, 
                RXX, 
                RYY, 
                TKE, 
                EXX, 
                EXY, 
                EYX, 
                EYY]
    
    if normalized:
        scalar_labels = [
                         "$V_{x}/U_{b}$", 
                         "$V_{y}/U_{b}$", 
                         "$|V|/U_{b}$",
                         # "$AKE/U^{2}_{b}$", "$\sigma_{V_{x}}/U_{b}$", "$\sigma_{V_{y}}/U_{b}$", "$\sigma_{|V|}/U_{b}$", 
                         "$R_{XY}/U^{2}_{b}$",
                         "$R_{XX}/U^{2}_{b}$", 
                         "$R_{YY}/U^{2}_{b}$", 
                         "$TKE/U^{2}_{b}$",
                         "$E_{XX}$",
                         "$E_{XY}$",
                         "$E_{YX}$",
                         "$E_{YY}$",
                         # "$TSS_{max}/U^{2}_{b}$"
                         ]
    else:
        scalar_labels = [
                         "$V_{x}$ [ms$^{-1}$]", 
                         "$V_{y}$ [ms$^{-1}$]", 
                         "$|V|$ [ms$^{-1}$]",
                         # "$AKE$ [m$^{2}$s$^{-2}$]", "$\sigma_{V_{x}}$ [ms$^{-1}$]", "$\sigma_{V_{y}}$ [ms$^{-1}$]", "$\sigma_{|V|}$ [ms$^{-1}$]", 
                         "$R_{XY}$ [m$^{2}$s$^{-2}$]",
                         "$R_{XY}$ [m$^{2}$s$^{-2}$]", 
                         "$R_{YY}$ [m$^{2}$s$^{-2}$]", 
                         "$TKE$ [m$^{2}$s$^{-2}$]", 
                         "$E_{XX}$ [$1/s$]",
                         "$E_{XY}$ [$1/s$]", 
                         "$E_{YX}$ [$1/s$]",
                         "$E_{YY}$ [$1/s$]", 
                         # "$TSS_{max}$ [m$^{2}$s$^{-2}$]"
                         ]
    
    scalar_titles = [
                     "Average velocity in horizontal direction",
                     "Average velocity in vertical direction", 
                     "Average absolute velocity",
                     # "Average Kinetic Energy", "Standard deviation of $V_{x}$", "Standard deviation of $V_{y}$", "Standard deviation of $|V|$", 
                     "Reynolds Normal Stress $R_{XX}$",
                     "Reynolds Shear Stress $R_{XY}$",
                     "Reynolds Normal Stress $R_{YY}$", 
                     "Turbulent Kinetic Energy", 
                     "Normal Strain $E_{XX}$",
                     "Shear Strain $E_{XY}$",
                     "Shear Strain $E_{YX}$",
                     "Normal Strain $E_{YY}$", 
                     # "$TSS_{max}$ $[m^{2}s^{-2}$]"
                     ]
    
    
    #%%% Plots
    plt.close("all")    
    
    #%%%% Plot velocity field [Figure 1]
    
    # Choose a scalar field
    scalar_index = 2
    scalar = scalars[scalar_index]
    label = scalar_labels[scalar_index]
    
    scalar_max = np.max(scalar)
    
    fig_scale = 1
    default_fig_dim = plt.rcParams["figure.figsize"]

    fig1, ax1 = plt.subplots(figsize=(fig_scale*default_fig_dim[0], fig_scale*default_fig_dim[1]), ncols=1, dpi=100)
    ax1.set_title(scalar_titles[scalar_index])
    
    cmin = 0
    cmax = 0
    
    while ((cmax%2 != 0) or (scalar_max > cmax)):
        cmax += 1
        # print(cmax)
    
    if normalized:
        cmax = 1.5
    
    #%%%%% Plot scalar field
    plot_field(fig1, ax1, X, Y, scalar, label, cmin, cmax)
    ax_x_lim = ax1.get_xlim()
    ax_y_lim = ax1.get_ylim()
    
    #%%%%% Plot vector field
    plot_vector_field(fig1, ax1, X, Y, AvgVx, AvgVy)
    
    contour_x = contour_correction2[:,0]
    contour_y = contour_correction2[:,1]
    ax1.plot(contour_x, contour_y, 'm', lw=1, marker='o')
    # ax1.plot(contour_x[250], contour_y[250], 'c', marker='x', ls='None')
    
    highest_row_index, corresponding_col_index = find_highest_negative_index(AvgVy)
    # print(highest_row_index, corresponding_col_index)
    ax1.plot(X[0, corresponding_col_index], Y[highest_row_index, 0], 'c', marker='x', ls='None')
    
    filtered_AvgVy = find_values_in_range(AvgVy, Y, left_bound=0, right_bound=10, lower_bound=-20, upper_bound=-10)
    row_idx, col_idx = find_first_positive_index(filtered_AvgVy)
    # print(row_idx, col_idx)
    ax1.plot(X[0, col_idx], Y[row_idx, 0], 'r', marker='x', ls='None')
    
    index_closest_to_zero = find_index_closest_to_zero_derivative(contour_correction2)
    
    contour_correction2_b = []
    contour_correction2_u = [] 
    
    
    for i in range(len(contour_correction2) - 1):
        
        perp_point1, perp_point2 = plot_perpendicular_line(contour_correction2[i], contour_correction2[i + 1], length=5)
        
        contour_correction2_b.append(perp_point1)
        contour_correction2_u.append(perp_point2)
        
    contour_correction2_b = np.array(contour_correction2_b)
    contour_correction2_u = np.array(contour_correction2_u)
    
    ax1.plot(contour_correction2_b[:, 0], contour_correction2_b[:, 1], 'r', lw=1, marker='o')
    ax1.plot(contour_correction2_u[:, 0], contour_correction2_u[:, 1], 'b', lw=1, marker='o')
    
    #%%%%% Plot streamlines
    # streamlines = plot_streamlines(fig1, ax1, X, Y, AvgVx, AvgVy)
    
    #%%%%% Detect walls from precording image
    pt1_liner, pt2_liner, pt1_core_left, pt2_core_left, cx, cy, radius = wall_detection(calibration_tif_file, pre_record_correction_file)
    
    #%%%%% Convert detected wall coordinates from pixels to mm
    # Convert liner tip coordinate in pixels to coordinate in mm
    pt1_liner_mm = coordinate_grid[:, pt1_liner[1], pt1_liner[0]]
    pt2_liner_mm = coordinate_grid[:, pt2_liner[1], pt2_liner[0]]
    
    # Convert coordinates of left wall of core flow in pixels to coordinate in mm
    pt1_core_left_mm = coordinate_grid[:, pt1_core_left[1], pt1_core_left[0]]
    pt2_core_left_mm = coordinate_grid[:, pt2_core_left[1], pt2_core_left[0]]
    
    cx_mm = x_raw[0, 0] + cx*np.mean(np.diff(x_raw))
    cy_mm = y_raw[0, 0] + cy*np.mean(np.diff(y_raw, axis=0))
    radius_mm = radius*np.mean(np.diff(x_raw))
    
    #%%%%% Draw walls
    cx_mm, cy_mm, radius_mm = draw_walls(ax1)
    ax1.set_xlim(ax_x_lim)
    ax1.set_ylim(ax_y_lim)
    
    #%%%% Plot scalar field without walls [Figure 2]
    fig2, ax2 = plt.subplots(figsize=(fig_scale*default_fig_dim[0], fig_scale*default_fig_dim[1]), ncols=1, dpi=100)
    plot_field(fig2, ax2, X, Y, scalar, label, cmin, cmax)
    
    #%%%% [Figure 3]
    # fig3, ax3 = plt.subplots()
    # X_check, Y_check, I_check, XYI = plot_image(fig3, ax3, nx, ny)
    
    #%%%% [Figure 4, Figure 5, Figure 6]
    
    # Initialize figures
    fig4, ax4 = plt.subplots(figsize=(fig_scale*default_fig_dim[0], fig_scale*default_fig_dim[1]), ncols=1, dpi=100)
    fig5, ax5 = plt.subplots(figsize=(fig_scale*default_fig_dim[0], fig_scale*default_fig_dim[1]), ncols=1, dpi=100)
    fig6, ax6 = plt.subplots(figsize=(fig_scale*default_fig_dim[0], fig_scale*default_fig_dim[1]), ncols=1, dpi=100)
    
    #%%%%% Important coordinates for Figure 4, 5, 6
    
    # Top left coordinates of velocity data in mm
    X0, Y0 = X[0][0], Y[0][0]
    
    # Spacing between velocity vector in mm
    dx_piv, dy_piv  = np.diff(X[0,:])[0], np.diff(Y[:,0])[0]
    
    # Coordinates of the liner tip in PIV "pixels". This is needed for extracting the profiles
    coord0_mm = pt2_liner_mm
    
    #%%%%% Plot scalar field [Figure 4]
    plot_field(fig4, ax4, X, Y, scalar, label, cmin, cmax)
    # ax4[1].set_title()
    # fig4.suptitle("Velocity profiles", fontsize=30)
    
    # Draw walls in figure
    draw_walls(ax4)
    ax4.set_xlim(ax2.get_xlim())
    ax4.set_ylim(ax2.get_ylim())
    
    #%%%%% Plot scalar field [Figure 5]
    # ax5[0].imshow(scalar)
    # quantity_plot = ax5[0].pcolor(X, Y, scalar, cmap="viridis", rasterized=True)
    # quantity_plot.set_clim(cmin, cmax)
    
    #%%%%% Define cross-sections for extraction of profiles
    
    # Initiate profile lists
    profile_lines = []
    Vn_avg_profiles = []
    
    # Profile colors
    colors = tableau
    
    # Angle with respect to the horizon of cross-section for profile in degrees
    # thetas_deg = [80, 60, 30, 0, 0, 0, 0] 
    thetas_deg = [60, 30, 0, 0, 0, 0, 0] 
    
    thetas = np.radians(thetas_deg) # Conversion to radians
    
    vertical_locs = [0, 0, 0, 7.5, 15, 22.5, 30] # in mm
    
    # vertical_locs = [vertical_coord, vertical_coord, vertical_coord, 9, 18, 27] # in mm
    
    profile_ids = list(range(len(thetas_deg)))
    
    colors_iter = iter(colors)
    
    for theta, vertical_loc in zip(thetas, vertical_locs):
        
        # Build rotation matrix
        c, s = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array(((c, -s), (s, c)))
        
        if vertical_loc != 0:
            
            # Find intersection of cross-section line with left wall of the core flow
            coord0_mm = pt2_liner_mm + np.array([0, vertical_loc])
            # p1_end = coord0_mm + np.array([100, vertical_loc])
            
            p2_start = pt1_core_left_mm
            p2_end = pt2_core_left_mm
            
            coord1_mm = line_line_intersection(coord0_mm, coord0_mm + np.array([1000, 0]), p2_start, p2_end)
            
        else:
            
            coord0_mm = pt2_liner_mm
            
            # Find intersection of cross-section line with dome wall
            circle_center = (cx_mm, cy_mm)
            circle_radius = radius_mm
            pt1 = pt2_liner_mm
            pt2 = pt2_liner_mm + 100*np.array([np.cos(theta), -np.sin(theta)]) 
            intersections = circle_line_segment_intersection(circle_center, circle_radius, pt1, pt2)
            coord1_mm = intersections[0]
            
        # Pick next color for plotting profile
        color = next(colors_iter)
        
        # Number of points of arbitrary line of profile
        x0_mm, y0_mm = coord0_mm
        x1_mm, y1_mm = coord1_mm
        arbitrary_line_length = np.sqrt((x1_mm - x0_mm)**2 + (y1_mm - y0_mm)**2)
        num = int(arbitrary_line_length/dx_piv)
        
        #%%%%% Plot normal velocity profiles with dimensions for cross-sections depending on theta [Figure 4]
        # arbitrary_line, Vx_avg_profile, Vy_avg_profile, Vt_avg_profile, Vn_avg_profile = plot_profile(fig4, ax4, rotation_matrix, coord0_mm, coord1_mm, AvgVx, AvgVy, label, cmin, cmax, color, num)
        profile_coords, profile_line = plot_profile_in_field(fig4, ax4, coord0_mm, coord1_mm, color, num)
        Vx_avg_profile, Vy_avg_profile, Vt_avg_profile, Vn_avg_profile = plot_profile(fig5, ax5, rotation_matrix, profile_coords, profile_line, AvgVx, AvgVy, label, cmin, cmax, color, num)
        
        #%%%%% Plot velocity profiles without dimensions [Figure 5]
        # plot_profile_nondim(fig5, ax5, rotation_matrix, coord0_mm, coord1_mm, AvgVx, AvgVy, label, cmin, cmax, color, num, order)
        
        #%%%%% Plot Reynolds normal stresses of selected cross-sections [Figure 6]
        # Rxx_profile, Ryy_profile, Rxy_profile, min_index, max_index = plot_reynolds_stress(fig6, ax6, rotation_matrix, theta, profile_coords, profile_line, label, cmin, cmax, color, num)
        
        #%%%%% Plot strains of selected cross-sections [Figure 6]
        Exx_profile, Eyy_profile, Exy_profile, min_index, max_index = plot_strain_rate(fig6, ax6, rotation_matrix, theta, profile_coords, profile_line, label, cmin, cmax, color, num)
        
        #%%%%% Plot strains of selected cross-sections [Figure 6]
        # TKE_profile, min_index, max_index = plot_tke(fig6, ax6, profile_coords, profile_line, label, cmin, cmax, color, num)
        
        # ax1.plot(profile_coords[min_index, 0], profile_coords[min_index, 1], c='r', marker="x")
        # ax1.plot(profile_coords[max_index, 0], profile_coords[max_index, 1], c='b', marker="x")
        
        # ax4.plot(profile_coords[min_index, 0], profile_coords[min_index, 1], c='r', marker="x")
        # ax4.plot(profile_coords[max_index, 0], profile_coords[max_index, 1], c='b', marker="x")
        
        ax4.plot(contour_x, contour_y, 'm', lw=2)
        
        #%%%%% Write data to lists
        profile_lines.append(profile_line)
        Vn_avg_profiles.append(Vn_avg_profile)
    
    figY, axY = plt.subplots(figsize=(fig_scale*default_fig_dim[0], fig_scale*default_fig_dim[1]), ncols=1, dpi=100)
    
    profile_coords = contour_correction2_u
    
    # Vx_avg_profile, Vy_avg_profile, Vt_avg_profile, Vn_avg_profile = plot_curved_profile(figY, axY, profile_coords, AvgVx, AvgVy, label, cmin, cmax, color, num)
    # Exx_profile, Eyy_profile, Exy_profile = plot_curved_strain_rate(figY, axY, profile_coords, label, cmin, cmax, color, num)
    Exx_profile, Eyy_profile, Exy_profile, min_index, max_index = plot_curved_strain_rate_symmetric(figY, axY, profile_coords, label, cmin, cmax, color, num)
    # Rxx_profile, Ryy_profile, Rxy_profile = plot_curved_reynolds_stress(figY, axY, profile_coords, label, cmin, cmax, color, num)
    # TKE_profile, min_index, max_index = plot_curved_tke(figY, axY, profile_coords, label, cmin, cmax, color, num)
    
    ax1.plot(profile_coords[min_index, 0], profile_coords[min_index, 1], c='r', marker="x")
    ax1.plot(profile_coords[max_index, 0], profile_coords[max_index, 1], c='b', marker="x")
    
    for profile_id in profile_ids:
        
        theta = thetas_deg[profile_id]
        vertical_loc = vertical_locs[profile_id]
        
        f  = open(f"V_n_profile_theta{theta}_vertical_location{vertical_loc}.csv", "w")
        f.write("distance [mm]" + "," + "V_n [m/s]\n")
        for i,j in zip(profile_lines[profile_id], Vn_avg_profiles[profile_id]):
            f.write(str(i) + "," + str(j) + "\n")
        f.close()

    #%%%% Plot scalar field [Figure 7]
    figX, axX = plt.subplots(figsize=(fig_scale*default_fig_dim[0], fig_scale*default_fig_dim[1]), ncols=1, dpi=100)
    
    # # Choose a scalar field
    # scalar_index = 7
    # scalar = scalars[scalar_index]
    # label = vector_stats_labels[scalar_index]
    # axX.set_title(vector_stats_titles[scalar_index])
    
    # scalar_max = np.max(scalar)
    
    cmin = 0
    cmax = 0
    
    while ((cmax%2 != 0) or (scalar_max > cmax)):
        cmax += 1
    
    if normalized:
        cmax = 0.1
        
    plot_field(figX, axX, X, Y, scalar, label, cmin, cmax)
    
    #%%%%% Draw walls
    draw_walls(axX)
    axX.set_xlim(ax2.get_xlim())
    axX.set_ylim(ax2.get_ylim())
    
    #%%%% Legends
    # figX.legend()
    figY.legend()                      
    
    #%%%% Tighten layouts                      
    fig1.tight_layout()
    fig2.tight_layout()
    # fig3.tight_layout()
    fig4.tight_layout()
    # fig5.tight_layout()
    fig6.tight_layout()
    figX.tight_layout()
    figY.tight_layout()
    
    
    #%%%% Save figures
    
    # fig1.savefig("figures/figure_1_avg_abs_V.svg", format="svg", dpi=1200, bbox_inches="tight")
    # fig1.savefig("figures/figure_1_avg_abs_V_vectors.svg", format="svg", dpi=1200, bbox_inches="tight")
    # fig1.savefig("figures/figure_1_avg_abs_V_streamlines.svg", format="svg", dpi=1200, bbox_inches="tight")
    
    # fig4.savefig("figures/figure_4_velocity_profiles.svg", format="svg", dpi=1200, bbox_inches="tight")






















