# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 16:11:23 2022

@author: laaltenburg
"""
#%% Import packages
import os
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
    
    
#%% Start
plt.close("all")

#%% Figure settings

# Color maps
tableau = cm.tab10.colors

#%% Main functions
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
    # window_size_x_raw, window_size_y_raw = np.mean(np.diff(x_raw_array)), -np.mean(np.diff(y_raw_array))
    
    # Parameters for correcting contours from pixel coordinates to physical coordinates
    x_left_raw = x_raw_array[0]
    x_right_raw = x_raw_array[-1]
    y_bottom_raw = y_raw_array[0]
    y_top_raw = y_raw_array[-1]
    
    # print(n_windows_x_raw, n_windows_y_raw)
    # print(y_bottom_raw, y_top_raw)
    
    return x_raw, y_raw


def read_velocity_data(file, normalized):
    
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
    
    pivot_U = pd.pivot_table(df_piv, values=headers[2], index=headers[1], columns=headers[0])
    pivot_V = pd.pivot_table(df_piv, values=headers[3], index=headers[1], columns=headers[0])
    pivot_absV = pd.pivot_table(df_piv, values=headers[4], index=headers[1], columns=headers[0])
    
    n_windows_x = pivot_absV.columns.size
    n_windows_y = pivot_absV.index.size
    
    X_array = pivot_absV.columns
    Y_array = pivot_absV.index
    X, Y = np.meshgrid(X_array, Y_array)
    
    
    # XYUV = np.genfromtxt(XYUV_file, delimiter=",")
    # df = pd.read_csv(XYUV_file)
    
    # X = XYUV[:,0].reshape(n_windows_y, n_windows_x)
    # Y = XYUV[:,1].reshape(n_windows_y, n_windows_x)
    # U = XYUV[:,2].reshape(n_windows_y, n_windows_x) # *-1 because -> inverse x-axis
    # V = XYUV[:,3].reshape(n_windows_y, n_windows_x)
    # velocity_abs = np.sqrt(U**2 + V**2)
    
    return n_windows_x, n_windows_y, X, Y, pivot_U, pivot_V, pivot_absV

def read_vector_statistics(data_dir, n_windows_x, n_windows_y, normalized):
    
    # Set if plot is normalized or non-dimensionalized
    if normalized:
        u_bulk = u_bulk_measured
    else:
        u_bulk = 1
    
    print(f'Set normalized: {normalized}')
    print(f'Set u_bulk_measured: {u_bulk_measured}')
    # AvgVx, AvgVy, abs(AvgV), AvgKineticE, StdevVx, StdevVy, abs(StdevV), TurbKineticE, ReynoldsStressXY, ReynoldsStressXX, ReynoldsStressYY, TSSmax2D
    
    vector_stats_file_nrs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] 
    vector_stats = np.zeros([n_windows_y, n_windows_x, len(vector_stats_file_nrs)])
    
    for file_nr in vector_stats_file_nrs: 
        
        # File name and scaling parameters from headers of file
        file = f'B{file_nr:04d}.csv'
        vector_stat_file = os.path.join(data_dir, file)
        
        df_stat = pd.read_csv(vector_stat_file)
        
        # Get the column headers of the file
        headers = df_stat.columns
        
        vector_stat = pd.pivot_table(df_stat, values=headers[2], index=headers[1], columns=headers[0])
        
        file_index_shift = vector_stats_file_nrs[0]
        if file_nr in [1, 2, 3, 5, 6, 7]:
            vector_stats[:,:,file_nr-file_index_shift] = vector_stat/u_bulk
        
        if file_nr in [4, 8, 9, 10, 11, 12]:
            vector_stats[:,:,file_nr-file_index_shift] = vector_stat/(u_bulk**2)
    
    AvgVx, AvgVy, AvgAbsV, AvgKineticE = vector_stats[:,:,0], vector_stats[:,:,1], vector_stats[:,:,2], vector_stats[:,:,3]
    StdevVx, StdevVy, StdevAbsV, TurbKineticE = vector_stats[:,:,4], vector_stats[:,:,5], vector_stats[:,:,6], vector_stats[:,:,7]
    RXY, RXX, RYY, TSSmax2D = vector_stats[:,:,8], vector_stats[:,:,9], vector_stats[:,:,10], vector_stats[:,:,11]
    
    return AvgVx, AvgVy, AvgAbsV, AvgKineticE, StdevVx, StdevVy, StdevAbsV, TurbKineticE, RXY, RXX, RYY, TSSmax2D


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
    scale = 2
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
    Vx = AvgVx1.values
    Vy = AvgVy1.values
    
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
    
    Vxi = griddata((X_values, Y_values), Vx.flatten(), (Xi, Yi), method='linear')
    Vyi = griddata((X_values, Y_values), Vy.flatten(), (Xi, Yi), method='linear')
    
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
    # profile_line = profile_line[1:-1]
    # profile_coords = profile_coords[1:-1]
    
    ax.plot(profile_coords[:, 0], profile_coords[:, 1], c=color, ls='solid')
    
    return profile_coords, profile_line

def plot_profile_dim(fig, ax, rotation_matrix, profile_coords, profile_line, quantity_x, quantity_y, label, cmin, cmax, color, num):

    X_values = X.flatten()
    Y_values = Y.flatten()
    quantity_x_values = quantity_x.flatten()
    quantity_y_values = quantity_y.flatten()
    
    quantity_x_profile = griddata((X_values, Y_values), quantity_x_values, profile_coords, method='nearest')
    quantity_y_profile = griddata((X_values, Y_values), quantity_y_values, profile_coords, method='nearest')
    
    # quantity_x_profile[0] = 0
    # quantity_y_profile[0] = 0
    # quantity_x_profile[-1] = 0
    # quantity_y_profile[-1] = 0
    
    # quantity_tangent_profile, quantity_normal_profile  = np.dot(rotation_matrix, np.array([quantity_x_profile, quantity_y_profile]))
    quantity_tangent_profile, quantity_normal_profile  = np.dot(rotation_matrix, np.array([quantity_x_profile, quantity_y_profile]))
    
    quantity_tangent_profile = quantity_x_profile*np.cos(theta) - quantity_y_profile*np.sin(theta)
    quantity_normal_profile = quantity_x_profile*np.sin(theta) + quantity_y_profile*np.cos(theta)
    
    ax.plot(profile_line, quantity_normal_profile, c=color, ls='solid', marker='o')
    
    ax.set_xlabel("distance along line [mm]", fontsize=fontsize)
    ax.set_ylabel("$V_{n}$ [ms$^-1$]", fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    # ax[1].set_ylabel("$R_{NN}$ [ms$^-1$]")
    
    
    ax.set_xlim(np.array([0, 30]))
    # ax[1].set_ylim(np.array([cmin, cmax]))
    ax.grid()
    ax.axhline(y=0, color='k')
    
    u_bulk_2d = trapezoid(quantity_normal_profile, profile_line)/10
    print(trapezoid(quantity_normal_profile, profile_line))
    print("u_bulk_measured_2d= {0:.1f} m/s".format(u_bulk_2d))
    
    return quantity_x_profile, quantity_y_profile, quantity_tangent_profile, quantity_normal_profile


def plot_profile_nondim(fig, ax, rotation_matrix, coord0_mm, coord1_mm, quantity_x, quantity_y, label, cmin, cmax, color, num, order):
    
    x0, y0 = (coord0_mm - np.array([X0, Y0]))/np.array([dx_piv, dy_piv])
    x1, y1 = (coord1_mm - np.array([X0, Y0]))/np.array([dx_piv, dy_piv])
    
    x_profile, y_profile = np.linspace(x0, x1, num), np.linspace(y0, y1, num)
    ax[0].plot(x_profile, y_profile, c=color, ls="--", marker="None")
    
    ax[0].set_xlabel('interrogation windows')
    ax[0].set_ylabel('interrogation windows')
    
    # Extract the values along the line, using first, second or third order interpolation
    quantity_x_profile = scipy.ndimage.map_coordinates(quantity_x, np.vstack((y_profile, x_profile)), order=order)
    quantity_y_profile = scipy.ndimage.map_coordinates(quantity_y, np.vstack((y_profile, x_profile)), order=order)
    
    # quantity_tangent_profile, quantity_normal_profile  = np.dot(rotation_matrix, np.array([quantity_x_profile, quantity_y_profile]))
    
    quantity_tangent_profile = quantity_x_profile*np.cos(theta) - quantity_y_profile*np.sin(theta)
    quantity_normal_profile = quantity_x_profile*np.sin(theta) + quantity_y_profile*np.cos(theta)
    
    ax[1].plot(quantity_normal_profile, c=color, ls='solid', marker='o')
    
    ax[1].set_xlabel('-')
    ax[1].set_ylabel("$V_{n}$ [ms$^-1$]")
    
    ax[1].set_xlim(np.array([0, len(quantity_x_profile)]))
    # ax[1].set_ylim(np.array([cmin, cmax]))
    ax[1].grid()
    ax[1].axhline(y=0, color='k')
    
    
def plot_reynolds_stress1_dim(fig, ax, rotation_matrix, theta, profile_coords, profile_line, label, cmin, cmax, color, num):
    
    # c, s = np.cos(theta), np.sin(theta)
    # rotation_matrix = np.array(((c, -s), (s, c)))
    # rotation_matrix_T = np.transpose(rotation_matrix)
    
    # x0_mm, y0_mm = coord0_mm
    # x1_mm, y1_mm = coord1_mm
    # x0, y0 = (coord0_mm - np.array([X0, Y0]))/np.array([dx_piv, dy_piv])
    # x1, y1 = (coord1_mm - np.array([X0, Y0]))/np.array([dx_piv, dy_piv])
    
    # x_profile, y_profile = np.linspace(x0, x1, num), np.linspace(y0, y1, num)
    # x_profile, y_profile = np.linspace(x0_mm, x1_mm, num), np.linspace(y0_mm, y1_mm, num)
    # profile_coords = np.column_stack((x_profile, y_profile))
    
    # Extract the values along the line, using first, second or third order interpolation
    # profile_line_length = np.sqrt((x1_mm - x0_mm)**2 + (y1_mm - y0_mm)**2)
    # profile_line = np.linspace(0, np.sqrt((x1_mm - x0_mm)**2 + (y1_mm - y0_mm)**2), int(profile_line_length/dx_piv))
    
    X_values = X.flatten()
    Y_values = Y.flatten()
    
    RXX_values = RXX.flatten()
    RYY_values = RYY.flatten()
    RXY_values = RXY.flatten()
    
    RXX_profile = griddata((X_values, Y_values), RXX_values, profile_coords, method='nearest')
    RYY_profile = griddata((X_values, Y_values), RYY_values, profile_coords, method='nearest')
    RXY_profile = griddata((X_values, Y_values), RXY_values, profile_coords, method='nearest')
    
    # RXX_profile = scipy.ndimage.map_coordinates(RXX, np.vstack((y_profile, x_profile)), order=order)
    # RYY_profile = scipy.ndimage.map_coordinates(RYY, np.vstack((y_profile, x_profile)), order=order)
    # RXY_profile = scipy.ndimage.map_coordinates(RXY, np.vstack((y_profile, x_profile)), order=order)
    
    ### Approach 1: Calculate Reynolds stresses on arbirtary line "manually"
    RTT = RXX_profile*(np.cos(theta))**2 - 2*RXY_profile*np.cos(theta)*np.sin(theta) + RYY_profile*(np.sin(theta))**2
    RTN = (RXX_profile - RYY_profile)*np.cos(theta)*np.sin(theta) + RXY_profile*((np.cos(theta))**2 - (np.sin(theta))**2)
    RNN = RXX_profile*(np.sin(theta))**2 + 2*RXY_profile*np.cos(theta)*np.sin(theta) + RYY_profile*(np.cos(theta))**2

    ### Approach 2: Calculate Reynolds stresses on arbirtary line using rotation matrix [CORRECT RESULT WITH "INCORRECT" CODE]
    # R_stress_tensor = np.array(((Rxx_profile, Rxy_profile), (Rxy_profile, Ryy_profile)))
    # R_stress_tensor_rotated = rotation_matrix.dot(rotation_matrix.dot(R_stress_tensor))
    
    # Rtt = R_stress_tensor_rotated[0,0,:]
    # Rnn = R_stress_tensor_rotated[1,1,:]
    # Rtn = R_stress_tensor_rotated[0,1,:]
    
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
    ax.grid()
    
    
    # ax.set_ylim(np.array([cmin, cmax]))
    
    return RXX_profile, RYY_profile, RXY_profile


def plot_reynolds_stress2_dim(fig, ax, rotation_matrix, coord0_mm, coord1_mm, n_images, U_transient, V_transient, Vt_avg_profile, Vn_avg_profile, label, cmin, cmax, color, num, order): 
    
    x0_mm, y0_mm = coord0_mm
    x1_mm, y1_mm = coord1_mm
    x0, y0 = (coord0_mm - np.array([X0, Y0]))/np.array([dx_piv, dy_piv])
    x1, y1 = (coord1_mm - np.array([X0, Y0]))/np.array([dx_piv, dy_piv])
    
    x_profile, y_profile = np.linspace(x0, x1, num), np.linspace(y0, y1, num)
    
    # Extract the values along the line, using first, second or third order interpolation
    arbitrary_line = np.linspace(0, np.sqrt((x1_mm - x0_mm)**2 + (y1_mm - y0_mm)**2), num)
    
    Rtt, Rnn, Rtn = (np.zeros(num) for i in range(3))
    
    for image_nr in progressbar.progressbar(range(1, n_images + 1)):
        
        U, V = U_transient[:,:,image_nr-1], V_transient[:,:,image_nr-1]

        Vx_profile = scipy.ndimage.map_coordinates(U, np.vstack((y_profile, x_profile)), order=order)
        Vy_profile = scipy.ndimage.map_coordinates(V, np.vstack((y_profile, x_profile)), order=order)
    
        Vt_profile, Vn_profile = np.dot(rotation_matrix, np.array([Vx_profile, Vy_profile]))
        
        Rtt = Rtt + (Vt_profile - Vt_avg_profile)**2
        Rnn = Rnn + (Vn_profile - Vn_avg_profile)**2
        Rtn = Rtn + (Vt_profile - Vt_avg_profile)*(Vn_profile - Vn_avg_profile)
    
    Rtt /= n_images
    Rnn /= n_images
    Rtn /= n_images
        
    ax.plot(arbitrary_line, Rnn, c=color, ls="-")
    ax.set_xlabel("distance along line [mm]")
    ax.set_ylabel("$R_{nn}$ [m$^2$s$^{-2}$]")
    ax.grid()
    
    ax.set_xlim(np.array([arbitrary_line[0], arbitrary_line[-1]]))
    # ax.set_ylim(np.array([cmin, cmax]))
    
    return Rtt, Rnn, Rtn
    # return Rxx, Ryy, Rxy
    
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
    
#%% Auxiliary functions
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
   
#%% Main

if __name__ == "__main__":

    #%%% Read info of the raw images
    x_raw, y_raw = read_xy_dimensions(calibration_csv_file)
    
    # Invert the y-axis
    y_raw = y_raw[::-1]
    # xv, yv = np.meshgrid(x, y, indexing='ij')
    coordinate_grid = np.array([x_raw, y_raw])

    #%%% Normalize data?
    normalized = False
    
    #%%% Read and save average velocity data
    # File name and scaling parameters from headers of file
    image_file = 'B0001.csv'
    piv_avgV_file = os.path.join(piv_avgV_dir, image_file)
    
    n_windows_x, n_windows_y, X, Y, AvgVx1, AvgVy1, AvgAbsV1 = read_velocity_data(piv_avgV_file, normalized) 
    
    #%%% Read and save transient velocity data
    # n_images = 2500
    # U_transient = np.zeros([n_windows_y, n_windows_x, n_images])
    # V_transient = np.zeros([n_windows_y, n_windows_x, n_images])
    
    # for image_nr in progressbar.progressbar(range(1, n_images + 1)):
        
    #     n_windows_x, n_windows_y, X, Y, U, V, velocity_abs = read_velocity_data(piv_transV_dir, image_nr, normalized)
        
    #     U_transient[:,:,image_nr-1] = U
    #     V_transient[:,:,image_nr-1] = V
    
    #%%% Read and save vector statistics    
    AvgVx, AvgVy, AvgAbsV, AvgKineticE, StdevVx, StdevVy, StdevAbsV, TurbKineticE, RXY, RXX, RYY, TSSmax2D = read_vector_statistics(piv_Rstress_dir, n_windows_x, n_windows_y, normalized)
    
    if normalized:
        vector_stats_labels = ["$V_{x}/U_{b}$", "$V_{y}/U_{b}$", "$|V|/U_{b}$",
                               "$AKE/U^{2}_{b}$", "$\sigma_{V_{x}}/U_{b}$", "$\sigma_{V_{y}}/U_{b}$",
                               "$\sigma_{|V|}/U_{b}$", "$TKE/U^{2}_{b}$", "$R_{XY}/U^{2}_{b}$",
                               "$R_{XX}/U^{2}_{b}$", "$R_{YY}/U^{2}_{b}$", "$TSS_{max}/U^{2}_{b}$"]
    else:
        vector_stats_labels = ["$V_{x}$ [ms$^{-1}$]", "$V_{y}$ [ms$^{-1}$]", "$|V|$ [ms$^{-1}$]",
                           "$AKE$ [m$^{2}$s$^{-2}$]", "$\sigma_{V_{x}}$ [ms$^{-1}$]", "$\sigma_{V_{y}}$ [ms$^{-1}$]",
                           "$\sigma_{|V|}$ [ms$^{-1}$]", "$TKE$ [m$^{2}$s$^{-2}$]", "$R_{XY}$ [m$^{2}$s$^{-2}$]",
                           "$R_{XX}$ [m$^{2}$s$^{-2}$]", "$R_{YY}$ [m$^{2}$s$^{-2}$]", "$TSS_{max}$ [m$^{2}$s$^{-2}$]"]
    
    vector_stats_titles = ["Average velocity in horizontal direction", "$Average velocity in vertical direction", "Average absolute velocity",
                       "Average Kinetic Energy", "Standard deviation of $V_{x}$", "Standard deviation of $V_{y}$",
                       "Standard deviation of $|V|$", "Turbulent Kinetic Energy", "Reynolds Shear Stress $R_{XY}$",
                       "Reynolds Normal Stress $R_{XX}$", "Reynolds Normal Stress $R_{YY}$", "$TSS_{max}$ $[m^{2}s^{-2}$]"]
    
    
    scalars = AvgVx, AvgVy, AvgAbsV, AvgKineticE, StdevVx, StdevVy, StdevAbsV, TurbKineticE, RXY, RXX, RYY, TSSmax2D
    
    #%%% Plots
    plt.close("all")    
    
    #%%%% Plot velocity field [Figure 1]
    
    fig_scale = 1.5
    default_fig_dim = plt.rcParams["figure.figsize"]

    fig1, ax1 = plt.subplots(figsize=(fig_scale*default_fig_dim[0], fig_scale*default_fig_dim[1]), ncols=1, dpi=100)
    ax1.set_title("Average absolute velocity - field")
    
    # Choose a scalar field
    scalar_index = 2
    scalar = scalars[scalar_index]
    label = vector_stats_labels[scalar_index]
    
    scalar_max = np.max(scalar)
    
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
    
    #%%%%% Plot streamlines
    # streamlines = plot_streamlines(fig1, ax1, X, Y, AvgVx1, AvgVy1)
    
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
    thetas_deg = [60, 30, 0, 0, 0] 
    
    thetas = np.radians(thetas_deg) # Conversion to radians
    
    vertical_locs = [0, 0, 0, 15, 30] # in mm
    
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
        # arbitrary_line, Vx_avg_profile, Vy_avg_profile, Vt_avg_profile, Vn_avg_profile = plot_profile_dim(fig4, ax4, rotation_matrix, coord0_mm, coord1_mm, AvgVx, AvgVy, label, cmin, cmax, color, num)
        profile_coords, profile_line = plot_profile_in_field(fig4, ax4, coord0_mm, coord1_mm, color, num)
        Vx_avg_profile, Vy_avg_profile, Vt_avg_profile, Vn_avg_profile = plot_profile_dim(fig5, ax5, rotation_matrix, profile_coords, profile_line, AvgVx, AvgVy, label, cmin, cmax, color, num)
        
        #%%%%% Plot velocity profiles without dimensions [Figure 5]
        # plot_profile_nondim(fig5, ax5, rotation_matrix, coord0_mm, coord1_mm, AvgVx, AvgVy, label, cmin, cmax, color, num, order)
        
        #%%%%% Plot Reynolds normal stresses of selected cross-sections [Figure 6]
        # Rxx_profile, Ryy_profile, Rxy_profile = plot_reynolds_stress1_dim(fig6, ax6, rotation_matrix, theta, coord0_mm, coord1_mm, label, cmin, cmax, color, num)
        Rxx_profile, Ryy_profile, Rxy_profile = plot_reynolds_stress1_dim(fig6, ax6, rotation_matrix, theta, profile_coords, profile_line, label, cmin, cmax, color, num)
        
        # Rtt, Rnn, Rtn = plot_reynolds_stress2_dim(fig6, ax6, rotation_matrix, coord0_mm, coord1_mm, n_images, U_transient, V_transient, Vt_avg_profile, Vn_avg_profile, label, 0, 1.5, color, num, order)
        
        #%%%%% Write data to lists
        profile_lines.append(profile_line)
        Vn_avg_profiles.append(Vn_avg_profile)

    # for profile_id in profile_ids:
        
    #     f  = open("V_n_profile" + str(profile_id) + ".csv", "w")
    #     f.write("distance [mm]" + "," + "V_n [m/s]\n")
    #     for i,j in zip(arbitrary_lines[profile_id], Vn_avg_profiles[profile_id]):
    #         f.write(str(i) + "," + str(j) + "\n")
    #     f.close()

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
        # print(cmax)
    
    if normalized:
        cmax = 0.1
        
    plot_field(figX, axX, X, Y, scalar, label, cmin, cmax)
    
    #%%%%% Draw walls
    draw_walls(axX)
    axX.set_xlim(ax2.get_xlim())
    axX.set_ylim(ax2.get_ylim())
    
    #%%%% Tighten layouts                      
    fig1.tight_layout()
    fig2.tight_layout()
    # fig3.tight_layout()
    fig4.tight_layout()
    # fig5.tight_layout()
    fig6.tight_layout()
    figX.tight_layout()
    
    #%%%% Save figures
    
    # fig1.savefig("figures/figure_1_avg_abs_V.svg", format="svg", dpi=1200, bbox_inches="tight")
    # fig1.savefig("figures/figure_1_avg_abs_V_vectors.svg", format="svg", dpi=1200, bbox_inches="tight")
    # fig1.savefig("figures/figure_1_avg_abs_V_streamlines.svg", format="svg", dpi=1200, bbox_inches="tight")
    
    # fig4.savefig("figures/figure_4_velocity_profiles.svg", format="svg", dpi=1200, bbox_inches="tight")






















