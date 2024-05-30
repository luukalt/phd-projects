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
from scipy.integrate import trapz
import progressbar
from wall_detection import *
# from wall_detection1 import *

import sys
import os

if sys.version_info[0] >= 3:
    import tkinter as tk
    from tkinter import ttk
else:
    import Tkinter as tk

from numpy.linalg import multi_dot

#%% Start
plt.close("all")

#%% Figure settings

# Color maps
tableau = cm.tab10.colors

#%% Main functions
def read_xy_dimensions(data_dir):
    
    raw_file = open(data_dir, "r")
    scaling_info = raw_file.readline()
    raw_file.close()
    scaling_info_raw = scaling_info.split()
    
    n_windows_x_raw = int(scaling_info_raw[3])
    n_windows_y_raw = int(scaling_info_raw[4])

    window_size_x_raw = float(scaling_info_raw[6])
    x_origin_raw = float(scaling_info_raw[7])
    
    window_size_y_raw = float(scaling_info_raw[10])
    y_origin_raw = float(scaling_info_raw[11])

    x_left_raw = x_origin_raw
    x_right_raw = x_origin_raw + (n_windows_x_raw - 1)*window_size_x_raw
    y_bottom_raw = y_origin_raw + (n_windows_y_raw - 1)*window_size_y_raw
    y_top_raw = y_origin_raw
    
    return (x_left_raw, x_right_raw, n_windows_x_raw, window_size_x_raw), (y_bottom_raw, y_top_raw, n_windows_y_raw, window_size_y_raw)


def read_velocity_data(data_dir, image_nr, normalized):
    
    # Set if plot is normalized or non-dimensionalized
    if normalized:
        U_bulk = 1
    else:
        U_bulk = 1
    
    if image_nr < 10:
        image_prefix = 'B000'
    if 10 <= image_nr <= 100:
        image_prefix = 'B00'
    if 100 <= image_nr <= 1000:
        image_prefix = 'B0' 
    if 1000 <= image_nr <= 10000:
        image_prefix = 'B' 
  
    # File name and scaling parameters from headers of file
    XYUV_file = data_dir + image_prefix + str(image_nr) + '.txt'
    
    piv_file = open(XYUV_file, "r")
    scaling_info = piv_file.readline()
    piv_file.close()
    scaling_info_vector = scaling_info.split()
    n_windows_x = int(scaling_info_vector[7])
    n_windows_y = int(scaling_info_vector[6])
    
    XYUV = np.genfromtxt(XYUV_file, delimiter=",")
    df = pd.read_csv(XYUV_file)
    
    X = XYUV[:,0].reshape(n_windows_y, n_windows_x)
    Y = XYUV[:,1].reshape(n_windows_y, n_windows_x)
    U = XYUV[:,2].reshape(n_windows_y, n_windows_x) # *-1 because -> inverse x-axis
    V = XYUV[:,3].reshape(n_windows_y, n_windows_x)
    velocity_abs = np.sqrt(U**2 + V**2)
    
    return n_windows_x, n_windows_y, X, Y, U, V, velocity_abs, XYUV

def read_reynolds_stress_data(data_dir, n_windows_x, n_windows_y, normalized):
    
    # Set if plot is normalized or non-dimensionalized
    if normalized:
        U_bulk = 1
    else:
        U_bulk = 1
    
    Re_stress_file_nrs = [9, 10, 11] # Rxy, Rxx, Ryy
    R_stresses = np.zeros([n_windows_y, n_windows_x, len(Re_stress_file_nrs)])
    
    for file_nr in Re_stress_file_nrs: 
        
        if file_nr < 10:
            image_prefix = 'B000'
        if 10 <= file_nr <= 100:
            image_prefix = 'B00'
        if 100 <= file_nr <= 1000:
            image_prefix = 'B0' 
        if 1000 <= file_nr <= 10000:
            image_prefix = 'B' 
  
        # File name and scaling parameters from headers of file
        R_stress_file = data_dir + image_prefix + str(file_nr) + '.txt'
        
        R_stress = np.genfromtxt(R_stress_file, delimiter=",")
        
        R_stresses[:,:,file_nr-Re_stress_file_nrs[0]] = R_stress/(U_bulk**2)
    
    return R_stresses


def plot_field(fig, ax, X, Y, quantity, label, cmin, cmax):
    
    quantity_plot = ax.pcolor(X, Y, quantity, cmap='viridis')
    # quantity_plot.set_clim(0, np.max(quantity))
    quantity_plot.set_clim(0, 16)
    # quantity_plot.set_clim(-1, 1)
    
    ax.set_xlabel('$x\ [mm]$')
    ax.set_ylabel('$y\ [mm]$')
    
    # Set aspect ratio of plot
    ax.set_aspect('equal')
    
    bar = fig.colorbar(quantity_plot, ax=ax)
    bar.set_label(label) #('$u$/U$_{b}$ [-]')
    
def plot_vector_field(fig, ax, X, Y, U, V):
    scale = 2
    headwidth = 6
    color = "r"
    skip = 2
    ax.quiver(X[0::skip, 0::skip], Y[0::skip, 0::skip], U[0::skip, 0::skip], V[0::skip, 0::skip], color=color, angles='xy', scale_units='xy', scale=scale, headwidth=headwidth)
    
def plot_streamlines(fig, ax, XYUV):
    
    XYUV[XYUV == -0] = 0
    
    X = XYUV[:,0].reshape(n_windows_y, n_windows_x)
    Y = XYUV[:,1].reshape(n_windows_y, n_windows_x)
    U = XYUV[:,2].reshape(n_windows_y, n_windows_x) # *-1 because -> inverse x-axis
    V = XYUV[:,3].reshape(n_windows_y, n_windows_x)
    
    from scipy.interpolate import interp2d
    # regularly spaced grid spanning the domain of x and y 
    xi = np.linspace(X.min(), X.max(), X.shape[1])
    yi = np.linspace(Y.min(), Y.max(), Y.shape[0])
    
    # bicubic interpolation
    uCi = interp2d(X, Y, U)(xi, yi)
    vCi = interp2d(X, Y, V)(xi, yi)
    
    start_points_1 = [[-2, j] for j in range(-18, -14, 2)]
    start_points_2 = [[i, 10] for i in range(-2, 25, 4)]
    start_points = start_points_1 + start_points_2 
    
    # StreamplotSet = ax.streamplot(xi, yi, uCi, vCi, density=0.4, color="k", minlength=0.1, arrowsize=2, broken_streamlines=False)
    StreamplotSet = ax.streamplot(xi, yi, uCi, vCi, start_points=start_points, density=0.5, color="k", minlength=0.1, arrowsize=2, broken_streamlines=False)
    
    return StreamplotSet

    
def plot_profile_dim(fig, ax, rotation_matrix, coord0_mm, coord1_mm, quantity_x, quantity_y, label, cmin, cmax, color, num, order):
    
    x0_mm, y0_mm = coord0_mm
    x1_mm, y1_mm = coord1_mm
    x0, y0 = (coord0_mm - np.array([X0, Y0]))/np.array([dx_piv, dy_piv])
    x1, y1 = (coord1_mm - np.array([X0, Y0]))/np.array([dx_piv, dy_piv])
    
    x_profile, y_profile = np.linspace(x0, x1, num), np.linspace(y0, y1, num)
    ax[0].plot(np.linspace(x0_mm, x1_mm, num), np.linspace(y0_mm, y1_mm, num), c=color, ls="--")
    
    # Extract the values along the line, using first, second or third order interpolation
    arbitrary_line = np.linspace(0, np.sqrt((x1_mm - x0_mm)**2 + (y1_mm - y0_mm)**2), num)
    
    quantity_x_profile = scipy.ndimage.map_coordinates(quantity_x, np.vstack((y_profile, x_profile)), order=order)
    quantity_y_profile = scipy.ndimage.map_coordinates(quantity_y, np.vstack((y_profile, x_profile)), order=order)
    
    quantity_tangent_profile, quantity_normal_profile  = np.dot(rotation_matrix, np.array([quantity_x_profile, quantity_y_profile]))
    
    quantity_tangent_profile = quantity_x_profile*np.cos(theta) - quantity_y_profile*np.sin(theta)
    quantity_normal_profile = quantity_x_profile*np.sin(theta) + quantity_y_profile*np.cos(theta)
    
    ax[1].plot(arbitrary_line, quantity_normal_profile, c=color, ls="-", marker="o")
    
    ax[1].set_xlabel("distance along line [mm]")
    ax[1].set_ylabel("$V_{nn}$ [ms$^-1$]")
    
    # ax[1].set_xlim(np.array([arbitrary_line[0], arbitrary_line[-1]]))
    ax[1].set_xlim(np.array([0, 30]))
    # ax[1].set_ylim(np.array([cmin, cmax]))
    ax[1].grid()
    ax[1].axhline(y=0, color='k')
    
    U_bulk_2d = trapz(quantity_normal_profile, arbitrary_line)/10
    # print(trapz(quantity_normal_profile, arbitrary_line))
    # print("U_bulk= {0:.1f} m/s".format(U_bulk_2d))
    
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
    
    ax[1].plot(quantity_normal_profile, c=color, ls="-", marker="o")
    
    ax[1].set_xlabel('-')
    ax[1].set_ylabel(label)
    
    ax[1].set_xlim(np.array([0, len(quantity_x_profile)]))
    # ax[1].set_ylim(np.array([cmin, cmax]))
    ax[1].grid()
    ax[1].axhline(y=0, color='k')
    
    
def plot_reynolds_stress1_dim(fig, ax, theta, coord0_mm, coord1_mm, label, cmin, cmax, color, num, order):
    
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array(((c, -s), (s, c)))
    rotation_matrix_T = np.transpose(rotation_matrix)
    
    x0_mm, y0_mm = coord0_mm
    x1_mm, y1_mm = coord1_mm
    x0, y0 = (coord0_mm - np.array([X0, Y0]))/np.array([dx_piv, dy_piv])
    x1, y1 = (coord1_mm - np.array([X0, Y0]))/np.array([dx_piv, dy_piv])
    
    x_profile, y_profile = np.linspace(x0, x1, num), np.linspace(y0, y1, num)
    
    # Extract the values along the line, using first, second or third order interpolation
    arbitrary_line_length = np.sqrt((x1_mm - x0_mm)**2 + (y1_mm - y0_mm)**2)
    arbitrary_line = np.linspace(0, np.sqrt((x1_mm - x0_mm)**2 + (y1_mm - y0_mm)**2), int(arbitrary_line_length/dx_piv))
    
    Rxx_profile = scipy.ndimage.map_coordinates(Rxx, np.vstack((y_profile, x_profile)), order=order)
    Ryy_profile = scipy.ndimage.map_coordinates(Ryy, np.vstack((y_profile, x_profile)), order=order)
    Rxy_profile = scipy.ndimage.map_coordinates(Rxy, np.vstack((y_profile, x_profile)), order=order)
    
    ### Approach 1: Calculate Reynolds stresses on arbirtary line "manually"
    Rtt = Rxx_profile*(np.cos(theta))**2 - 2*Rxy_profile*np.cos(theta)*np.sin(theta) + Ryy_profile*(np.sin(theta))**2
    Rtn = (Rxx_profile - Ryy_profile)*np.cos(theta)*np.sin(theta) + Rxy_profile*((np.cos(theta))**2 - (np.sin(theta))**2)
    Rnn = Rxx_profile*(np.sin(theta))**2 + 2*Rxy_profile*np.cos(theta)*np.sin(theta) + Ryy_profile*(np.cos(theta))**2

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
    
    ax.plot(arbitrary_line, Rnn, c=color, ls="-")
    # ax.set_xlim(np.array([arbitrary_line[0], arbitrary_line[-1]]))
    ax.set_xlim(np.array([0, 30]))
    
    ax.set_xlabel("distance along line [mm]")
    ax.set_ylabel("$R_{nn}$ [m$^2$s$^{-2}$]")
    ax.grid()
    
    
    # ax.set_ylim(np.array([cmin, cmax]))
    
    return Rxx_profile, Ryy_profile, Rxy_profile



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
    
    # Back plate
    pt1 = coordinate_grid[:, pt1_core_right[0], pt1_core_right[1]]
    pt2 = coordinate_grid[:, -1, 0]
    pt3 = coordinate_grid[:, -1, -1]
    pt4 = coordinate_grid[:, pt2_core_right[0], pt2_core_right[1]]
    ax.add_patch(Polygon([pt1, pt2, pt3, pt4], color="silver"))
    
    
    cx_mm = x_info[0] + cx*x_info[3]
    cy_mm = y_info[1] + cy*y_info[3]
    radius_mm = radius*x_info[3]
    
    num = 1000
    theta1 = np.linspace(np.pi/4, -3*np.pi/4, num)
    
    y_theta1 = cy_mm + radius_mm*np.sin(theta1)
    array = y_theta1
    value = y[pt2_core_left[1]]
    idx1 = find_nearest(array, value)
    
    theta2 = np.linspace(theta1[idx1], -np.pi, num)
    
    x_theta2 = cx_mm + radius_mm*np.cos(theta2)
    y_theta2 = cy_mm + radius_mm*np.sin(theta2)
    
    xy_zip  = zip(x_theta2, y_theta2)
    xy_unzip_list = list(xy_zip)
    
    pt1 = x_theta2[-1], coordinate_grid[:, 0, 0][1]
    pt2 = x_theta2[-1]-7.5, coordinate_grid[:, 0, 0][1] #coordinate_grid[:, 0, 0]
    pt3 = x_theta2[-1]-7.5, coordinate_grid[:, -1, -1][1] 
    pt4 = np.array([pt1_core_left_mm[0] + 4, pt2_core_right_mm[1]])
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

#%% Objects
class App(tk.Frame):

    def __init__(self, master):
        
        tk.Frame.__init__(self, master)
        
        self.main_dir = "Y:/"
        
        exclude = ['davis', '.exp', '.im7', '.set', '.xml', '.txt', '.cih', '.attr', '.scales', 'mraw']
        
        self.dict = {}
        self.dict2 = {}
        
        projects = os.listdir(self.main_dir)
        projects = [project for project in projects if not any(e in project for e in exclude)]
        
        for project in projects:
            
            records = os.listdir(self.main_dir + project + "/")
            records = [record for record in records if not any(e in record for e in exclude)]
            
            self.dict[project] = records
        
            for record in records:
                
                piv_folders = os.listdir(self.main_dir + project + "/" + record + "/")
                piv_folders = [piv_folder for piv_folder in piv_folders if not any(e in piv_folder for e in exclude)]
                
                self.dict2[record] = piv_folders
            
        self.project_variable = tk.StringVar(self)
        self.record_variable = tk.StringVar(self)
        self.pre_record_variable = tk.StringVar(self)
        self.piv_variable = tk.StringVar(self)
        
        self.project_variable.trace('w', self.update_options1)
        self.record_variable.trace('w', self.update_options2)

        self.projects_optionmenu = tk.OptionMenu(self, self.project_variable, *self.dict.keys())
        self.records_optionmenu = tk.OptionMenu(self, self.record_variable, '')
        self.pre_records_optionmenu = tk.OptionMenu(self, self.pre_record_variable, '')
        self.piv_optionmenu = tk.OptionMenu(self, self.piv_variable, *self.dict2.keys())
        
        project_nr = 1
        self.project_variable.set(projects[project_nr])
        self.record_variable.set(self.dict[projects[project_nr]][0])
        self.pre_record_variable.set(self.dict[projects[project_nr]][0])    
        # self.piv_variable.set(self.dict2[self.dict[projects[project_nr]][0]][-1])
        
        self.projects_label = tk.Label(self, text="Choose project:").grid(row=0, column=0)
        self.projects_optionmenu.grid(row=0, column=1)
        
        self.projects_label = tk.Label(self, text="Choose recording:").grid(row=1, column=0)
        self.records_optionmenu.grid(row=1, column=1)
        
        self.projects_label = tk.Label(self, text="Choose pre-recording:").grid(row=2, column=0)
        self.pre_records_optionmenu.grid(row=2, column=1)
        
        self.piv_label = tk.Label(self, text="Choose PIV result:").grid(row=3, column=0)
        self.piv_optionmenu.grid(row=3, column=1)
        
        # Button for loading data
        self.exit_button = ttk.Button(self, text="LOAD DATA", command=self.quit_me)
        self.exit_button.grid(row=4, column=1)
        
        self.pack()
        
    def quit_me(self):
            root.quit()
            root.destroy()
        
    def update_options1(self, *args):
        records = self.dict[self.project_variable.get()]
        self.record_variable.set(records[0])

        menu = self.records_optionmenu['menu']
        menu1 = self.pre_records_optionmenu['menu']
        
        menu.delete(0, 'end')
        menu1.delete(0, 'end')

        for record in records:
            menu.add_command(label=record, command=lambda recording=record: self.record_variable.set(recording))   
            menu1.add_command(label=record, command=lambda recording=record: self.pre_record_variable.set(recording))
    
    def update_options2(self, *args):
        piv_folders = self.dict2[self.record_variable.get()]
        self.piv_variable.set(piv_folders[0])

        menu = self.piv_optionmenu['menu']
        menu.delete(0, 'end')

        for piv_folder in piv_folders:
            menu.add_command(label=piv_folder, command=lambda piv=piv_folder: self.piv_variable.set(piv))    
#%% Main

# Choose project
# root = tk.Tk()
# root.title("Choose experimental data")
# root.geometry('600x300')
# app = App(root)
# app.mainloop() 

# main_dir = app.main_dir
# project_name = app.project_variable.get()
# record_name = app.record_variable.get()
# pre_record_name = app.pre_record_variable.get()
# piv_result = app.piv_variable.get()

# H2% = 0, phi = 0, Re_H = 2500, image_rate = 0.2 kHz
# main_dir = "Y:/"
# project_name = "flamesheet_2d_day5-1"
# record_name = "Recording_Date=220915_Time=131449_temp_Conversion"
# pre_record_name = "Recording_Date=220915_Time=131136"
# piv_result = "PIV_MP(3x32x32_50%ov_ImgCorr)_Mask"

# # H2% = 0, phi = 0.7, Re_H = 2500, image_rate = 0.2 kHz
# main_dir = "Y:/"
# project_name = "flamesheet_2d_day5-2"
# record_name = "Recording_Date=220915_Time=150533_temp_Conversion"
# pre_record_name = "Recording_Date=220915_Time=145548"
# piv_result = "PIV_MP(3x32x32_50%ov_ImgCorr)_Mask"

# H2% = 40, phi = 0.6, Re_H = 5000, image_rate = 0.2 kHz

# H2% = 60, phi = 0.5, Re_H = 5500, image_rate = 0.2 kHz

# H2% = 80, phi = 0.4, Re_H = 6000, image_rate = 0.2 kHz

# H2% = 0, phi = 0, Re_H = 7000, image_rate = 0.2 kHz
main_dir = os.path.join('Y:', 'laaltenburg', 'flamesheet_2d_campaign1')
project_name = '/flamesheet_2d_day14'

# record_name = "Recording_Date=221118_Time=144758_01"
# pre_record_name = "Recording_Date=221118_Time=112139"
# piv_result = "PIV_MP(3x32x32_50%ov_ImgCorr)"

# H2% = 100, phi = 0.3, Re_H = 7000, image_rate = 0.2 kHz 
record_name = "Recording_Date=221118_Time=115220_01"
pre_record_name = "Recording_Date=221118_Time=112139"
piv_result = "PIV_MP(3x32x32_50%ov_ImgCorr)_01"

# # H2% = 100, phi = 0.3, Re_H = 7000, image_rate = 4.5 kHz
record_name = "Recording_Date=221118_Time=125724_01"
pre_record_name = "Recording_Date=221118_Time=112139"
piv_result = "PIV_MP(3x32x32_50%ov_ImgCorr)_01"

project_dir = main_dir + project_name

calibration_txt_dir = project_dir + "/Properties/Calibration/DewarpedImages1/Export/B0001.txt"
calibration_tif_dir = project_dir + "/Properties/Calibration/DewarpedImages1/Export_01/B0001.tif"

pre_record_raw_dir = project_dir + "/" + pre_record_name + "/Reorganize frames/Export/B0001.tif" 
pre_record_correction_dir = project_dir + "/" + pre_record_name + "/Correction/Reorganize frames/Export/B0001.tif" 

piv_avgV_dir = project_dir + "/" + record_name + "/" + piv_result + "/Avg_Stdev/Export/"
piv_transV_dir = project_dir + "/" + record_name + "/" + piv_result + "/Export/"
piv_Rstress_dir =  project_dir + "/" + record_name + "/" + piv_result + "/AvgVx_AvgVy_AvgV_StdevVx_StdevVy_StdevV_Rxy_Rxx_Ryy_AKE_TKE_TSS/Export/"

normalized = False
image_nr = 1

# READ INFO OF THE RAW IMAGES
x_info, y_info = read_xy_dimensions(calibration_txt_dir)
nx, ny = x_info[2], y_info[2]
x, y = np.linspace(x_info[0], x_info[1], nx), np.linspace(y_info[0], y_info[1], ny)

# INVERT THE Y-AXIS
y = y[::-1]
xv, yv = np.meshgrid(x, y, indexing='ij')
coordinate_grid = np.array([xv, yv])

# READ AND SAVE AVERAGE VELOCITY DATA
n_windows_x, n_windows_y, X, Y, U_avg, V_avg, velocity_abs_avg, XYUV = read_velocity_data(piv_avgV_dir, image_nr, normalized) 

# READ AND SAVE TRANSIENT VELOCITY DATA
n_images = 2500
U_transient = np.zeros([n_windows_y, n_windows_x, n_images])
V_transient = np.zeros([n_windows_y, n_windows_x, n_images])

# for image_nr in progressbar.progressbar(range(1, n_images + 1)):
    
#     n_windows_x, n_windows_y, X, Y, U, V, velocity_abs = read_velocity_data(piv_transV_dir, image_nr, normalized)
    
#     U_transient[:,:,image_nr-1] = U
#     V_transient[:,:,image_nr-1] = V
    
R_stresses = read_reynolds_stress_data(piv_Rstress_dir, n_windows_x, n_windows_y, normalized)
Rxy, Rxx, Ryy = R_stresses[:,:,0], R_stresses[:,:,1], R_stresses[:,:,2]

#%% Plots
plt.close("all")    

fig1, ax1 = plt.subplots()
colors = tableau
label = "$|V|$ [ms$^-1$]"
cmin = 0
cmax = 1.5

# CHOOSE SCALAR FIELD
scalar = velocity_abs_avg
plot_field(fig1, ax1, X, Y, scalar, label, cmin, cmax)
plot_vector_field(fig1, ax1, X, Y, U_avg, V_avg)
StreamplotSet = plot_streamlines(fig1, ax1, XYUV)

pt1_liner, pt2_liner, pt1_core_left, pt2_core_left, pt1_core_right, pt2_core_right, cx, cy, radius = wall_detection(calibration_tif_dir, pre_record_correction_dir)

# Convert liner tip coordinate in pixels to coordinate in mm
pt1_liner_mm = coordinate_grid[:, pt1_liner[0], pt1_liner[1]]
pt2_liner_mm = coordinate_grid[:, pt2_liner[0], pt2_liner[1]]

# Convert coordinates of left wall of core flow in pixels to coordinate in mm
pt1_core_left_mm = coordinate_grid[:, pt1_core_left[0], pt1_core_left[1]]
pt2_core_left_mm = coordinate_grid[:, pt2_core_left[0], pt2_core_left[1]]

# Convert coordinates of left wall of core flow in pixels to coordinate in mm
pt1_core_right_mm = coordinate_grid[:, pt1_core_right[0], pt1_core_right[1]]
pt2_core_right_mm = coordinate_grid[:, pt2_core_right[0], pt2_core_right[1]]

fig2, ax2 = plt.subplots()
plot_field(fig2, ax2, X, Y, scalar, label, cmin, cmax)


# fig3, ax3 = plt.subplots()
# X_check, Y_check, I_check, XYI = plot_image(fig3, ax3, nx, ny)

# Draw walls
cx_mm, cy_mm, radius_mm = draw_walls(ax1)

# Top left coordinates of velocity data in mm
X0, Y0 = X[0][0], Y[0][0]

# Spacing between velocity vector in mm
dx_piv, dy_piv  = np.diff(X[0,:])[0], np.diff(Y[:,0])[0]

# Coordinates of the liner tip in PIV "pixels". This is needed for extracting the profiles
coord0_mm = pt2_liner_mm

fig4, ax4 = plt.subplots(ncols=2)
fig5, ax5 = plt.subplots(ncols=2)
# fig6, ax6 = plt.subplots()
fig7, ax7 = plt.subplots()

# Plot absolute velocity field
plot_field(fig4, ax4[0], X, Y, scalar, label, cmin, cmax)
ax5[0].imshow(scalar)

# DEFINE CROSS_SECTIONS FOR EXTRACTION OF PROFILES
# Angle with respect to the horizon of cross-section for profile in degrees
thetas_deg = [80, 60, 30, 0, 0, 0, 0] 
thetas = np.radians(thetas_deg) # Conversion to radians

vertical_locs = [0, 0, 0, 0, 9, 18, 27] # in mm

# Number of points for arbitrary line AND Order of profile fit for arbitrary line
order = 0

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
    
    # Plot normal velocity profiles with dimensions for cross-sections depending on theta
    Vx_avg_profile, Vy_avg_profile, Vt_avg_profile, Vn_avg_profile = plot_profile_dim(fig4, ax4, rotation_matrix, coord0_mm, coord1_mm, U_avg, V_avg, label, cmin, cmax, color, num, order)
    
    # Plot velocity profiles without dimensions
    plot_profile_nondim(fig5, ax5, rotation_matrix, coord0_mm, coord1_mm, U_avg, V_avg, label, cmin, cmax, color, num, order)
    
    # Plot Reynolds normal stresses of selected cross-sections
    Rxx_profile, Ryy_profile, Rxy_profile = plot_reynolds_stress1_dim(fig7, ax7, theta, coord0_mm, coord1_mm, label, cmin, cmax, color, num, order)
    # Rtt, Rnn, Rtn = plot_reynolds_stress2_dim(fig6, ax6, rotation_matrix, coord0_mm, coord1_mm, n_images, U_transient, V_transient, Vt_avg_profile, Vn_avg_profile, label, 0, 1.5, color, num, order)
    

    
# Draw walls in figure
draw_walls(ax4[0])
ax4[0].set_xlim(ax2.get_xlim())
ax4[0].set_ylim(ax2.get_ylim())
                       
fig1.tight_layout()
fig2.tight_layout()
# fig3.tight_layout()
fig4.tight_layout()
fig5.tight_layout()
fig7.tight_layout()
























