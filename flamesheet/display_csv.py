# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 12:22:11 2022

@author: luuka
"""

#%% Import packages
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from copy import copy, deepcopy
import imutils

#%% Start
cv2.destroyAllWindows()
plt.close("all")

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


def read_image(data_dir, nx, ny):
    
    XYI = np.genfromtxt(data_dir, delimiter=",", skip_header=1)
    X = XYI[:,0].reshape(ny, nx)
    Y = XYI[:,1].reshape(ny, nx)
    I = XYI[:,2].reshape(ny, nx)
    
    return X, Y, I, XYI
    
def plot_image(fig, ax, X, Y, I, brightness_factor):

    intensity_plot = ax.pcolor(X, Y, I, cmap='gray')
    
    intensity_plot.set_clim(0, I.max()/brightness_factor)
    
    ax.set_xlabel('$x\ [mm]$')
    ax.set_ylabel('$y\ [mm]$')
    
    # Set aspect ratio of plot
    ax.set_aspect('equal')
    
    bar = fig.colorbar(intensity_plot)
    bar.set_label('Image intensity [counts]') #('$u$/U$_{b}$ [-]')
    
    return X, Y, I, XYI

#%% Auxiliary functions

def showInMovedWindow(winname, img, x, y):
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, x, y)   # Move it to (x,y)
    cv2.imshow(winname, img)
    
def format_coord(x, y, Z):
    xarr = X_zoom[0,:]
    yarr = Y_zoom[:,0]
    if ((x > xarr.min()) & (x <= xarr.max()) &
        (y > yarr.min()) & (y <= yarr.max())):
        col = np.searchsorted(xarr, x) - 1
        row = len(yarr) - (np.searchsorted(np.flip(yarr), y))
        z = Z[row, col]
        return f'x={x:1.4f}, y={y:1.4f}, z={z:1.4f}   [{row},{col}]'
    else:
        return f'x={x:1.4f}, y={y:1.4f}'
    
def format_coord_Z1(x, y):
    return format_coord(x, y, I_zoom)

def format_coord_Z1_1(x, y):
    return format_coord(x, y, contrast)

def format_coord_Z2(x, y):
    return format_coord(x, y, avg1)

def format_coord_Z2_1(x, y):
    return format_coord(x, y, binary_zoom_x)

def format_coord_Z3(x, y):
    return format_coord(x, y, average_pixel_density_gradient)

def format_coord_Z4(x, y):
    return format_coord(x, y, binary_zoom)

# def format_coord_Z5(x, y):
#     return format_coord(x, y, img_blur)

# def format_coord_ZX1(x, y):
#     return format_coord(x, y, average_pixel_density_gradient_y)

# def format_coord_ZX2(x, y):
#     return format_coord(x, y, binary_zoom1)

#%% Main

if __name__ == "__main__":
    
    #%%% Define data to read 
    
    # READ INFO OF THE RAW IMAGES
    main_dir = "Y:/laaltenburg/flamesheet_2d_campaign1/"
    project_name = "flamesheet_2d_day16"
    project_dir = main_dir + project_name
    calibration_txt_dir = project_dir + "/Properties/Calibration/DewarpedImages1/Export/B0001.txt"
    record_name = "Recording_Date=221223_Time=132754_01"
    
    x_info, y_info = read_xy_dimensions(calibration_txt_dir)
    nx, ny = x_info[2], y_info[2]
    # nx, ny = 1024, 1024 #1034, 1038
    
    record_correction_csv_dir = project_dir + "/" + record_name + "/Correction/Reorganize frames/Export_02/"
    # record_correction_csv_dir = "Export_02_flame_1/"
    
    #%%%% Define image number
    image_nr = str("3319")
    filename = "B" + str(image_nr)+ ".csv" 
    
    image_dir = record_correction_csv_dir + filename
    
    #%%% Read data
    
    X, Y, I, XYI = read_image(image_dir, nx, ny)
    
    x_left, x_right = 0, 5
    y_bottom, y_top = -16, -10
    
    x_left, x_right = -3.4, 16
    y_bottom, y_top = -16, -8
    
    # x_left, x_right = -4, 35
    # y_bottom, y_top = -23, 50
    
    # x_left, x_right = X.min(), X.max()
    # y_bottom, y_top = Y.min(), Y.max()
    
    
    xarr = X[0,:]
    yarr = Y[:,0]
    col_left = np.searchsorted(xarr, x_left) - 1
    col_right = np.searchsorted(xarr, x_right) - 1
    
    row_top = len(yarr) - (np.searchsorted(np.flip(yarr), y_top))
    row_bottom = len(yarr) - (np.searchsorted(np.flip(yarr), y_bottom))
    
    X_zoom = X[row_top:row_bottom, col_left:col_right]
    Y_zoom = Y[row_top:row_bottom, col_left:col_right]
    I_zoom = I[row_top:row_bottom, col_left:col_right]
    
    
    #%%% Figure 1 : raw image
    fig1, ax1 = plt.subplots()
    
    # I_zoom[I_zoom < 525] = 0
    # I_zoom[I_zoom > 900] = 0
    
    brightness_factor = 2
    plot_image(fig1, ax1, X_zoom, Y_zoom, I_zoom, brightness_factor)
    ax1.format_coord = format_coord_Z1
    ax1.set_xlim(np.array([x_left, x_right]))
    ax1.set_ylim(np.array([y_bottom, y_top]))
    
    #%%%% Figure 1-1 : contrast raw image
    # contrast1 = deepcopy(I_zoom)
    # contrast2 = deepcopy(I_zoom)
    
    # threshold = 500
    # contrast1[contrast1 < threshold] = 0
    # contrast1 *= 2
    
    # contrast2[contrast2 >= threshold] = 0
    # contrast2 /= 2
    
    # contrast = contrast1 + contrast2
    
    # contrast = contrast1
    
    # fig11, ax11 = plt.subplots()
    # plot_image(fig11, ax11, X_zoom, Y_zoom, contrast)
    # ax11.format_coord = format_coord_Z1_1
    
    #%%%% Closing
    # size = 3
    # kernel_shape = cv2.MORPH_CROSS # cv2.MORPH_RECT MORPH_ELLIPSE
    # kernel = cv2.getStructuringElement(kernel_shape, (size,size))
    # closing = cv2.morphologyEx(contrast, cv2.MORPH_CLOSE, kernel, iterations = 1)
    
    # fig12, ax12 = plt.subplots()
    # plot_image(fig12, ax12, X_zoom, Y_zoom, closing)
    # ax12.format_coord = format_coord_Z1_2
    
    #%%% Figure 2 : averaged image
    fig2, ax2 = plt.subplots()
    
    # ksize = 3
    # area = ksize*ksize 
    # kernel = np.ones((ksize,ksize),np.float32)
    # avg1 = cv2.filter2D(closing/255, -1, kernel)
    
    src = I_zoom  # Input image array
    ksize = (3, 3) # Gaussian Kernel Size. [height width]. height and width should be odd and can have different values. If ksize is set to [0 0], then ksize is computed from sigma values.
    sigmaX = 0 #3 # Kernel standard deviation along X-axis (horizontal direction).
    avg1 = cv2.GaussianBlur(src, ksize, sigmaX)
    
    # avg1_area_zoom = avg1/area
    plot_image(fig2, ax2, X_zoom, Y_zoom, avg1, brightness_factor)
    ax2.format_coord = format_coord_Z2
    
    #%%%% Figure 2-1 : threshold averaged image
    # binary_zoom_x = deepcopy(avg1_area_zoom)
    # threshold = 0
    # binary_zoom_x[binary_zoom_x < threshold] = 0
    
    # fig21, ax21 = plt.subplots()
    # plot_image(fig21, ax21, X_zoom, Y_zoom, binary_zoom_x)
    # ax21.format_coord = format_coord_Z2_1
    
    #%%% Figure 3 : gradient of averaged image
    
    average_pixel_density = deepcopy(avg1)
    average_pixel_density_ravel = average_pixel_density.ravel()
    
    pixel_density_gradient = np.gradient(average_pixel_density)
    average_pixel_density_gradient_x = np.abs(pixel_density_gradient[0])
    average_pixel_density_gradient_y = np.abs(pixel_density_gradient[1])
    average_pixel_density_gradient = average_pixel_density_gradient_x + average_pixel_density_gradient_y
    # average_pixel_density_gradient = average_pixel_density_gradient_y
    
    fig3, ax3 = plt.subplots()
    brightness_factor = 8
    plot_image(fig3, ax3, X_zoom, Y_zoom, average_pixel_density_gradient, brightness_factor)
    ax3.format_coord = format_coord_Z3
    
    # figX, axX = plt.subplots()
    # plot_image(figX, axX, X_zoom, Y_zoom, I_zoom*average_pixel_density_gradient)
    # # axX.format_coord = format_coord_Z3
    
    
    #%%% Figure 4 : thresholding of gradient image
    binary_zoom = deepcopy(average_pixel_density_gradient)
    threshold = 20
    binary_zoom[binary_zoom < threshold] = 0
    
    fig4, ax4 = plt.subplots()
    brightness_factor = 1
    plot_image(fig4, ax4, X_zoom, Y_zoom, binary_zoom, brightness_factor)
    ax4.format_coord = format_coord_Z4
    
    final1 = np.ones(I_zoom.shape)
    final2 = np.ones(I_zoom.shape)
    
    
    # final1 = deepcopy(I_zoom)
    # final2 = deepcopy(binary_zoom)
    
    final1[I_zoom < 650] = 0
    final2[average_pixel_density_gradient < threshold] = 0
    
    final = final1 + final2
    
    
    final[final == 2] = 1
    
    fig5, ax5 = plt.subplots()
    plot_image(fig5, ax5, X_zoom, Y_zoom, final, brightness_factor)
    
    size = 3
    kernel_shape = cv2.MORPH_RECT # cv2.MORPH_RECT MORPH_ELLIPSE
    kernel = cv2.getStructuringElement(kernel_shape, (size,size))
    closing2 = cv2.morphologyEx(final, cv2.MORPH_CLOSE, kernel, iterations = 1)
    
    fig6, ax6 = plt.subplots()
    brightness_factor = 1
    plot_image(fig6, ax6, X_zoom, Y_zoom, closing2, brightness_factor)
    
    # size = 9
    # kernel_shape = cv2.MORPH_ELLIPSE # cv2.MORPH_ELLIPSE
    # kernel = cv2.getStructuringElement(kernel_shape, (size,size))
    opening = cv2.morphologyEx(closing2, cv2.MORPH_OPEN, kernel, iterations = 1)
    
    fig7, ax7 = plt.subplots()
    plot_image(fig7, ax7, X_zoom, Y_zoom, opening)
    
    
    # ax4.format_coord = format_coord_Z4
    
    # figX3, axX3 = plt.subplots()
    # I_zoom_density = deepcopy(I_zoom)
    # I_zoom_density_ravel = I_zoom_density.ravel()
    # axX3.hist(I_zoom_density_ravel, bins=255, density=True, fc='k', ec='k') #calculating histogram
    
    # fig3, ax3 = plt.subplots()
    # ax3.hist(average_pixel_density_ravel, bins=255, density=True, fc='k', ec='k') #calculating histogram
    
    # src = average_pixel_density  # Input image array
    # ksize = (3, 3) # Gaussian Kernel Size. [height width]. height and width should be odd and can have different values. If ksize is set to [0 0], then ksize is computed from sigma values.
    # sigmaX = 3 #3 # Kernel standard deviation along X-axis (horizontal direction).
    # img_blur = cv2.GaussianBlur(src, ksize, sigmaX)
    
    # fig5, ax5 = plt.subplots()
    # plot_image(fig5, ax5, X_zoom, Y_zoom, img_blur)
    
    # fig4, ax4 = plt.subplots()
    
    # binary_zoom = deepcopy(img_blur)
    # binary_zoom = deepcopy(pixel_density)
    
    # threshold = 2.5
    # binary_zoom[binary_zoom < threshold] = 0
    # # z[z >= threshold] = 1
    
    # plot_image(fig4, ax4, X_zoom, Y_zoom, binary_zoom)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # ax4.format_coord = format_coord_Z4
    # ax5.format_coord = format_coord_Z5
    # axX1.format_coord = format_coord_ZX1
    
    
    # ax1.set_xlim(np.array([x_left, x_right]))
    # ax1.set_ylim(np.array([y_bottom, y_top]))
    
    # ax2.set_xlim(ax1.get_xlim())
    # ax2.set_ylim(ax1.get_ylim())
    
    # ax4.set_xlim(ax1.get_xlim())
    # ax4.set_ylim(ax1.get_ylim())
    
    # ax5.set_xlim(ax1.get_xlim())
    # ax5.set_ylim(ax1.get_ylim())
