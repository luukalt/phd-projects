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
    
    # bar = fig.colorbar(intensity_plot)
    # bar.set_label('Image intensity [counts]') #('$u$/U$_{b}$ [-]')
    
    #hide x-axis
    ax.get_xaxis().set_visible(False)
    
    #hide y-axis
    ax.get_yaxis().set_visible(False)
    
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
        
        # col = len(xarr) - (np.searchsorted(np.flip(xarr), x)) # flame 4
        
        row = len(yarr) - (np.searchsorted(np.flip(yarr), y))
        
        z = Z[row, col]
        
        return f'x={x:1.4f}, y={y:1.4f}, z={z:1.4f}   [{row},{col}]'
    
    else:
        
        return f'x={x:1.4f}, y={y:1.4f}'
    
def format_coord_Z1(x, y):
    return format_coord(x, y, I_zoom)

# def format_coord_Z1_1(x, y):
#     return format_coord(x, y, contrast)

def format_coord_Z2(x, y):
    return format_coord(x, y, blur_gaus)

# def format_coord_Z2_1(x, y):
#     return format_coord(x, y, binary_zoom_x)

def format_coord_Z3(x, y):
    return format_coord(x, y, pixel_density_gradient_combined)

def format_coord_Z4(x, y):
    return format_coord(x, y, thresholding_1)

def format_coord_Zopen(x, y):
    return format_coord(x, y, opening)

def format_coord_Zclose(x, y):
    return format_coord(x, y, closing)

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
    record_name = record_name + "/" + "SubOverTimeAvg_sl=all"
    
    x_info, y_info = read_xy_dimensions(calibration_txt_dir)
    nx, ny = x_info[2], y_info[2]
    # nx, ny = 512, 497 #1034, 1038
    
    record_correction_csv_dir = project_dir + "/" + record_name + "/Correction/Reorganize frames/Export_02/"
    # record_correction_csv_dir = "Export_02_flame_4/"
    
    #%%%% Define image number
    image_nr = str("3318")
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
    
    # X_zoom = X
    # Y_zoom = Y
    # I_zoom = I
    
    
    #%%% Figure 1 : raw image
    fig1, ax1 = plt.subplots()
    
    brightness_factor = 4
    plot_image(fig1, ax1, X_zoom, Y_zoom, I_zoom, brightness_factor)
    ax1.format_coord = format_coord_Z1
    ax1.set_xlim(np.array([x_left, x_right]))
    ax1.set_ylim(np.array([y_bottom, y_top]))
    
    #%%% Open -> close [0]
    fig_open, ax_open = plt.subplots()
    src = I_zoom
    size = 2
    kernel_shape = cv2.MORPH_ELLIPSE # cv2.MORPH_RECT MORPH_ELLIPSE
    kernel = cv2.getStructuringElement(kernel_shape, (size,size))
    opening = cv2.morphologyEx(src, cv2.MORPH_OPEN, kernel, iterations = 2)
    
    brightness_factor = 2
    plot_image(fig_open, ax_open, X_zoom, Y_zoom, opening, brightness_factor)
    ax_open.format_coord = format_coord_Zopen

    fig_close, ax_close = plt.subplots()
    src = opening
    size = 2
    kernel_shape = cv2.MORPH_ELLIPSE # cv2.MORPH_RECT MORPH_ELLIPSE
    kernel = cv2.getStructuringElement(kernel_shape, (size,size))
    closing = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel, iterations = 2)
    
    brightness_factor = 8
    plot_image(fig_close, ax_close, X_zoom, Y_zoom, closing, brightness_factor)
    ax_close.format_coord = format_coord_Zclose
    
    # fig_close.tight_layout()
    
    # fig_close.savefig("test.png",bbox_inches='tight')

    
    threshold_pixel = 5
    final = np.zeros(I_zoom.shape)
    final[closing > threshold_pixel] = 1

    fig_final, ax_final = plt.subplots()
    brightness_factor = 1
    plot_image(fig_final, ax_final, X_zoom, Y_zoom, final, brightness_factor)
    
    
    
    #%%% Figure 2 : Gausian blur the raw image
    fig2, ax2 = plt.subplots()
    
    # src = I_zoom
    # ksize = 3
    # area = ksize*ksize 
    # kernel = np.ones((ksize,ksize),np.float32)
    # blur_gaus = cv2.filter2D(src/area, -1, kernel)
    
    src = closing  # Input image array
    size = 3
    ksize = (size, size) # Gaussian Kernel Size. [height width]. height and width should be odd and can have different values. If ksize is set to [0 0], then ksize is computed from sigma values.
    sigmaX = 0 #3 # Kernel standard deviation along X-axis (horizontal direction).
    blur_gaus = cv2.GaussianBlur(src, ksize, sigmaX)
    
    brightness_factor = 2
    plot_image(fig2, ax2, X_zoom, Y_zoom, blur_gaus, brightness_factor)
    
    
    #%%% Brighten image by blending gray image and blurred image (alpha and beta are the blending weights of the images)
    fig22, ax22 = plt.subplots()
    src1 = I_zoom # First input image array
    alpha = -2 # -1 # Weight of the first array elements
    src2 = blur_gaus# Second input image array
    beta = 6 # 3 Weight of the first array elements
    gamma = 0 # Scalar added to each sum
    bright = cv2.addWeighted(src1, alpha, src2, beta, gamma)
    
    brightness_factor = 2
    plot_image(fig22, ax22, X_zoom, Y_zoom, bright, brightness_factor)
    
    #%%% Figure 3 : gradient of Gaussian blurred image
    
    fig3, ax3 = plt.subplots()
    pixel_density = deepcopy(blur_gaus)
    
    pixel_density_ravel = pixel_density.ravel()
    
    pixel_density_gradient = np.gradient(pixel_density)
    pixel_density_gradient_x = np.abs(pixel_density_gradient[0])
    pixel_density_gradient_y = np.abs(pixel_density_gradient[1])
    pixel_density_gradient_combined = pixel_density_gradient_x + pixel_density_gradient_y
    
    
    brightness_factor = 2
    plot_image(fig3, ax3, X_zoom, Y_zoom, pixel_density_gradient_combined, brightness_factor)
    ax3.format_coord = format_coord_Z3
    
    
    #%%% Figure 4 : thresholding of gradient image
    fig4, ax4 = plt.subplots()
    
    thresholding_1 = deepcopy(pixel_density_gradient_combined)
    
    threshold_gradient = 20
    thresholding_1[thresholding_1 < threshold_gradient] = 0
    
    
    brightness_factor = 4
    plot_image(fig4, ax4, X_zoom, Y_zoom, thresholding_1, brightness_factor)
    ax4.format_coord = format_coord_Z4
    
    
    #%%% Figure 5 : thresholding of raw image + thresholding of gradient image (previous step)
    fig5, ax5 = plt.subplots()
    
    threshold_pixel = 20
    threshold_gradient = threshold_gradient
    
    final1 = np.ones(I_zoom.shape)
    final2 = np.ones(I_zoom.shape)
    
    final1[closing < threshold_pixel] = 0
    final2[pixel_density_gradient_combined < threshold_gradient] = 0
    
    final = final1 + final2
    
    final[final == 1] = 2
    
    # final1 = deepcopy(I_zoom)
    # final2 = deepcopy(pixel_density_gradient_combined)
    
    # final = final1 * final2
    
    brightness_factor = 1
    plot_image(fig5, ax5, X_zoom, Y_zoom, final, brightness_factor)
    
    
    
        
    # order = 0
    
    # if order == 0:
    #     #%%% Open -> close [0]
    #     fig_open, ax_open = plt.subplots()
    #     src = I_zoom
    #     size = 2
    #     kernel_shape = cv2.MORPH_RECT # cv2.MORPH_RECT MORPH_ELLIPSE
    #     kernel = cv2.getStructuringElement(kernel_shape, (size,size))
    #     opening = cv2.morphologyEx(src, cv2.MORPH_OPEN, kernel, iterations = 1)
        
    #     brightness_factor = 4
    #     plot_image(fig_open, ax_open, X_zoom, Y_zoom, opening, brightness_factor)
    #     ax_open.format_coord = format_coord_Zopen
        
        
    #     fig_close, ax_close = plt.subplots()
    #     src = opening
    #     size = 3
    #     kernel_shape = cv2.MORPH_RECT # cv2.MORPH_RECT MORPH_ELLIPSE
    #     kernel = cv2.getStructuringElement(kernel_shape, (size,size))
    #     closing = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel, iterations = 2)
        
    #     brightness_factor = 4
    #     plot_image(fig_close, ax_close, X_zoom, Y_zoom, closing, brightness_factor)
    
    # else:
    
    #     #%%% Close -> open [1]
    #     fig_close, ax_close = plt.subplots()
    #     src = final
    #     size = 3
    #     kernel_shape = cv2.MORPH_RECT # cv2.MORPH_RECT MORPH_ELLIPSE
    #     kernel = cv2.getStructuringElement(kernel_shape, (size,size))
    #     closing = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel, iterations = 1)
        
    #     brightness_factor = 1
    #     plot_image(fig_close, ax_close, X_zoom, Y_zoom, closing, brightness_factor)
        
        
    #     fig_open, ax_open = plt.subplots()
    #     src = closing
    #     size = 2
    #     kernel_shape = cv2.MORPH_ELLIPSE # cv2.MORPH_RECT MORPH_ELLIPSE
    #     kernel = cv2.getStructuringElement(kernel_shape, (size,size))
    #     opening = cv2.morphologyEx(src, cv2.MORPH_OPEN, kernel, iterations = 1)
        
    #     brightness_factor = 1
    #     plot_image(fig_open, ax_open, X_zoom, Y_zoom, opening, brightness_factor)
        
    # #%%% Figure 6 : Gausian blur the raw image
    # fig6, ax6 = plt.subplots()
    
    # src = closing  # Input image array
    # size = 3
    # ksize = (size, size) # Gaussian Kernel Size. [height width]. height and width should be odd and can have different values. If ksize is set to [0 0], then ksize is computed from sigma values.
    # sigmaX = 0 #3 # Kernel standard deviation along X-axis (horizontal direction).
    # blur_gaus = cv2.GaussianBlur(src, ksize, sigmaX)
    
    # brightness_factor = 1
    # plot_image(fig6, ax6, X_zoom, Y_zoom, blur_gaus, brightness_factor)
    
    # #%% Canny
    # fig_canny, ax_canny = plt.subplots()
    
    # data = pixel_density_gradient_combined/pixel_density_gradient_combined.max()
    # data = 255 * data
    
    # src = data.astype(np.uint8)
    # threshold1 = 0.05
    # threshold2 = 0.2
    # apertureSize = 3
    # L2gradient = False
    # edges = cv2.Canny(src, threshold1, threshold2)
    
    # brightness_factor = 8
    # plot_image(fig_canny, ax_canny, X_zoom, Y_zoom, src, brightness_factor)
    
    
    # ax2.set_xlim(ax1.get_xlim())
    # ax2.set_ylim(ax1.get_ylim())
    
    # ax4.set_xlim(ax1.get_xlim())
    # ax4.set_ylim(ax1.get_ylim())
    
    # ax5.set_xlim(ax1.get_xlim())
    # ax5.set_ylim(ax1.get_ylim())
