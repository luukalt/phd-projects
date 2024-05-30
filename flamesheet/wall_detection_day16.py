# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 15:18:55 2022

@author: laaltenburg

Detect walls in PIV images of the quasi 2D FlameSheet combustor

"""
#%% Import packages
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import cv2
import copy

#%% Start
cv2.destroyAllWindows()
plt.close("all")

default_fig_dim = matplotlib.rcParams["figure.figsize"]
cmap = cm.viridis

# Define blue, green and red color in BGR
color_green = (0, 255, 0)
color_blue = (255, 0, 0)
color_red = (0, 0, 255)

#%% Main functions
def showInMovedWindow(winname, img, x, y):
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, x, y)   # Move it to (x,y)
    # img = cv2.resize(img, (512, 512))                # Resize image
    cv2.imshow(winname, img)

def define_circle(p1, p2, p3):
    """
    Returns the center and radius of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).
    """
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    if abs(det) < 1.0e-6:
        return (None, np.inf)

    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return ((cx, cy), radius)

def wall_detection(calibration_dir, pre_record_dir):
    
    image_nr = 1
    extension = ".tif"
    
    x_screen = 0
    y_screen = 0
    
    #%% Load image
    if image_nr < 10:
        image_prefix = 'B000'
    if 10 <= image_nr <= 100:
        image_prefix = 'B00'
    if 100 <= image_nr <= 1000:
        image_prefix = 'B0' 
    if 1000 <= image_nr <= 10000:
        image_prefix = 'B'   
    
    dewarped_img_dir = calibration_dir #"Y:/flamesheet_2d_nonreact_day5/Properties/Calibration/DewarpedImages1/Export_01/B0001.tif"
    # dewarped_img_dir = "Y:/flamesheet_2d_react_day7/Properties/Calibration/DewarpedImages1/Export_01/B0001.tif"
    
    dewarped_img_16bit = cv2.imread(dewarped_img_dir, cv2.IMREAD_ANYDEPTH)
    dewarped_img_scaled = cv2.normalize(dewarped_img_16bit, dst=None, alpha=0, beta=2**16-1, norm_type=cv2.NORM_MINMAX)
    dewarped_img_8bit = (dewarped_img_scaled/256).astype(np.uint8)
    
    initial_img_dir = pre_record_dir #"Y:/flamesheet_2d_nonreact_day5/Recording_Date=220915_Time=131136_temp_Conversion/Correction/Reorganize frames/Export/B0001.tif"
    # initial_img_dir = "Y:/flamesheet_2d_react_day7/Recording_Date=220919_Time=142345_temp_Conversion/Correction/Reorganize frames/Export/B0001.tif"
    
    initial_img_16bit = cv2.imread(initial_img_dir, cv2.IMREAD_ANYDEPTH)
    initial_img_scaled = cv2.normalize(initial_img_16bit, dst=None, alpha=0, beta=2**16-1, norm_type=cv2.NORM_MINMAX)
    img_scaled_8bit = (initial_img_scaled/256).astype(np.uint8)
    
    img_thres = copy.deepcopy(initial_img_scaled)
    offset = 2**15-1 #15
    img_thres[img_thres < 2**16-offset] = 0
    img_thres[img_thres >= 2**16-offset] = 2**16-1
    img_walls = (img_thres/256).astype(np.uint8)
    # img_walls = cv2.cvtColor(img_walls, cv2.COLOR_GRAY2BGR)
    
    #%% Liner wall
    # fig1, ax1 = plt.subplots()
    x_start_liner = 75
    x_stop_liner = x_start_liner + 30
    y_start_liner = 650
    
    wall_coords_liner = []
    
    for j in range(0, 120, 1): #50
        
        y = y_start_liner + j
        line = img_walls[y, x_start_liner:x_stop_liner]
        x_loc_index = []
        
        for i, pixel in enumerate(line):
            
            if pixel == 255:
                x_loc_index.append(x_start_liner + i) 
                
            x_loc_mean = np.mean(x_loc_index)
            
            if not np.isnan(x_loc_mean):
                wall_coord = (int(x_loc_mean)+1, y)
                wall_coords_liner.append(wall_coord)
    
    
    pt1_liner = (wall_coords_liner[0][0], 0)
    pt2_liner = wall_coords_liner[-1]
    color = (255, 0, 0)
    
    cv2.line(img_walls, pt1_liner, pt2_liner, color, 1)
    cv2.line(dewarped_img_8bit, pt1_liner, pt2_liner, color, 1)
    cv2.line(img_scaled_8bit, pt1_liner, pt2_liner, color, 1)
    
    #%% Core flow left wall
    x_start_core_left = 430
    x_stop_core_left = x_start_core_left + 30
    y_start_core_left = 380
    
    wall_coords_core_left = []
    
    for j in range(0, 384, 1): #380
        
        y = y_start_core_left + j
        line = img_walls[y, x_start_core_left:x_stop_core_left]
        x_loc_index = []
        
        for i, pixel in enumerate(line):
            
            if pixel == 255:
                x_loc_index.append(x_start_core_left + i) 
                
            x_loc_mean = np.mean(x_loc_index)
            
            if not np.isnan(x_loc_mean):
                wall_coord = (int(x_loc_mean), y)
                wall_coords_core_left.append(wall_coord)
        
    pt1_core_left = (wall_coords_core_left[0][0], wall_coords_core_left[0][1])
    pt2_core_left = (wall_coords_core_left[-1][0], wall_coords_core_left[-1][1])
    
    # print(pt2_core_left)
    
    color = (255, 0, 0)
    
    cv2.line(img_walls, pt1_core_left, pt2_core_left, color, 1)
    cv2.line(dewarped_img_8bit, pt1_core_left, pt2_core_left, color, 1)
    cv2.line(img_scaled_8bit, pt1_core_left, pt2_core_left, color, 1)
    
    #%% Core flow right wall
    x_start_core_right = 600
    x_stop_core_right = x_start_core_right + 30
    y_start_core_right = 0
    
    wall_coords_core_right = []
    
    for j in range(0, img_walls.shape[0], 1):
        
        y = y_start_core_right + j
        line = img_walls[y, x_start_core_right:x_stop_core_right]
        x_loc_index = []
        
        for i, pixel in enumerate(line):
            
            if pixel == 255:
                x_loc_index.append(x_start_core_right + i) 
                
            x_loc_mean = np.mean(x_loc_index)
            
            if not np.isnan(x_loc_mean):
                wall_coord = [int(x_loc_mean), y]
                wall_coords_core_right.append(wall_coord)
    
    xA, yA, xB, yB = wall_coords_core_right[0][0], wall_coords_core_right[0][1], wall_coords_core_right[-1][0], wall_coords_core_right[-1][1]      
    a = (yA-yB)/(xA-xB)
    b = yA - a*xA
    y_top, y_bottom = 0, img_walls.shape[0]-1 
    x_at_ytop = -b/a
    x_at_ybottom = (img_walls.shape[0]-1 - b)/a
    pt1_core_right = (int(x_at_ytop), y_top)
    pt2_core_right = (int(x_at_ybottom), y_bottom)
    color = (255, 0, 0)
    
    cv2.line(img_walls, pt1_core_right, pt2_core_right, color, 1)
    cv2.line(dewarped_img_8bit, pt1_core_right, pt2_core_right, color, 1)
    cv2.line(img_scaled_8bit, pt1_core_right, pt2_core_right, color, 1)
     
    #%% Dome section
    
    # left
    y_start_dome = 950 #100
    y_stop_dome = 990 #110 #120
    x_start_dome = 40 #870 #850
    
    wall_found = False
    i = 0
    
    while not wall_found:
        
        x = x_start_dome + i
        line = img_walls[y_start_dome:y_stop_dome, x]
        
        for j, pixel in enumerate(line):
            
            if pixel == 255:
                wall_coord_dome_left = (x, y_start_dome + j)
                wall_found = True
                
                break
            
        i += 1
    
    # mid
    y_start_dome = 1000 #945 # 1020
    y_stop_dome = 1030 #955 # 1035
    x_start_dome = 90 #170 # 320
    
    line = img_walls[y_start_dome:y_stop_dome, x_start_dome]
    
    for j, pixel in enumerate(line):
        
        if pixel == 255:
            
            wall_coord_dome_mid = (x_start_dome, y_start_dome + j) 
            break
            
    p_dome_left = wall_coord_dome_left
    p_dome_mid = wall_coord_dome_mid
    p_dome_right = pt2_core_left
    ((cx, cy), radius) = define_circle(p_dome_left, p_dome_mid, p_dome_right)
    
    cv2.circle(img_walls, (int(cx)+1, int(cy)+1), int(radius), color, 1)
    cv2.circle(dewarped_img_8bit, (int(cx)+1, int(cy)+1), int(radius), color, 1)
    cv2.circle(img_scaled_8bit, (int(cx)+1, int(cy)+1), int(radius), color, 1)
    
    #%% Plot in images
    
    showInMovedWindow('initial image', img_scaled_8bit, x_screen, y_screen)
    showInMovedWindow('dewarped calibration image', dewarped_img_8bit, x_screen, y_screen)
    showInMovedWindow('threshold image', img_walls, x_screen, y_screen)

    return pt1_liner, pt2_liner, pt1_core_left, pt2_core_left, pt1_core_right, pt2_core_right, cx, cy, radius

#%% Main
if __name__ == "__main__":
    
    main_dir = "Y:/"
    
    # H2% = 0, phi = 0, Re_H = 2500, image_rate = 0.2
    # project_name = "flamesheet_2d_day5-1"
    # pre_record_name = "Recording_Date=220915_Time=131136"
    
    
    # H2% = 0, phi = 0.7, Re_H = 2500, image_rate = 0.2
    
    
    # H2% = 40, phi = 0.6, Re_H = 5000, image_rate = 0.2

    # H2% = 60, phi = 0.5, Re_H = 5500, image_rate = 0.2

    # H2% = 80, phi = 0.4, Re_H = 6000, image_rate = 0.2

    # H2% = 100, phi = 0.3, Re_H = 7000, image_rate = 0.2
    project_name = "flamesheet_2d_day16"
    pre_record_name = "Recording_Date=221223_Time=131437"
    
    
    project_dir = main_dir + project_name
    
    calibration_tif_dir = project_dir + "/Properties/Calibration/DewarpedImages1/Export_01/B0001.tif"
    pre_record_correction_dir = project_dir + "/" + pre_record_name + "/Correction/Reorganize frames/Export/B0001.tif" 
    
    wall_detection(calibration_tif_dir, pre_record_correction_dir)
    
    
    

