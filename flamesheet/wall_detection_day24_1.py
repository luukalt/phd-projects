# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 15:18:55 2022

@author: laaltenburg

Detect walls in PIV images of the quasi 2D FlameSheet combustor

"""
#%% Import packages
import os
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
    
    x_screen = 0
    y_screen = 0
    
    dewarped_img_dir = calibration_dir #"Y:/flamesheet_2d_nonreact_day5/Properties/Calibration/DewarpedImages1/Export_01/B0001.tif"
    # dewarped_img_dir = "Y:/flamesheet_2d_react_day7/Properties/Calibration/DewarpedImages1/Export_01/B0001.tif"
    
    dewarped_img_16bit = cv2.imread(dewarped_img_dir, cv2.IMREAD_ANYDEPTH)
    dewarped_img_scaled = cv2.normalize(dewarped_img_16bit, dst=None, alpha=0, beta=2**12-1, norm_type=cv2.NORM_MINMAX)
    # dewarped_img_8bit = (dewarped_img_scaled/256).astype(np.uint8)
    # showInMovedWindow('dewarped calibration image', dewarped_img_8bit, x_screen, y_screen)
    
    
    initial_img_dir = pre_record_dir #"Y:/flamesheet_2d_nonreact_day5/Recording_Date=220915_Time=131136_temp_Conversion/Correction/Reorganize frames/Export/B0001.tif"
    # initial_img_dir = "Y:/flamesheet_2d_react_day7/Recording_Date=220919_Time=142345_temp_Conversion/Correction/Reorganize frames/Export/B0001.tif"
    
    initial_img_16bit = cv2.imread(initial_img_dir, cv2.IMREAD_ANYDEPTH)
    initial_img_scaled = cv2.normalize(initial_img_16bit, dst=None, alpha=0, beta=2**12-1, norm_type=cv2.NORM_MINMAX)
    # img_scaled_8bit = (initial_img_scaled/256).astype(np.uint8)
    
    fig, ax = plt.subplots()
    ax.imshow(initial_img_scaled)
    # plt.axis('off')  # Optional: Turn off axis
    
    img_thres = copy.deepcopy(initial_img_scaled)
    offset = 2**10-1 #15
    img_thres[img_thres < 2**12-offset] = 0
    img_thres[img_thres >= 2**12-offset] = 2**12-1
    img_walls = (img_thres/16).astype(np.uint8)
    # img_walls = cv2.cvtColor(img_walls, cv2.COLOR_GRAY2BGR)
    
    #%% Liner wall
    # fig1, ax1 = plt.subplots()
    x_start_liner = 90
    x_stop_liner = 110
    y_start_liner = 600
    
    wall_coords_liner = []
    
    for j in range(0, 131, 1): #50
        
        y = y_start_liner + j
        line = img_walls[y, x_start_liner:x_stop_liner]
        x_loc_index = []
        
        for i, pixel in enumerate(line):
            
            if pixel == 255:
                x_loc_index.append(x_start_liner + i) 
        
        # Check if x_loc_index is not empty before calculating the mean
        if x_loc_index:
            x_loc_mean = np.mean(x_loc_index)
            if not np.isnan(x_loc_mean):
                wall_coord = (int(x_loc_mean) + 1, y)
                wall_coords_liner.append(wall_coord)
            
            # if len(x_loc_index) > 0:
            #     x_loc_mean = np.mean(x_loc_index)
            
            # if not np.isnan(x_loc_mean):
            #     wall_coord = (int(x_loc_mean)+1, y)
            #     wall_coords_liner.append(wall_coord)
    
    
    pt1_liner = (wall_coords_liner[0][0], 0)
    pt2_liner = wall_coords_liner[-1]
    color = (255, 0, 0)
    
    cv2.line(img_walls, pt1_liner, pt2_liner, color, 1)
    # cv2.line(dewarped_img_8bit, pt1_liner, pt2_liner, color, 1)
    # cv2.line(img_scaled_8bit, pt1_liner, pt2_liner, color, 1)
    
    #%% Core flow left wall
    x_start_core_left = 420
    x_stop_core_left = 430
    y_start_core_left = 350
    
    wall_coords_core_left = []
    
    for j in range(0, 420, 1): #380
        
        y = y_start_core_left + j
        line = img_walls[y, x_start_core_left:x_stop_core_left]
        x_loc_index = []
        
        for i, pixel in enumerate(line):
            
            if pixel == 255:
                x_loc_index.append(x_start_core_left + i) 
            
        
        # Check if x_loc_index is not empty before calculating the mean
        if x_loc_index:
            x_loc_mean = np.mean(x_loc_index)
            if not np.isnan(x_loc_mean):
                wall_coord = (int(x_loc_mean) + 1, y)
                wall_coords_core_left.append(wall_coord)
                
            # if len(x_loc_index) > 0:
            #     x_loc_mean = np.mean(x_loc_index)
            
            # if not np.isnan(x_loc_mean):
            #     wall_coord = (int(x_loc_mean), y)
            #     wall_coords_core_left.append(wall_coord)
        
    pt1_core_left = (wall_coords_core_left[0][0], wall_coords_core_left[0][1])
    pt2_core_left = (wall_coords_core_left[-1][0], wall_coords_core_left[-1][1])
    
    xA, yA = pt1_core_left
    xB, yB = pt2_core_left 
    
    # print((0, yA))
    # print(pt1_core_left)
    # print(pt2_core_left)
    
    a = (yA-yB)/(xA-xB)
    b = yA - a*xA
    x_left, x_right = 0, img_walls.shape[1]-1 
    y_at_xleft = b
    y_at_xright = a*(img_walls.shape[1]-1) + b
    pt1_core_left_mask = (x_left, int(y_at_xleft))
    pt2_core_left_mask = (x_right, int(y_at_xright))
    color = (255, 0, 0)
    
    cv2.line(img_walls, pt1_core_left, pt2_core_left, color, 1)
    # cv2.line(dewarped_img_8bit, pt1_core_left, pt2_core_left, color, 1)
    # cv2.line(img_scaled_8bit, pt1_core_left, pt2_core_left, color, 1)
    
    #%% Core flow right wall
    # x_start_core_right = 540
    # x_stop_core_right = 570
    # y_start_core_right = 0
    
    # wall_coords_core_right = []
    
    # for j in range(0, img_walls.shape[0], 1):
        
    #     y = y_start_core_right + j
    #     line = img_walls[y, x_start_core_right:x_stop_core_right]
    #     x_loc_index = []
        
    #     for i, pixel in enumerate(line):
            
    #         if pixel == 255:
    #             x_loc_index.append(x_start_core_right + i) 
                
    #         x_loc_mean = np.mean(x_loc_index)
            
    #         if not np.isnan(x_loc_mean):
    #             wall_coord = [int(x_loc_mean), y]
    #             wall_coords_core_right.append(wall_coord)
    
    # xA, yA, xB, yB = wall_coords_core_right[0][0], wall_coords_core_right[0][1], wall_coords_core_right[-1][0], wall_coords_core_right[-1][1]      
    # a = (yA-yB)/(xA-xB)
    # b = yA - a*xA
    # y_top, y_bottom = 0, img_walls.shape[0]-1 
    # x_at_ytop = -b/a
    # x_at_ybottom = (img_walls.shape[0]-1 - b)/a
    # pt1_core_right = (int(x_at_ytop), y_top)
    # pt2_core_right = (int(x_at_ybottom), y_bottom)
    # color = (255, 0, 0)
    
    # cv2.line(img_walls, pt1_core_right, pt2_core_right, color, 1)
    # cv2.line(dewarped_img_8bit, pt1_core_right, pt2_core_right, color, 1)
    # cv2.line(img_scaled_8bit, pt1_core_right, pt2_core_right, color, 1)
     
    #%% Dome section
    
    # left
    y_start_dome = 950 #100
    y_stop_dome = 990 #110 #120
    x_start_dome = 90 #870 #850
    
    wall_found = False
    i = 0
    
    while not wall_found:
        
        x = x_start_dome + i
        line = img_walls[y_start_dome:y_stop_dome, x]
        
        for j, pixel in enumerate(line):
            
            if pixel == 255:
                wall_coord_dome_left = (x, y_start_dome + j)
                wall_found = True
                print("bottom dome left found:" + str(wall_found))
                break
            
        i += 1
    
    # mid
    y_start_dome = 970 #945 # 1020
    y_stop_dome = 1010 #955 # 1035
    x_start_dome = 180 #170 # 320
    
    line = img_walls[y_start_dome:y_stop_dome, x_start_dome]
    
    # wall_coord_dome_mid = (180, 1000)
    
    for j, pixel in enumerate(line):
        
        if pixel == 255:
            
            wall_coord_dome_mid = (x_start_dome, y_start_dome + j) 
            
            print("bottom dome right found: True")
            break
        
    # # right
    # x_start_dome = 540 # 580
    # x_stop_dome = 575 # 600
    # y_start_dome = 765 # 800
    
    # wall_found = False
    # j = 0
    # while not wall_found:
        
    #     y = y_start_dome + j
    #     line = img_walls[y, x_start_dome:x_stop_dome]
        
    #     for i, pixel in enumerate(line):
            
    #         if pixel == 255:
    #             wall_coord_dome_right = (x_start_dome + i, y)
    #             wall_found = True
    #             break
    #     j += 1
            
    p_dome_left = wall_coord_dome_left
    p_dome_mid = wall_coord_dome_mid
    p_dome_right = pt2_core_left
    ((cx, cy), radius) = define_circle(p_dome_left, p_dome_mid, p_dome_right)
    
    cv2.circle(img_walls, (int(cx)+1, int(cy)+1), int(radius), color, 1)
    # cv2.circle(dewarped_img_8bit, (int(cx)+1, int(cy)+1), int(radius), color, 1)
    # cv2.circle(img_scaled_8bit, (int(cx)+1, int(cy)+1), int(radius), color, 1)
    
    pt2_liner_custom = (101, 763)
    pt2_liner = pt2_liner_custom
    
    print(f' pt1_core_left: {pt1_core_left}')
    print(f' Dome right: {p_dome_right}')
    print(f' Dome mid: {p_dome_mid}')
    print(f' Dome left: {p_dome_left}')
    print(f' pt2_liner (real): {pt2_liner}')
    # print(f' pt2_liner: (761, 412)')
    print(f' pt1_liner: {pt1_liner}')
    # print(f' pt2_core_left_mask: {pt2_core_left_mask}')
    
    points = [pt1_core_left, p_dome_right, p_dome_mid, p_dome_left, pt2_liner, pt1_liner]
    
    for point in points:
        ax.plot(point[0], point[1], 'rx')
        
    #%% Plot in images
    
    # showInMovedWindow('initial image', img_scaled_8bit, x_screen, y_screen)
    # showInMovedWindow('initial image', img_scaled_8bit, x_screen, y_screen)
    # showInMovedWindow('dewarped calibration image', dewarped_img_8bit, x_screen, y_screen)
    # showInMovedWindow('threshold image', img_walls, x_screen, y_screen)
    
    fig, ax = plt.subplots()
    ax.imshow(img_walls)
    
    # return pt1_liner, pt2_liner, pt1_core_left, pt2_core_left, pt1_core_right, pt2_core_right, cx, cy, radius
    return pt1_liner, pt2_liner, pt1_core_left, pt2_core_left, cx, cy, radius

#%% Main
if __name__ == "__main__":
    
    main_dir = os.path.join('Y:', 'laaltenburg', 'flamesheet_2d_campaign1')
    
    day_nr = '24-1'
    pre_record_name = "Recording_Date=230216_Time=103309"
    
    project_name = "flamesheet_2d_day" + day_nr
    project_dir = os.path.join(main_dir, project_name)
    
    calibration_csv_file =  os.path.join(project_dir, 'Properties', 'Calibration', 'DewarpedImages1', 'Export', 'B0001.csv')
    calibration_tif_file = os.path.join(project_dir, 'Properties', 'Calibration', 'DewarpedImages1', 'Export_01', 'B0001.tif')

    pre_record_raw_file = os.path.join(project_dir, pre_record_name, 'Reorganize frames', 'Export', 'B0001.tif')
    pre_record_correction_file = os.path.join(project_dir, pre_record_name, 'ImageCorrection', 'Reorganize frames', 'Export', 'B0001.tif')

    wall_detection(calibration_tif_file, pre_record_correction_file)
    
    
    

