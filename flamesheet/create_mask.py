# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 14:46:03 2022

@author: laaltenburg
"""

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

def wall_detection_pre(pre_record_dir, day_nr):
    
    # image_nr = 1
    # extension = ".tif"
    
    x_screen = 0
    y_screen = 0
    
    # %% Load image
    # if image_nr < 10:
    #     image_prefix = 'B000'
    # if 10 <= image_nr <= 100:
    #     image_prefix = 'B00'
    # if 100 <= image_nr <= 1000:
    #     image_prefix = 'B0' 
    # if 1000 <= image_nr <= 10000:
    #     image_prefix = 'B'   
    

    initial_img_dir = pre_record_dir #"Y:/flamesheet_2d_nonreact_day5/Recording_Date=220915_Time=131136_temp_Conversion/Correction/Reorganize frames/Export/B0001.tif"
    # initial_img_dir = "Y:/flamesheet_2d_react_day7/Recording_Date=220919_Time=142345_temp_Conversion/Correction/Reorganize frames/Export/B0001.tif"
    
    initial_img_16bit = cv2.imread(initial_img_dir, cv2.IMREAD_ANYDEPTH)
    initial_img_scaled = cv2.normalize(initial_img_16bit, dst=None, alpha=0, beta=2**12-1, norm_type=cv2.NORM_MINMAX)
    # img_scaled_8bit = (initial_img_scaled/256).astype(np.uint8)
    
    # showInMovedWindow('initial image', img_scaled_8bit, x_screen, y_screen)
    
    fig, ax = plt.subplots()
    ax.imshow(initial_img_scaled)
    
    # plt.axis('off')  # Optional: Turn off axis
    # plt.show()

    img_thres = copy.deepcopy(initial_img_scaled)
    offset = 2**10-1 #15
    img_thres[img_thres < 2**12-offset] = 0
    img_thres[img_thres >= 2**12-offset] = 2**12-1
    img_walls = (img_thres/16).astype(np.uint8)
    # img_walls = img_thres
    
    #%% Liner wall
    # fig1, ax1 = plt.subplots()
    x_start_liner = 0
    y_start_liner = 400
    y_stop_liner = 500
    
    wall_coords_liner = []
    
    for i in range(0, 731, 1): #50
        
        x = x_start_liner + i
        line = img_walls[y_start_liner:y_stop_liner, x]
        y_loc_index = []
        
        for j, pixel in enumerate(line):
            
            if pixel == 255:
                y_loc_index.append(y_start_liner + j) 
                
            y_loc_mean = np.mean(y_loc_index)
            
            if not np.isnan(y_loc_mean):
                wall_coord = (x, int(y_loc_mean))
                wall_coords_liner.append(wall_coord)
    
    pt1_liner = (0, wall_coords_liner[0][1])
    pt2_liner = (wall_coords_liner[-1][0], wall_coords_liner[-1][1]) 
    color = (255, 0, 0)
    
    # print(pt1_liner)
    
    cv2.line(img_walls, pt1_liner, pt2_liner, color, 1)
    # cv2.line(img_scaled_8bit, pt1_liner, pt2_liner, color, 1)
    
    #%% Core flow left wall
    x_start_core_left = 360
    y_start_core_left = 75
    y_stop_core_left = 90
    
    wall_coords_core_left = []
    
    for i in range(0, 500, 1): #380
        
        x = x_start_core_left + i
        line = img_walls[y_start_core_left:y_stop_core_left, x]
        y_loc_index = []
        
        for j, pixel in enumerate(line):
            
            if pixel == 255:
                y_loc_index.append(y_start_core_left + j) 
                
            y_loc_mean = np.mean(y_loc_index)
            
            if not np.isnan(y_loc_mean):
                wall_coord = (x, int(y_loc_mean))
                wall_coords_core_left.append(wall_coord)
    
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
    
    # print(img_walls.shape)
    # cv2.line(img_walls, pt1_core_left_mask, pt2_core_left_mask, color, 1)
    # cv2.line(img_scaled_8bit, pt1_core_left_mask, pt2_core_left_mask, color, 1)
    
    # cv2.line(img_walls, pt1_core_left, pt2_core_left, color, 1)
    # cv2.line(img_scaled_8bit, pt1_core_left, pt2_core_left, color, 1)
    
    #%% Core flow right wall
    # x_start_core_right = 0
    # y_start_core_right = 70
    # y_stop_core_right = 90
    
    # wall_coords_core_right = []
    
    # for i in range(0, img_walls.shape[1], 1):
        
    #     x = x_start_core_right + i
    #     line = img_walls[y_start_core_right:y_stop_core_right, x]
    #     y_loc_index = []
        
    #     for j, pixel in enumerate(line):
            
    #         if pixel == 255:
    #             y_loc_index.append(y_start_core_right + j) 
                
    #         y_loc_mean = np.mean(y_loc_index)
            
    #         if not np.isnan(y_loc_mean):
    #             wall_coord = [x, int(y_loc_mean)]
    #             wall_coords_core_right.append(wall_coord)
    
    # xA, yA, xB, yB = wall_coords_core_right[0][0], wall_coords_core_right[0][1], wall_coords_core_right[-1][0], wall_coords_core_right[-1][1]      
    # a = (yA-yB)/(xA-xB)
    # b = -yA+a*xA
    # x_at_ytop = b/a
    # x_at_ybottom = (img_walls.shape[1]-1 + b)/a
    # pt1_core_right = (int(x_at_ytop), 0)
    # pt2_core_right = (int(x_at_ybottom), img_walls.shape[1]-1)
    # color = (255, 0, 0)
    
    # cv2.line(img_walls, pt1_core_right, pt2_core_right, color, 1)
    # cv2.line(img_scaled_8bit, pt1_core_right, pt2_core_right, color, 1)
     
    #%% Dome section
    
    # bottom
    x_start_dome = 800
    x_stop_dome = 1000
    y_start_dome = 420
    
    wall_found = False
    j = 0
    
    while not wall_found:
        
        y = y_start_dome + j
        line = img_walls[y, x_start_dome:x_stop_dome]
        
        for i, pixel in enumerate(line):
            
            if pixel == 255:
                wall_coord_dome_left = (x_start_dome + i, y)
                wall_found = True
                # print("bottom dome found:" + str(wall_found))
                break
            
        j += 1
        
    # mid
    x_start_dome = 800
    x_stop_dome = 1000
    y_start_dome = 300
    
    line = img_walls[y_start_dome, x_start_dome:x_stop_dome]
    
    for i, pixel in enumerate(line):
        
        if pixel == 255:
            
            wall_coord_dome_mid = (x_start_dome + i, y_start_dome)
            break
        
            
    p_dome_left = wall_coord_dome_left
    p_dome_mid = wall_coord_dome_mid
    p_dome_right = pt2_core_left
    ((cx, cy), radius) = define_circle(p_dome_left, p_dome_mid, p_dome_right)
    
    pt2_liner_custom = (762, 410) # day 23-2
    # pt2_liner_custom = (761, 412) # day 24-1
    pt2_liner = pt2_liner_custom
    
    print(f' pt1_core_left: {pt1_core_left}')
    print(f' Dome right: {p_dome_right}')
    print(f' Dome mid: {p_dome_mid}')
    print(f' Dome left: {p_dome_left}')
    print(f' pt2_liner (real): {pt2_liner}')
    # print(f' pt2_liner: (761, 412)')
    print(f' pt1_liner: {pt1_liner}')
    print(f' pt1_core_left_mask: {pt1_core_left_mask}')
    # print(f' pt2_core_left_mask: {pt2_core_left_mask}')
    
    
    circle_image = np.zeros((1024,1024), np.uint8)
    circle_pixels = cv2.circle(circle_image, (int(cx)+1, int(cy)+1), int(radius), color, 1)
    circle_pixels_y, circle_pixels_x = np.nonzero(circle_pixels)
    
    dome_partial_coords = []
    
    string = "Points="
    string = string + str(pt1_core_left_mask[0]) + " " + str(pt1_core_left_mask[1]) + " "
    
    string_dome = ""
    for x, y in zip(circle_pixels_x, circle_pixels_y):
        if y <= pt2_liner[1] and x >= pt2_core_left[0]:
            dome_partial_coords.append([x, y])
            string = string + str(x) + " " + str(y) + " "
            string_dome = string_dome + str(x) + " " + str(y) + " "
    
    string = string + str(pt2_liner[0]) + " " + str(pt2_liner[1]) + " "
    string = string + str(pt1_liner[0]) + " " + str(pt1_liner[1]) + " "
    string = string + str(pt1_core_left_mask[0]) + " " + str(pt1_core_left_mask[1])
    
    # print(string)
    
    # print(string_dome)
    # cv2.polylines(img_scaled_8bit, [np.array(dome_partial_coords, dtype=np.int32)], False, color, 1)
    cv2.polylines(img_walls, [np.array(dome_partial_coords, dtype=np.int32)], False, color, 1)
    #%% Plot in images
    
    # showInMovedWindow('initial image', img_scaled_8bit, x_screen, y_screen)
    # showInMovedWindow('threshold image', img_walls, x_screen, y_screen)
    
    # plt.imshow(img_walls)
    # plt.axis('off')  # Optional: Turn off axis
    # plt.show()
    
    points = [pt1_core_left, p_dome_right, p_dome_mid, p_dome_left, pt2_liner, pt1_liner]
    
    for point in points:
        ax.plot(point[0], point[1], 'rx')
    
    filename = "mask_coordinates_day" + day_nr + ".txt"
    text_file = open(filename, "w")
    text_file.write(string)
    text_file.close()
    # return mask_string

    # return pt1_liner, pt2_liner, pt1_core_left, pt2_core_left, pt1_core_right, pt2_core_right, cx, cy, radius

#%% Main
if __name__ == "__main__":
    
    main_dir = os.path.join('Y:', 'laaltenburg', 'flamesheet_2d_campaign1')
    
    # H2% = 0, phi = 0, Re_H = 2500, image_rate = 0.2
    # day_nr = "5-1"
    # pre_record_name = "Recording_Date=220915_Time=131136"
    
    # H2% = 0, phi = 0.7, Re_H = 2500, image_rate = 0.2
    # day_nr = "5-2"
    # pre_record_name = "Recording_Date=220915_Time=145548"
    
    # # H2% = 40, phi = 0.6, Re_H = 5000, image_rate = 0.2
    # day_nr = "5-2"
    # pre_record_name = "Recording_Date=220915_Time=145548"

    # # H2% = 60, phi = 0.5, Re_H = 5500, image_rate = 0.2
    # day_nr = "5-1"
    # pre_record_name = "Recording_Date=220915_Time=131136"

    # # H2% = 80, phi = 0.4, Re_H = 6000, image_rate = 0.2
    # day_nr = "5-1"
    # pre_record_name = "Recording_Date=220915_Time=131136"

    # # H2% = 100, phi = 0.3, Re_H = 7000, image_rate = 0.2
    # day_nr = "14"
    # pre_record_name = "Recording_Date=221118_Time=112139"
    
    # test99: H2% = 100, phi = 0.35, Re_H = 5000, image_rate = 0.2 & test101: H2% = 100, phi = 0.35, Re_H = 7000, image_rate = 0.2
    # day_nr = "23-2"
    # pre_record_name = "Recording_Date=230215_Time=153306"
    
    # # test103: H2% = 0, phi = 0, Re_H = 7000, image_rate = 0.2 & test105: H2% = 0, phi = 0, Re_H = 5000, image_rate = 0.2
    day_nr = "24-1"
    pre_record_name = "Recording_Date=230216_Time=103309"
    
    
    
    project_name = "flamesheet_2d_day" + day_nr
    project_dir = os.path.join(main_dir, project_name)
    
    pre_record_tif_file = os.path.join(project_dir, pre_record_name, "Reorganize frames", "Export", "B0001.tif")
    
    # pt1_liner, pt2_liner, pt1_core_left, pt2_core_left, pt1_core_right, pt2_core_right, cx, cy, radius = wall_detection_pre(pre_record_tif_file, day_nr)
    
    wall_detection_pre(pre_record_tif_file, day_nr)
    
    

