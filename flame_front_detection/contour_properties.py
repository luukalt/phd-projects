# -*- coding: utf-8 -*-
"""
Calculate contour properties

"""
#%% IMPORT PACKAGES
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

#%% SEGMENT ANGLE
def segment_properties(dy, dx):
    
    L = np.sqrt(dx**2 + dy**2)
    
    if dx == 0 and dy > 0:
        segment_angle = np.pi/2
    elif dx == 0 and dy < 0:
        segment_angle = -np.pi/2
    elif dx == 0 and dy == 0:
        segment_angle = 0
    else:
        segment_angle = np.arctan(dy/dx)
    
    return L, segment_angle

#%% SEGMENT MIDPOINT
# def segment_midpoint(x, y):
    
#     for i in range(len(x)-1):
        
#         x_A, y_A, x_B, y_B  = x[i], y[i], x[i+1], y[i+1]
        
#         x_mid = (x_A + x_B) / 2
#         y_mid = (y_A + y_B) / 2
    
#     return x_mid, y_mid

#%% FUNCTION: CONTOUR SEGMENTATION
def contour_segmentation(contour, segment_length_pixels):
    
    x, y = contour[:,0,0], contour[:,0,1]
    
    # Linear length on the line
    distance = np.cumsum(np.sqrt( np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2 ))
    distance = distance/distance[-1]
    
    # print(len(distance))
    # fx, fy = interp1d(distance, flatten(contour_x)), interp1d(distance, flatten(contour_y))
    fx, fy = interp1d(distance, x), interp1d(distance, y)
    
    n_interp_points = np.linspace(0, 1, 10001)
    x_interp, y_interp = fx(n_interp_points), fy(n_interp_points)

    x_segment = [x_interp[0]]
    y_segment = [y_interp[0]]
    
    i = 0
    counter = 0
    
    while i < len(x_interp):
        
        counter += 1
        
        for j in range(i, len(x_interp)):
            
            segment_distance = np.sqrt((x_interp[j] - x_interp[i])**2 + (y_interp[j] - y_interp[i])**2)
            
            if segment_distance >= segment_length_pixels:
                
                x_segment.append(x_interp[j])
                y_segment.append(y_interp[j])
                break
        
        i = j
        
        if counter > len(x_interp):
            break
    
    segmented_contour_x = np.array(x_segment)
    segmented_contour_y = np.array(y_segment)
    
    # Combine the x and y coordinates into a single array of shape (n_coords, 2)
    segmented_coords = np.array([segmented_contour_x, segmented_contour_y]).T

    # Create a new array of shape (n_coords, 1, 2) and assign the coordinates to it
    segmented_contour = np.zeros((len(segmented_contour_x), 1, 2))
    segmented_contour[:,0,:] = segmented_coords
    
    
    return segmented_contour_x, segmented_contour_y, segmented_contour
    
#%% FUNCTION: SLOPE OF CONTOUR SEGMENTS 
def slope(contour):
    
    norm = np.pi
    
    slopes_of_segmented_contour = []
    
    x, y = contour[:,0,0], contour[:,0,1]
    
    # Flame front direction (tangent)
    for i in range(0, len(x) - 1):
        
        dx = x[i+1] - x[i]
        dy = y[i+1] - y[i]
        
        alpha_tan = np.arctan2(dy, dx)/norm
        slopes_of_segmented_contour.append(alpha_tan)
    
    return slopes_of_segmented_contour

#%% FUNCTION: CHANGE OF SLOPE OF CONTOUR SEGMENTS
def slope_change(contour):
    
    n_digits = 5
    
    n_digits_alpha = 3
    
    norm = np.pi
    
    slope_changes_of_segmented_contour = []
    
    x, y = contour[:,0,0], contour[:,0,1]
    
    for i in range(1, len(x) - 1):
        
        alphas = []
        
        v1_x = x[i] - x[i-1]
        v1_y = y[i] - y[i-1]
        v1 = np.array([v1_x, v1_y])
        
        v2_x = x[i+1] - x[i]
        v2_y = y[i+1] - y[i]
        v2 = np.array([v2_x, v2_y])
        
        v1_magnitude = np.linalg.norm(v1)
        v2_magnitude = np.linalg.norm(v2)
        
        C = (v1_x*v2_x + v1_y*v2_y)/(v1_magnitude*v2_magnitude)
        S = (v1_x*v2_y - v1_y*v2_x)/(v1_magnitude*v2_magnitude)
        
        # Rounding C to prevent error when calculating np.arccos(C)
        C = np.round(C, n_digits)
        S = np.round(S, n_digits)
        
        try:
            alpha_1 = np.arccos(C)
            alpha_2 = -np.arccos(C)
            alpha_3 = np.arcsin(S)
            alpha_4 = np.pi - np.arcsin(S)
        except Warning as warn:
            print(f"A warning occurred: {str(warn)}")
            print(C, S)
        
        alphas = [np.round(angle, n_digits_alpha) for angle in np.array([alpha_1, alpha_2, alpha_3, alpha_4])/norm]

        alpha = np.round(alphas[0], n_digits_alpha)
        
        if alphas.count(alpha) < 2:
            
            alpha = alphas[1]
        
        slope_changes_of_segmented_contour.append(alpha)
    
    return slope_changes_of_segmented_contour

#%% FUNCTION: CURVATURE OF CONTOUR SEGMENTS

def radius_of_curvature(p1, p2, p3):
    
    x_A, y_A = p1
    x_B, y_B = p2
    x_C, y_C = p3

    # Calculate slopes of segments
    if x_B != x_A:
        a1 = (y_B - y_A) / (x_B - x_A)
    else:
        a1 = np.inf
    if x_C != x_B:
        a2 = (y_C - y_B) / (x_C - x_B)
    else:
        a2 = np.inf

    # Find the slopes of the perpendicular lines
    a1_perp = -1.0 / a1
    a2_perp = -1.0 / a2

    # Calculate the intersection point of the perpendicular lines
    if a1_perp == np.inf or a2_perp == np.inf:
        x_intersect = x_A if a1_perp == np.inf else x_C
        y_intersect = y_A if a2_perp == np.inf else y_C
    else:
        b1 = y_A - a1_perp * x_A
        b2 = y_C - a2_perp * x_C
        x_intersect = (b2 - b1) / (a1_perp - a2_perp)
        y_intersect = a1_perp * x_intersect + b1
    
    r = np.sqrt((x_A-x_intersect)**2 + (y_A-y_intersect)**2)
    
    p_intersect = (x_intersect, y_intersect)
    angle = angle_between_vectors(p_intersect, p1, p3)
    
    return x_intersect, y_intersect, r, angle
  
    
def angle_between_vectors(origin, point1, point2):
    vec1 = np.array(point1) - np.array(origin)
    vec2 = np.array(point2) - np.array(origin)
    dot_product = np.dot(vec1, vec2)
    magnitude_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    angle_in_radians = np.arccos(dot_product / magnitude_product)
    angle_in_degrees = np.degrees(angle_in_radians)
    return angle_in_radians

def draw_radius_of_curvature(ax, color, cx, cy, radius):
    
    circle = plt.Circle((cx, cy), radius, edgecolor=color, fill=False)
    ax.add_artist(circle)









    
    
    
    
    
  
    
    
    
    
    
    

    