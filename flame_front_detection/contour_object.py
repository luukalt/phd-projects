# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 15:00:34 2022

@author: laaltenburg

Objects for frame front analysis
"""

#%% IMPORT PACKAGES
import os
import numpy as np
from tqdm import tqdm
from contour_properties import *
from get_contour_procedures import *

#%% FRAME OBJECT
# Data of contours of frames of the specified number of images (image pairs) of a flame recording
class frame:
    
    def __init__(self, flame, frame_nr):
        
        cwd = os.getcwd()
        pre_data_path = os.path.join(cwd, flame.pre_data_folder, flame.name, f'session_{flame.session_nr:03}' , flame.record_name, 'Correction', 'Resize', f'Frame{frame_nr}', 'Export_01')
        post_data_path = os.path.join(cwd, flame.post_data_folder, flame.name, f'session_{flame.session_nr:03}', f'{flame.record_name}_Frame{frame_nr}')

        # Create a new directory because it does not exist 
        if not os.path.exists(post_data_path):
          
          os.makedirs(post_data_path)
        
        # self.contour_data.append(contour_data(flame, pre_data_path, post_data_path))
        self.contour_data = contour_data(flame, pre_data_path, post_data_path)
            
#%% CONTOUR DATA OBJECT
class contour_data:
    
    def __init__(self, flame, pre_data_path, post_data_path):
        
        window_size = flame.window_size
        segment_length_pixels = flame.segment_length_pixels
        extension = flame.extension
        save_image = flame.save_image
        toggle_plot = False
        
        procedure_nr = flame.get_contour_procedure_nr
        image_nrs = list(range(flame.start_image, flame.start_image + flame.n_images))
        
        self.contours = []
        self.list_of_contour_lengths_in_pixels= []
        # self.accepted_images = []
        
        self.segmented_contours = []
        self.slopes_of_segmented_contours = []
        self.slope_changes_of_segmented_contours = []
        
        for image_nr in tqdm(image_nrs):
            
            shape, contour, contour_length_pixels = get_contour_data(procedure_nr, window_size, pre_data_path, post_data_path, image_nr, extension, toggle_plot, save_image)
            
            contour_x = contour[:,0,0]
            contour_y = contour[:,0,1]
            
            # Shape of image (height, width)
            resolution_x = shape[1]
            resolution_y = shape[0]
            
        
            self.contours.append(contour)
            self.list_of_contour_lengths_in_pixels.append(contour_length_pixels)
            # self.accepted_images.append(image_nr)
            
            # Segmented contour
            segmented_contour_x, segmented_contour_y, segmented_contour = contour_segmentation(contour, segment_length_pixels)
            self.segmented_contours.append(segmented_contour)
            
            # Slopes of contour
            slopes_of_segmented_contour = slope(segmented_contour)
            self.slopes_of_segmented_contours.append(slopes_of_segmented_contour)
            
            # Change of slopes of contour
            slope_changes_of_segmented_contour = slope_change(segmented_contour)
            self.slope_changes_of_segmented_contours.append(slope_changes_of_segmented_contour)
            
        x = np.linspace(0, shape[1] - 1, shape[1])
        y = np.linspace(0, shape[0] - 1, shape[0])
        
        # Distribution of the flame front contours
        self.X, self.Y = np.meshgrid(x, y)
        self.contour_distribution = np.zeros(shape)
        self.frame_resolution = shape

        for contour in self.contours:
            
            for coord in contour:
                
                x_coord = coord[0][0]
                y_coord = coord[0][1]
                
                self.contour_distribution[y_coord][x_coord] += 1
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
                