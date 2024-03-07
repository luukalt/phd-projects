# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 15:02:03 2022

@author: laaltenburg
"""

#%% IMPORT PACKAGES
from premixed_flame_properties import *
from contour_object import *

#%% FLAME OBJECT
class flame():
    
    def __init__(self, 
                 name, 
                 T_lab,
                 p_lab,
                 D_in,
                 Q_air_nL,
                 Q_fuel_nL,
                 phi,
                 H2_percentage,
                 Re_D,
                 u_bulk_measured,
                 run_nr,
                 session_nr,
                 record_name,
                 frame_nrs,
                 scale,
                 start_image,
                 n_images,
                 image_rate,
                 dt,
                 get_contour_procedure_nr,
                 segment_length_mm,
                 window_size,
                 pre_data_folder,
                 post_data_folder,
                 extension,
                 save_image):
        
        self.name = name
        self.T_lab = T_lab
        self.p_lab = p_lab
        self.D_in = D_in
        self.Q_air_nL = Q_air_nL
        self.Q_fuel_nL = Q_fuel_nL
        self.phi = phi
        self.H2_percentage = H2_percentage
        self.Re_D = Re_D
        self.u_bulk_measured = u_bulk_measured
        self.run_nr = run_nr
        self.session_nr = session_nr
        self.record_name = record_name
        self.frame_nrs = frame_nrs
        self.scale = scale
        self.start_image = start_image
        self.n_images = n_images
        self.image_rate = image_rate
        self.dt = dt
        self.get_contour_procedure_nr = get_contour_procedure_nr
        self.segment_length_mm = segment_length_mm
        self.window_size = window_size
        self.pre_data_folder = pre_data_folder
        self.post_data_folder = post_data_folder
        self.extension = extension
        self.save_image = save_image
        
        self.properties = PremixedFlame(phi, H2_percentage, T_lab, p_lab)
        self.segment_length_pixels = segment_length_mm*scale
        self.frames = [frame(self, frame_nr) for frame_nr in frame_nrs]
        


