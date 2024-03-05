# -*- coding: utf-8 -*-
"""
Information of experiments
"""

#%% IMPORT PACKAGES
import os
import numpy as np
from matplotlib import pyplot as plt
import cv2
import copy
from flame_object import *
from contour_properties import *
import json

#%% START
cv2.destroyAllWindows()
plt.close('all')

print('Batch Job Running!')

#%% FLAME INFO
name = 'react_h0_f2700_hs_record1'
T_lab = 273.15 + 23.3                                           # units: K   
p_lab = 102.12e3                                                # units: Pa
D_in = 25.16                                                    # units: mm
Q_air_nL = None                                                 # units: nL.min^{-1}
Q_fuel_nL = None                                                # units: nL.min^{-1}
phi = 1.0                                                       # units: -
H2_percentage = 0                                               # units: %
Re_D = 2700
u_bulk_measured = 1.66                                          # units: m.s^{-1}   
run_nr = 21
session_nr = 3                                                    
record_name = 'Recording_Date=230622_Time=105158_01'            
frame_nrs = [0, 1]                                              
scale = 15.11                                                   # units: pixels.mm^{-1}
start_image = 1                                                 
n_images = 3300                                                    
image_rate = 2000                                               # units: Hz
dt = 250                                                        # units: mu.s
get_contour_procedure_nr = 2                                    
segment_length_mm = 1                                           # units: mm
window_size = int(np.ceil(2*scale) // 2 * 2 + 1)                # Get closest odd number units: pixels
pre_data_folder = 'pre_data'                                    
post_data_folder = 'post_data'                                  
extension = '.tif'                                              
save_image = True						

#%% BUILD FLAME OBJECT
flame_info = flame(name, 
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
                save_image)

#%% SAVE FLAME OBJECT TO PICKLE FILE

flame_pickle = flame_info

# Open a file and use dump()
fname = f"{flame_pickle.name}_segment_length_{flame_pickle.segment_length_mm}mm_wsize_{flame_pickle.window_size}pixels"

spydata_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spydata')
if not os.path.isdir(spydata_dir):
    os.mkdir(spydata_dir)

with open(os.path.join(spydata_dir, fname + '.pkl'), 'wb') as f:
    pickle.dump(flame_pickle, f)

print('Batch Job Completed!')

# def to_dict(x):
    # if    hasattr(x,'shape'):
        # x = x.tolist()
    
    # elif  type(x) == dict:
        # for k,v in x.items():
            # x[k] = to_dict(v)
    
    # elif type(x) in [list, tuple]:
        # x = [to_dict(v) for v in x]
    
    # elif hasattr(x,'__dict__'):
        # x = to_dict(x.__dict__)
    
    # elif '<cantera' in str(x):
        # x = str(x)

    # elif '<built-in' in str(x):
        # x = str(x)

    # return x

# with open(os.path.join(spydata_dir, fname + '.json'), 'w') as f:
    # f.write(str(to_dict(flame_pickle))+'\n')

#%% LOAD FLAME OBJECT FROM PICKLE FILE AND APPEND ATTRIBUTE AND SAVE OBJECT TO PICKLE
# flame_nr = 4
# filtered_data = 0 # 1:True, 0:False
# segment_length_mm = 1 # units: mm
# window_size = 27 # units: pixels

# spydata_dir = 'spydata'

# file_pickle = 'flame_' + str(flame_nr) + '_' + 'unfiltered' + '_' +'segment_length_' + str(segment_length_mm) + 'mm_' + 'wsize_' + str(window_size) + 'pixels'

# with open(spydata_dir + sep + file_pickle + '.pkl', 'rb') as f:
#     flame = pickle.load(f)

# frame_nrs = [0, 1]

# for frame_nr in frame_nrs:
    
#     segmented_contours = flame.frames[frame_nr].contour_data.segmented_contours
    
#     slopes_of_segmented_contours = []
#     tortuosity_of_segmented_contours = []
    
#     for segmented_contour in segmented_contours:
        
#         slopes_of_segmented_contour = slope(segmented_contour)
#         slopes_of_segmented_contours.append(slopes_of_segmented_contour)
        
#         tortuosity_of_segmented_contour = tortuosity(segmented_contour)
#         tortuosity_of_segmented_contours.append(tortuosity_of_segmented_contour)


#     flame.frames[frame_nr].contour_data.slopes_of_segmented_contours = slopes_of_segmented_contours
#     flame.frames[frame_nr].contour_data.tortuosity_of_segmented_contours = tortuosity_of_segmented_contours

# with open(spydata_dir + sep + file_pickle + '.pkl', 'wb') as f:
      
#     # A new file will be created
#     pickle.dump(flame, f)
    
#%% TEST WITH FLAME OBJECT
# flame_4.tortuosity_data = tortuosity(flame_4, 0)




