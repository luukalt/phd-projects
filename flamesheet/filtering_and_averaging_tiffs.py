#%% IMPORT PACKAGES 
import os
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

#%% IMPORT USER DEFINED PACKAGES
from sys_paths import parent_directory
import sys_paths
import rc_params_settings
from parameters import *
from plot_params import colormap, fontsize, fontsize_legend, fontsize_label, fontsize_fraction


def read_tiff_images(folder_path, window_size, save_path=None):
    image_list = []
    
    file_names = [file_name for file_name in os.listdir(folder_path) if file_name.endswith('.tiff') or file_name.endswith('.tif')]
    
    for file_name in tqdm(file_names, desc="Processing images"):
        file_path = os.path.join(folder_path, file_name)
        
        # print(file_path)
        
        image = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)
        
        img_normalized = cv2.normalize(image, dst=None, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        w_size = int(window_size)
        filter_diameter = w_size
        sigma_color = 0.1
        sigma_space = filter_diameter/2.0
        
        img_bilateral = cv2.bilateralFilter(img_normalized, filter_diameter, sigma_color, sigma_space)
        
        if save_path:
            save_image(img_bilateral, save_path, file_name)
            
        if image is not None:
            image_list.append(img_bilateral)
            
    return image_list

def save_image(image, save_path, file_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file_path = os.path.join(save_path, file_name)
    # Convert the image to uint8 format before saving
    img_to_save = (image * 255).astype(np.uint8)
    cv2.imwrite(save_file_path, img_to_save)

def save_image_16bit(image, save_path, file_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file_path = os.path.join(save_path, file_name)
    # Convert the image to uint16 format before saving
    img_to_save = (image * 65535).astype(np.uint16)
    cv2.imwrite(save_file_path, img_to_save)
    
def calculate_average_image(image_list):
    # Assuming all images have the same shape
    sum_image = np.zeros_like(image_list[0], dtype=np.float64)
    for image in tqdm(image_list, desc="Calculating Average..."):
        sum_image += image
    avg_image = sum_image / len(image_list)
    return avg_image

def plot_image(image, title='Average Image'):
    
    fig, ax = plt.subplots()
    
    ax.imshow(image, cmap="viridis", vmin=0, vmax=np.max(image.flatten())/128)
    
    ax.set_xlabel('pixels', fontsize=16)
    ax.set_ylabel('pixels', fontsize=16)



if __name__ == '__main__':
    
    google_red = '#db3236'
    google_green = '#3cba54'
    google_blue = '#4885ed'
    google_yellow = '#f4c20d'
    
    cv2.destroyAllWindows()
    plt.close("all")
    
    save_path = 'figures'
    
    data_dir = 'Y:\\laaltenburg\\flamesheet_2d_campaign1'
    cwd = os.getcwd()
    
   
    pre_data_folder = "pre_data"
    post_data_folder = "post_data"
    
    # day_nr = '23-2'         
    # record_name = 'Recording_Date=230215_Time=143726_01'                                          
    # pre_record_name = 'Recording_Date=230215_Time=153306_01'                                                
    # scale = 10.68
    # frame_nr = 0
    # segment_length_mm = 1                                           # units: mm
    # window_size = int(np.ceil(2*scale) // 2 * 2 + 1) 
    # extension = '.tif'

    # pre_data_path = os.path.join(cwd, flame.pre_data_folder, flame.name, f'session_{flame.session_nr:03}' , flame.record_name, 'Correction', 'Resize', f'Frame{frame_nr}', 'Export_01')
    # record_data_path = os.path.join(data_dir, f'flamesheet_2d_day{day_nr:03}', record_name, 'Correction', 'NonLinear_SubSlidingMin', f'Frame{frame_nr}', 'Export_01')
    record_data_path = os.path.join(data_dir, f'flamesheet_2d_day{day_nr:03}', record_name, 'Correction', 'SubOverTimeMin_sl=99', f'Frame{frame_nr}', 'Export')
    
    post_data_path = 'post_data'

    image_nr = 5
    
    toggle_plot = True
    save_image_toggle  = False
    
    procedure_nr = 2
    

    folder_path = record_data_path
    image_list = read_tiff_images(folder_path, window_size, save_path if save_image_toggle else None)
    if not image_list:
        print("No TIFF images found in the folder.")
        
    avg_image = calculate_average_image(image_list)
    
    save_file_path = os.path.join('pickles', f'{record_name}_AvgOfBfm_wsize_{window_size}.pkl')
    with open(save_file_path, 'wb') as f:
        pickle.dump(avg_image, f)
        
    save_image_16bit(avg_image, 'pickles', f'{record_name}_AvgOfBfm_wsize_{window_size}_16bit.tiff')
     
    plot_image(avg_image)


