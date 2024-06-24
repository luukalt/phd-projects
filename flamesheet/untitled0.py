import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def read_tiff_images(folder_path, window_size, save_path=None):
    image_list = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.tiff') or file_name.endswith('.tif'):
            file_path = os.path.join(folder_path, file_name)
            
            # print(file_path)
            
            image = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)
            
            img_normalized = cv2.normalize(image, dst=None, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            
            w_size = int(window_size*1)
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
    
def calculate_average_image(image_list):
    # Assuming all images have the same shape
    sum_image = np.zeros_like(image_list[0], dtype=np.float64)
    for image in image_list:
        sum_image += image
    avg_image = sum_image / len(image_list)
    return avg_image

def plot_image(image, title='Average Image'):
    
    fig, ax = plt.subplots()
    
    ax.imshow(image, cmap="viridis", vmin=np.min(image.flatten()), vmax=np.max(image.flatten()))
    
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
    
    day_nr = '23-2'         
    record_name = 'Recording_Date=230215_Time=143726_01'                                          
    pre_record_name = 'Recording_Date=230215_Time=153306_01'                                                
    scale = 10.68
    frame_nr = 0
    segment_length_mm = 1                                           # units: mm
    window_size = int(np.ceil(2*scale) // 2 * 2 + 1) 
    extension = '.tif'

    # pre_data_path = os.path.join(cwd, flame.pre_data_folder, flame.name, f'session_{flame.session_nr:03}' , flame.record_name, 'Correction', 'Resize', f'Frame{frame_nr}', 'Export_01')
    record_data_path = os.path.join(data_dir, f'flamesheet_2d_day{day_nr:03}', record_name, 'Correction', 'NonLinear_SubSlidingMin', f'Frame{frame_nr}', 'Export_01')
    
    post_data_path = 'post_data'

    image_nr = 5
    
    toggle_plot = True
    save_image_toggle  = True
    
    procedure_nr = 2
    

    folder_path = record_data_path
    image_list = read_tiff_images(folder_path, window_size, save_path if save_image_toggle else None)
    if not image_list:
        print("No TIFF images found in the folder.")
        
    avg_image = calculate_average_image(image_list)
    #%%
    def plot_image(image, title='Average Image'):
        
        fig, ax = plt.subplots()
        
        ax.imshow(image, cmap="viridis", vmin=0, vmax=np.max(image.flatten())/128)
        
        ax.set_xlabel('pixels', fontsize=16)
        ax.set_ylabel('pixels', fontsize=16)
    
    import pickle
    
    save_file_path = os.path.join('figures', 'avg_bfm')
    with open(save_file_path, 'wb') as f:
        pickle.dump(avg_image, f)
        
    
    plot_image(avg_image)


