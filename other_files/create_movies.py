import cv2
import os
import progressbar
from parameters import *

def create_movie_from_png(folder_path, video_name, fps=30):
    """
    Creates a movie from a sequence of PNG images in a folder using OpenCV.

    Parameters:
    folder_path (str): The path to the folder containing the PNG images.
    video_name (str): The path to the output video file.
    fps (int): The frame rate of the output video. Default is 30.

    Returns:
    None
    """
    # Get the list of PNG images in the folder that don't contain the underscore character
    images = [img for img in os.listdir(folder_path) if img.endswith('.png') and '_' not in img]

    # Sort the images alphabetically
    images.sort()

    # Get the dimensions of the first image in the list
    frame = cv2.imread(os.path.join(folder_path, images[0]))
    height, width, layers = frame.shape
    
    # print(height, width)
    height_scaled = 720 #int(height/8)
    width_scaled = 960  #int(width/8)
    
    size = (width_scaled,height_scaled) 
    
    # Create the video writer object
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    # Loop through the images and add them to the video
    images = images
    for image in progressbar.progressbar(images):
        img = cv2.imread(os.path.join(folder_path, image))
        img = cv2.resize(img, size)
        video.write(img)

    # Release the video writer object and destroy all windows
    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    
    folder_path = os.path.join('post_data', record_name, 'bfm',  f'w_size_{window_size}')
    output_file_path = os.path.join('post_data', record_name, 'videos',  f'w_size_{window_size}')
    filename = 'vid'
    fps = 24
    
    # folder_path = 'post_data/flame_5/Cam_Date=211021_Time=162640_ExtractVolume_Frame0/bilateral_filter_method/w_size_29'
    # output_file_path = 'post_data/flame_5/videos/'
    # filename = 'bilteral_filter_w_size=29_sigma_space=13.5_sigma_color=0.1_60Hz'
    # fps = 60
    
    if not os.path.exists(output_file_path):
      
      os.makedirs(output_file_path)
      
    # Set the video filename
    video_name = output_file_path + filename + '.mp4'
    
    create_movie_from_png(folder_path, video_name, fps)
    