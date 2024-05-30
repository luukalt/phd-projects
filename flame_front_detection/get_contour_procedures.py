# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 15:00:34 2022

@author: laaltenburg

Algoritmes to detect the flame front of confinded and unconfined flames from raw PIV images
"""

#%% IMPORT PACKAGES
import os
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks, peak_prominences
from scipy import stats
import imutils
import pickle

# Add the 'main' folder to sys.path
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# flame_front_detection_directory = os.path.abspath(os.path.join(parent_folder, 'flame_front_detection'))
flame_simulations_directory = os.path.abspath(os.path.join(parent_folder, 'flame_simulations'))
plot_parameters_directory = os.path.abspath(os.path.join(parent_folder, 'plot_parameters'))

# Add the flame_object_directory to sys.path
sys.path.append(parent_folder)
# sys.path.append(flame_front_detection_directory)
sys.path.append(flame_simulations_directory)
sys.path.append(plot_parameters_directory)

from contour_properties import contour_segmentation
from plot_params import colormap, fontsize, fontsize_legend

figures_folder = 'figures'
if not os.path.exists(figures_folder):
        os.makedirs(figures_folder)

#%% GET CONTOUR DATA
def get_contour_data(procedure_nr, window_size, pre_data_path, post_data_path, image_nr, extension, toggle_plot, save_image):
    
    google_red = '#db3236'
    google_green = '#3cba54'
    google_blue = '#4885ed'
    google_yellow = '#f4c20d'
    bright_green = '#66ff00'
    
    if procedure_nr == 1:
        color = bright_green #google_red
        shape, contour, contour_length_pixels = get_contour_procedure_pixel_density_method(window_size, pre_data_path, post_data_path, image_nr, extension, color, toggle_plot, save_image)
    elif procedure_nr == 2:
        color = google_red #google_green
        shape, contour, contour_length_pixels = get_contour_procedure_bilateral_filter_method(window_size, pre_data_path, post_data_path, image_nr, extension, color, toggle_plot, save_image)
    elif procedure_nr == 3:
        color = google_red #google_green
        shape, contour, contour_length_pixels = get_contour_procedure_bilateral_filter_method2(window_size, pre_data_path, post_data_path, image_nr, extension, color, toggle_plot, save_image)

    return shape, contour, contour_length_pixels

#%% METHOD A: PIXEL DENSITY METHOD
def get_contour_procedure_pixel_density_method(window_size, pre_data_path, post_data_path, image_nr, extension, color, toggle_plot, save_image):
    
    #%% Step 1: read raw image
    image_file = f'B{image_nr:04d}{extension}'
    image_path = os.path.join(pre_data_path, image_file)
    img_raw = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
    shape = img_raw.shape
    
    #%% Step 2: normalize the signal intensity of raw image to unity based on global raw image maxima and minima
    img_normalized = cv2.normalize(img_raw, dst=None, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    #%% Step 3: apply averaging filter on normalized image
    w_size = int(window_size)
    kernel = np.ones((w_size, w_size), np.float32)/(w_size**2)
    img_pixel_density = cv2.filter2D(img_normalized, -1, kernel)
    
    #%% Step 4: obtain threshold value corresponding to a minimum probability 
    threshold_value = get_thresholding_value(img_pixel_density, toggle_plot) 
    
    #%% Step 5: binarize the bilateral filtered image
    threshold_type, maxVal  = cv2.THRESH_BINARY, 1                              
    ret, img_binary = cv2.threshold(img_pixel_density, threshold_value, maxVal, threshold_type)
    
    #%% Step 6: extract largest contour == flame front
    contour, contour_length_pixels = find_and_draw_flame_contours(img_binary)
    
    #%% Turn plots on or off
    brighten_factor = 8
    
    if toggle_plot:
        
        toggle_contour = False
        title = 'raw image (\#' + str(image_nr) + ')'
        # title = ''
        plot_image(title, img_raw, brighten_factor, contour, toggle_contour, color)
        
        title = 'Average filtered image (\#' + str(image_nr) + ')'
        # title = ''
        plot_image(title, img_pixel_density, brighten_factor, contour, toggle_contour, color)
        
        toggle_contour = True
        title = 'raw image (\#' + str(image_nr) + ')' + ' with contour'
        # title = ''
        plot_image(title , img_raw, brighten_factor, contour, toggle_contour, color)
        

    #%% Final step: save images with contour drawn into the raw image
    if save_image:
        path = os.path.join(post_data_path, 'pdm', f'w_size_{w_size}')
        save_contour_images(path, image_nr, img_raw, brighten_factor, contour, color)
    
    return shape, contour, contour_length_pixels

    
#%% METHOD B: BILATERAL FILTER METHOD
def get_contour_procedure_bilateral_filter_method(window_size, pre_data_path, post_data_path, image_nr, extension, color, toggle_plot, save_image): 
    
    # image_file = f'B{image_nr:04d}{extension}'
    # image_path = os.path.join(pre_data_path, image_file)
    # img_raw_real = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)

    #%% [1] Read raw image
    image_file = f'B{image_nr:04d}{extension}'
    image_path = os.path.join(pre_data_path, image_file)
    img_raw = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
    shape = img_raw.shape
    
    #%% [2] Normalize the signal intensity of raw image to unity based on global raw image maxima and minima
    img_normalized = cv2.normalize(img_raw, dst=None, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    #%% [3] Apply bilateral filter on normalized image
    w_size = int(window_size*1)
    filter_diameter = w_size
    sigma_color = 0.1
    sigma_space = filter_diameter/2.0
    img_bilateral = cv2.bilateralFilter(img_normalized, filter_diameter, sigma_color, sigma_space)
    
    #%% [4] Obtain threshold value corresponding to a minimum probability 
    threshold_value = get_thresholding_value(img_bilateral, toggle_plot)

    #%% [5] Binarize the bilateral filtered image 
    threshold_type, maxVal  = cv2.THRESH_BINARY, 1                             
    ret, img_binary = cv2.threshold(img_bilateral, threshold_value, maxVal, threshold_type)
    
    #%% [6] Extract largest contour == flame front
    contour, contour_length_pixels = find_and_draw_flame_contours(img_binary)
    
    #%% [7]Turn plots on or off
    brighten_factor = 8
    
    toggle_plot = True
    
    if toggle_plot:
        
        fig, axs = plt.subplots(1, 3,) # figsize=(10, 6))
        plt.subplots_adjust(wspace=-.3)
        plot_images(axs, img_raw, img_bilateral, brighten_factor, contour, color)
        
        fig.tight_layout()
        filename = f'H{flame.H2_percentage}_Re{flame.Re_D}_detection_B{image_nr}'
        eps_path = os.path.join('figures', f"{filename}.eps")
        fig.savefig(eps_path, format='eps', dpi=300, bbox_inches='tight')
        
        toggle_contour = False
        # title = 'raw image (\#' + str(image_nr) + ')'
        title = ''
        plot_image(title, img_raw, brighten_factor, contour, toggle_contour, color)
        
        # title = 'Bilateral filtered image (\#' + str(image_nr) + ')'
        toggle_contour = False
        title = ''
        plot_image(title, img_bilateral, brighten_factor, contour, toggle_contour, color)
        
        toggle_contour = True
        # title = 'raw image (\#' + str(image_nr) + ')' + ' with contour'
        title = ''
        plot_image(title , img_raw, brighten_factor, contour, toggle_contour, color)
        
    #%% [8] Save images with contour drawn into the raw image
    if save_image:
    
        path = os.path.join(post_data_path, 'bfm', f'w_size_{w_size}')
        save_contour_images(path, image_nr, img_raw, brighten_factor, contour, color)
    
    return shape, contour, contour_length_pixels

#%% METHOD C: BILATERAL FILTER METHOD + INTENSITY CORRECTION
def get_contour_procedure_bilateral_filter_method2(window_size, pre_data_path, post_data_path, image_nr, extension, color, toggle_plot, save_image): 
    
    #%% [1] Read raw image
    image_file = f'B{image_nr:04d}{extension}'
    image_path = os.path.join(pre_data_path, image_file)
    img_raw = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
    shape = img_raw.shape
    
    w_size = int(window_size)
    
    #%% [2] Normalize the signal intensity of raw image to unity based on global raw image maxima and minima
    img_normalized = cv2.normalize(img_raw, dst=None, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img_raw_boxfilter = cv2.boxFilter(img_normalized, -1, (w_size, w_size))
    img_divide = np.divide(img_normalized, img_raw_boxfilter)
    img_normalized = cv2.normalize(img_divide, dst=None, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    #%% [3] Apply bilateral filter on normalized image
    filter_diameter = w_size
    sigma_color = 0.1
    sigma_space = filter_diameter/2.0
    img_bilateral = cv2.bilateralFilter(img_normalized, filter_diameter, sigma_color, sigma_space)
    
    #%% [4] Obtain threshold value corresponding to a minimum probability 
    threshold_value = get_thresholding_value(img_bilateral, toggle_plot)

    #%% [5] Binarize the bilateral filtered image 
    threshold_type, maxVal  = cv2.THRESH_BINARY, 1                             
    ret, img_binary = cv2.threshold(img_bilateral, threshold_value, maxVal, threshold_type)
    
    #%% [6] Extract largest contour == flame front
    contour, contour_length_pixels = find_and_draw_flame_contours(img_binary)
    
    #%% [7]Turn plots on or off
    brighten_factor = 4
    
    toggle_plot = True
    
    if toggle_plot:
        
        fig, axs = plt.subplots(1, 3, figsize=(10, 6))
        
        plot_images(axs, img_raw, img_bilateral, brighten_factor, contour, color)
        
        fig.tight_layout()
        filename = f'H{flame.H2_percentage}_Re{flame.Re_D}_detection_B{image_nr}'
        eps_path = os.path.join('figures', f"{filename}.eps")
        # fig.savefig(eps_path, format='eps', dpi=300, bbox_inches='tight')
        
        toggle_contour = False
        # title = 'raw image (\#' + str(image_nr) + ')'
        title = ''
        plot_image(title, img_raw, brighten_factor, contour, toggle_contour, color)
        
        # title = 'Bilateral filtered image (\#' + str(image_nr) + ')'
        toggle_contour = False
        title = ''
        plot_image(title, img_divide, brighten_factor, contour, toggle_contour, color)
        
        toggle_contour = True
        # title = 'raw image (\#' + str(image_nr) + ')' + ' with contour'
        title = ''
        plot_image(title , img_raw, brighten_factor, contour, toggle_contour, color)
        
    #%% [8] Save images with contour drawn into the raw image
    if save_image:
    
        path = os.path.join(post_data_path, 'bfm', f'w_size_{w_size}')
        save_contour_images(path, image_nr, img_raw, brighten_factor, contour, color)
    
    return shape, contour, contour_length_pixels
#%% MAIN FUNCTIONS

def get_thresholding_value(image, toggle_plot):
    
    quantity = image.flatten()
    
    # Compute the 99th percentile
    percentile_99 = np.percentile(quantity, 99)
    percentile_99_double = 1*percentile_99
    
    # Filter the data using the double the value of the 99th percentile
    quantity = quantity[quantity <= percentile_99_double]
    
    # Create a Gaussian Kernel Density Estimation (KDE) object using quantity
    gkde_obj = stats.gaussian_kde(quantity)
    
    x_pts = np.linspace(0, np.max(quantity), 1000)
    estimated_pdf = gkde_obj.evaluate(x_pts)
    
    # Extract fitted distribution
    dist_data_x = x_pts
    dist_data_y = estimated_pdf
    
    # Find the peaks in the distribution
    peaks, _ = find_peaks(dist_data_y)
    
    # Obtain the prominences of the peaks
    prominences = peak_prominences(dist_data_y, peaks)[0]
    
    # Create a list of tuples containing peak information: (peak index, prominence, x coordinate, y coordinate)
    peak_coords = list(zip(peaks, prominences, dist_data_x[peaks], dist_data_y[peaks]))
    
    # Sort the peak coordinates in descending order based on prominence
    peak_coords_prominence_descending = sorted(peak_coords, key = lambda x:x[1], reverse=True)
    
    # Sort the top 2 peak coordinates in ascending order based on probability
    peak_coords_sorted_probability = sorted(peak_coords_prominence_descending[0:2], key = lambda x:x[3])
    
    # Sort the peak coordinates based on the index
    accepted_peak_coords = sorted(peak_coords_sorted_probability, key = lambda x:x[0])
    
    # print(accepted_peak_coords)
    
    accepted_peaks = [accepted_peak_coord[0] for accepted_peak_coord in accepted_peak_coords]
    
    quantity_range = dist_data_x[accepted_peaks[0]:accepted_peaks[-1]]
    probability_range = dist_data_y[accepted_peaks[0]:accepted_peaks[-1]]
    
    min_probability_index = np.argmin(probability_range)
    
    toggle_plot = True
    
    if toggle_plot:
        
        quantity_coord = quantity_range[min_probability_index]
        probability_coord = probability_range[min_probability_index]
        
        fig, ax = plot_pixel_density_histogram(image, percentile_99_double)
        
        ax.plot(dist_data_x, dist_data_y, color='#000080', ls='-', label='kernel density estimation')
        # ax.plot(quantity_coord, probability_coord, color='#db3236', marker='x', ms=10, mew=2, ls='None')
        
        # Plot the vertical line on the axis
        ax.axvline(x=quantity_coord, ymin=0, ymax=0.6, color='#db3236', ls='--', label='minimum probability') 
        # ax.axvline(x=0.5*(quantity_range[0] + quantity_range[-1]), ymin=0, ymax=0.6, color='k', ls='--') 
        
        # This is in data coordinates
        point = (quantity_coord, probability_coord)

        trans = ax.transData.transform(point)
        trans = ax.transAxes.inverted().transform(trans)

        # Add text to the left of the line
        text_left = 'products'
        ax.text(trans[0] - 0.025, 0.5, text_left, ha='right', va='center', transform=ax.transAxes, fontsize=12)

        # Add text to the right of the line
        text_right = 'reactants'
        ax.text(trans[0] + 0.025, 0.5, text_right, ha='left', va='center', transform=ax.transAxes, fontsize=12)
        
        ax.legend(loc='best', prop={'size': 16})
        
        fig.tight_layout()
        filename = f'H{flame.H2_percentage}_Re{flame.Re_D}_histogram_B{image_nr}'
        eps_path = os.path.join('figures', f"{filename}.eps")
        fig.savefig(eps_path, format='eps', dpi=300, bbox_inches='tight')
        
    # Determine threshold pixel density (separating unburnt and burnt side) [value corresponding to minimum probability]
    threshold = quantity_range[min_probability_index]
    
    return threshold

def find_and_draw_flame_contours(img_binary):
    
    img_binary_8bit = (img_binary * (2 ** 8 - 1)).astype(np.uint8)
    
    # Find the contours
    src = img_binary_8bit # Input image array
    contour_retrieval = cv2.RETR_TREE # Contour retrieval mode
    contours_approx = cv2.CHAIN_APPROX_NONE
    contours_found = cv2.findContours(src, contour_retrieval, contours_approx)
    contours = imutils.grab_contours(contours_found)
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # print(largest_contour[:])
    
    # Compute point where flame anchors at burner rim on right and left hand side
    # image axis start in top left corner and are positive in right and downward direction 
    contour_all_y_values = np.hstack(largest_contour[:,:,1])
    y_max = np.where(contour_all_y_values == max(contour_all_y_values))[0]
    
    contour_x_values_at_y_max = np.hstack(largest_contour[y_max,:,0])
    x_right = np.where(contour_x_values_at_y_max == max(contour_x_values_at_y_max))[0]
    x_left = np.where(contour_x_values_at_y_max == min(contour_x_values_at_y_max))[0]
    
    right_point_index = y_max[x_right[0]] 
    left_point_index = y_max[x_left[0]] 
    
    diff = len(largest_contour) - right_point_index
    
    right_indices = np.linspace(right_point_index, (len(largest_contour) - 1), diff)
    left_indices = np.linspace(0, left_point_index, left_point_index + 1)
    total_indices = np.concatenate((right_indices, left_indices)).astype('int64')
    
    contour = largest_contour[total_indices]
    closed_contour = False
    contour_length_pixels = cv2.arcLength(contour, closed_contour)
    
    return contour, contour_length_pixels


def bilateral_filter(image, diameter, sigma_color, sigma_space):
    
    # create a zero-filled output image
    filtered_image = np.zeros_like(image)

    # compute half of the diameter for convenience
    radius = diameter // 2

    # create a Gaussian kernel for spatial smoothing
    x, y = np.meshgrid(np.arange(-radius, radius+1), np.arange(-radius, radius+1))
    spatial_kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma_space ** 2))

    # compute the weights for the color distances
    color_weights = np.exp(-np.arange(2**0) ** 2 / (2 * sigma_color ** 2))

    # loop over all pixels in the input image
    for i in range(radius, image.shape[0] - radius):
        for j in range(radius, image.shape[1] - radius):
            
            # extract the current pixel
            pixel = image[i, j]

            # compute the squared color distances for all neighboring pixels
            color_dists = (pixel - image[i - radius:i + radius + 1, j - radius:j + radius + 1]) ** 2

            # compute the weights for the color distances
            color_weights_ = color_weights[np.sqrt(color_dists).astype(np.int32)]

            # compute the combined weights for all pixels
            combined_weights = color_weights_ * spatial_kernel

            # normalize the weights
            weights_sum = np.sum(combined_weights)
            normalized_weights = combined_weights / weights_sum if weights_sum > 0 else combined_weights

            # compute the weighted sum of neighboring pixels
            filtered_pixel = np.sum(normalized_weights * image[i - radius:i + radius + 1, j - radius:j + radius + 1])

           # assign the filtered pixel to the output image
            filtered_image[i, j] = filtered_pixel

    return filtered_image


#%% AUXILIARY PLOT FUNCTIONS

def plot_images(axs, image1, image2, brighten_factor, contour, color):
    
    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]
    
    ax1.imshow(image1, cmap="gray", vmin=np.min(image1.flatten())/brighten_factor, vmax=np.max(image1.flatten())/brighten_factor)
    ax1.set_ylabel('pixels', fontsize=24)
    ax1.yaxis.set_label_coords(-.1, .8)  # Adjust the position as needed

    toggle_contour = False
    if toggle_contour:
        contour_x = contour[:,:,0]
        contour_y = contour[:,:,1]
        ax1.plot(contour_x, contour_y, color)
    
    custom_y_ticks = [0, 400, 800]
    ax1.set_yticks(custom_y_ticks)
    
    ax2.imshow(image2, cmap="gray", vmin=np.min(image2.flatten())/brighten_factor, vmax=np.max(image2.flatten())/brighten_factor)
    ax2.set_xlabel('pixels', fontsize=24)
    toggle_contour = False
    if toggle_contour:
        contour_x = contour[:,:,0]
        contour_y = contour[:,:,1]
        ax1.plot(contour_x, contour_y, color)
    
    ax2.tick_params(axis='y', labelleft=False)
    
    ax3.imshow(image1, cmap="gray", vmin=np.min(image1.flatten())/brighten_factor, vmax=np.max(image1.flatten())/brighten_factor)

    toggle_contour = True
    if toggle_contour:
        contour_x = contour[:,:,0]
        contour_y = contour[:,:,1]
        ax3.plot(contour_x, contour_y, color)
    
    ax3.set_ylabel('')
    ax3.tick_params(axis='y', labelleft=False)
    
    labels = ['a', 'b', 'c']
    
    for i, ax in enumerate(axs):
        
        ax.tick_params(axis='both', labelsize=20)
        
        # Add textbox with timestamp
        left, width = .2, .7
        bottom, height = .25, .7
        right = left + width
        top = bottom + height
        
        ax.text(right, top,  labels[i], 
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes,
                fontsize=24,
                bbox=dict(facecolor="w", edgecolor='k', boxstyle='round')
                )
    
def plot_image(title, image, brighten_factor, contour, toggle_contour, color):
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(title)
    ax.imshow(image, cmap="gray", vmin=np.min(image.flatten())/brighten_factor, vmax=np.max(image.flatten())/brighten_factor)
    ax.set_xlabel('pixels', fontsize=fontsize)
    ax.set_ylabel('pixels', fontsize=fontsize)
    
    segmented_contour_x, segmented_contour_y, segmented_contour = contour_segmentation(contour, flame.segment_length_pixels)
    
    if toggle_contour:
        contour_x = contour[:,:,0]
        contour_y = contour[:,:,1]
        ax.plot(contour_x, contour_y, color, lw=4)
    
    if toggle_contour:
        contour_x = segmented_contour[:,:,0]
        contour_y = segmented_contour[:,:,1]
        ax.plot(contour_x, contour_y, c='y', marker='o', ms=10, ls='solid', lw=2)
    
    # custom_y_ticks = [0,  800]
    # ax.set_yticks(custom_y_ticks)
    
    shift = 25
    ax.set_xlim(left=100-shift, right=200-shift)
    ax.set_ylim(bottom=600, top=500)
    
    # plt.title(title)
    # plt.imshow(image, cmap="gray", vmin=np.min(image.flatten())/brighten_factor, vmax=np.max(image.flatten())/brighten_factor)
    # plt.xlabel('pixels')
    # plt.ylabel('pixels')
    
    # if toggle_contour:
    #     contour_x = contour[:,:,0]
    #     contour_y = contour[:,:,1]
    #     plt.plot(contour_x, contour_y, color)
    
    # return fig
    
def plot_pixel_density_histogram(image, x_lim_right):
    
    fig, ax = plt.subplots(figsize=(6, 6))
    # fig, ax = plt.subplots()
    
    ax.grid()
    
    quantity = image.flatten()
    
    y, x, _ = ax.hist(quantity, bins='auto', density=True, color='lightblue', edgecolor='k') 
    
    ax.set_xlim(0, .1)
    ax.set_ylim(0, 100)
    
    custom_x_ticks = [.0, .02, .04, .06, .08, .1]
    custom_x_tick_labels =  [f'{tick:.2f}' for tick in custom_x_ticks] # Replace with your desired tick labels
    ax.set_xticks(custom_x_ticks)
    ax.set_xticklabels(custom_x_tick_labels)  # Use this line to set custom tick labels
    
    custom_y_ticks = [0, 25, 50, 75, 100]
    ax.set_yticks(custom_y_ticks)
    
    ax.tick_params(axis='both', labelsize=16)
    
    ax.set_xlabel('$I_{f}$', fontsize=20)
    ax.set_ylabel('pdf', fontsize=20)
    
    return fig, ax


def save_contour_images(path,image_nr, img_raw, brighten_factor, contour, color):
    
    # create a new directory because it does not exist 
    if not os.path.exists(path):
      
      os.makedirs(path)
    
    filename = os.path.join(path, f"B{image_nr:04d}")
    dpi = 300
    
    toggle_contour = False
    plot_image('', img_raw, brighten_factor, contour, toggle_contour, color)
    # plt.savefig(filename + '_raw.png', dpi=dpi)
    # plt.clf()
    
    toggle_contour = True
    plot_image('', img_raw, brighten_factor, contour, toggle_contour, color)
    plt.savefig(filename + '.eps', dpi=dpi,  bbox_inches='tight')
    # plt.clf()
    
    # Using cv2.imwrite() method
    # Saving the image
    # cv2.imwrite(file_name + '_raw.png', img)
    # cv2.imwrite(file_name + '.png', img)
    
    # fig1.savefig(file_name + '_raw.png', dpi=dpi) 
    # fig2.savefig(file_name + '.png', dpi=dpi) 
        
#%% MAIN

if __name__ == "__main__":
    
    google_red = '#db3236'
    google_green = '#3cba54'
    google_blue = '#4885ed'
    google_yellow = '#f4c20d'
    
    cv2.destroyAllWindows()
    plt.close("all")
    
    # data_dir = "Y:/tube/"
    data_dir = 'U:\\staff-umbrella\\High hydrogen\\laaltenburg\\data\\tube_burner_campaign2\\selected_runs\\'
    
    cwd = os.getcwd()
    
    #%%% Define cases
    react_names_ls =    [
                        # ('react_h0_c3000_ls_record1', 57),
                        # ('react_h0_s4000_ls_record1', 58),
                        # ('react_h100_c12000_ls_record1', 61),
                        # ('react_h100_c12500_ls_record1', 61),
                        # ('react_h100_s16000_ls_record1', 62)
                        ]
    
    react_names_hs =    [
                        # ('react_h0_f2700_hs_record1', 57),
                        # ('react_h0_c3000_hs_record1', 57),
                        ('react_h0_s4000_hs_record1', 58),
                        # ('react_h100_c12500_hs_record1', 61),
                        # ('react_h100_s16000_hs_record1', 62)
                        ]
    
    if react_names_ls:
        spydata_dir = os.path.join(parent_folder, 'spydata\\udf')
    elif react_names_hs:
        spydata_dir = os.path.join(parent_folder, 'spydata')
        
    pre_data_folder = "pre_data"
    post_data_folder = "post_data"
    
    # flame_nr = 1
    # flame_name = 'flame_' + str(flame_nr)
    # record_name = "Cam_Date=190923_Time=164708"
    # frame_nr = 0
    # extension = ".tif"
    
    # flame_nr = 2
    # flame_name = 'flame_' + str(flame_nr)
    # record_name = "Cam_Date=190920_Time=173417"
    # frame_nr = 0
    # extension = ".tif"
    
    # flame_nr = 3
    # flame_name = 'flame_' + str(flame_nr)
    # record_name = "Cam_Date=210708_Time=150209_ExtractVolume"
    # frame_nr = 0
    # extension = ".tif"
    
    # flame_nr = 4
    # flame_name = 'flame_' + str(flame_nr)
    # record_name = 'Cam_Date=210917_Time=134643_ExtractVolume'
    # frame_nr = 0
    # extension = '.tif'
    # scale = 12.7493 # units: pixels.mm^{-1} 
    
    # flame_nr = 5
    # flame_name = 'flame_' + str(flame_nr)
    # record_name = "Cam_Date=211021_Time=162640_ExtractVolume"
    # frame_nr = 0
    # extension = ".tif"
    # scale = 13.8182 # units: pixels.mm^{-1}
    
    # flame_nr = 6
    # flame_name = 'flame_' + str(flame_nr)
    # record_name = 'Cam_Date=210910_Time=153000_phi=1'
    # frame_nr = 0
    # extension = '.tif'
    # scale = 13.6028  # units: pixels.mm^{-1} 
    
    frame_nr = 0
    segment_length_mm = 1 # units: mm
    window_size = 31 # units: pixels
    
    react_names_ls =    [
                        # ('react_h0_c3000_ls_record1', 57),
                        # ('react_h0_s4000_ls_record1', 58),
                        # ('react_h100_c12000_ls_record1', 61),
                        # ('react_h100_c12500_ls_record1', 61),
                        # ('react_h100_s16000_ls_record1', 62)
                        ]
    
    react_names_hs =    [
                        # ('react_h0_f2700_hs_record1', 57),
                        # ('react_h0_c3000_hs_record1', 57),
                        ('react_h0_s4000_hs_record1', 58),
                        # ('react_h100_c12500_hs_record1', 61),
                        # ('react_h100_s16000_hs_record1', 62)
                        ]
    
    react_names = react_names_ls + react_names_hs
    
    for name, nonreact_run_nr in react_names:
    
        fname = f'{name}_segment_length_{segment_length_mm}mm_wsize_{window_size}pixels'
    
        with open(os.path.join(spydata_dir, fname + '.pkl'), 'rb') as f:
            flame = pickle.load(f)
        
# =============================================================================
# %%%    0: Load flame data
# =============================================================================
    
    # segment_length_mm = 1 #0.125 # units: mm
    # window_size = 27 # units: pixels
    # spydata_dir = "spydata/"
    # file_pickle = spydata_dir + "flame_" + str(flame_nr) + '_' + 'unfiltered' + '_' +'segment_length_' + str(segment_length_mm) + 'mm_' + 'wsize_' + str(window_size) + 'pixels.pkl'

    # with open(file_pickle, 'rb') as f:
    #     flame = pickle.load(f)
    
    # pre_data_path = data_dir + sep + pre_data_folder + sep + flame_name + sep + record_name + "_Frame" + str(frame_nr) + sep
    # post_data_path = data_dir + sep + post_data_folder + sep + flame_name + sep + record_name + "_Frame" + str(frame_nr) + sep
    
    # image_nr = 1
    
    # img_raw = cv2.imread(pre_data_path + 'B%.4d' % image_nr + extension, cv2.IMREAD_ANYDEPTH)
    
    # toggle_plot = True
    # toggle_contour = False
    # save_image = False
    
    # brighten_factor = 8
    # color = 'r'
    # # plot_image('raw image ' + str(image_nr), img_raw, brighten_factor, [], toggle_contour, color)
    # procedure_nr = flame.get_contour_procedure_nr
    
    # shape, contour, contour_length_pixels = get_contour_data(procedure_nr, window_size, pre_data_path, post_data_path, image_nr, extension, toggle_plot, save_image)
    
    # contour_nr = image_nr - 1
    # segmented_contour =  flame.frames[frame_nr].contour_data.segmented_contours[contour_nr]
    
    # segmented_contour_x =  segmented_contour[:,0,0]
    # segmented_contour_y =  segmented_contour[:,0,1]
    
    # plt.plot(segmented_contour_x, segmented_contour_y, color)
     
# =============================================================================
#     0: Load flame data
# =============================================================================
    
# =============================================================================
# %%%   A: Single procedure, single image, single frame
# =============================================================================
    
    # pre_data_path = os.path.join(cwd, flame.pre_data_folder, flame.name, f'session_{flame.session_nr:03}' , flame.record_name, 'Correction', 'Resize', f'Frame{frame_nr}', 'Export_01')
    pre_data_path = os.path.join(data_dir, f'session_{flame.session_nr:03}' , flame.record_name, 'Correction', 'Resize', f'Frame{frame_nr}', 'Export_01')
    
    post_data_path = os.path.join(cwd, flame.post_data_folder, flame.name, f'session_{flame.session_nr:03}', f'{flame.record_name}_Frame{frame_nr}')
    
    # pre_data_path = os.path.join(data_dir, pre_data_folder, flame_name, f"{record_name}_Frame{frame_nr}")
    # post_data_path = os.path.join(data_dir, post_data_folder, flame_name, f"{record_name}_Frame{frame_nr}")

    image_nr = 1699
    
    toggle_plot = True
    save_image = True
    
    procedure_nr = 2
    
    extension = flame.extension
    shape, contour, contour_length_pixels = get_contour_data(procedure_nr, window_size, pre_data_path, post_data_path, image_nr, extension, toggle_plot, save_image)

# =============================================================================
#     A: Single procedure, single image, single frame
# =============================================================================
    

# =============================================================================
#%%%     B: Multiple procedures, single image, single frame
# =============================================================================
    
    # window_size = np.ceil(2*scale) // 2 * 2 + 1 # Get closest odd number
    
    # pre_data_path = data_dir + sep + pre_data_folder + sep + flame_name + sep + record_name + "_Frame" + str(frame_nr) + sep
    # post_data_path = data_dir + sep + post_data_folder + sep + flame_name + sep + record_name + "_Frame" + str(frame_nr) + sep
    
    # image_nr = 197
    # img_raw = cv2.imread(pre_data_path + 'B%.4d' % image_nr + extension, cv2.IMREAD_ANYDEPTH)
    
    # toggle_plot = False
    # save_image = False
    
    # fig, ax = plot_image('raw image ' + str(image_nr), img_raw, brighten_factor=8)
    
    # procedure_nrs = [1, 2]
    # colors = ['r', 'g']
    
    # for i, procedure_nr in enumerate(procedure_nrs):
        
    #     shape, contour, contour_length_pixels = get_contour_data(procedure_nr, window_size, pre_data_path, post_data_path, image_nr, extension, toggle_plot, save_image)
    
    #     contour_x = contour[:,:,0]
    #     contour_y = contour[:,:,1]
    #     ax.plot(contour_x, contour_y, colors[i])
    
# =============================================================================
#     B: Multiple procedures, single image, single frame
# =============================================================================

# =============================================================================
#%%%     C: Single procedure, single image, double frame
# =============================================================================
    
    # procedure_nr = 2

    # frame_nrs = [0, 1]
    # colors = ['r', 'g']

    # image_nr = 197
    # pre_data_path = data_dir + sep + pre_data_folder + sep + flame_name + sep + record_name + "_Frame" + str(frame_nrs[0]) + sep
    # img_raw = cv2.imread(pre_data_path + 'B%.4d' % image_nr + extension, cv2.IMREAD_ANYDEPTH)
    
    # toggle_plot = False
    # save_image = False
    
    # fig, ax = plot_image('raw image ' + str(image_nr), img_raw, brighten_factor=8)
    
    # window_size = np.ceil(2*scale) // 2 * 2 + 1 # Get closest odd number
    
    # for i, frame_nr in enumerate(frame_nrs):
        
    #     pre_data_path = data_dir + sep + pre_data_folder + sep + flame_name + sep + record_name + "_Frame" + str(frame_nr) + sep
    #     post_data_path = data_dir + sep + post_data_folder + sep + flame_name + sep + record_name + "_Frame" + str(frame_nr) + sep

    #     shape, contour, contour_length_pixels = get_contour_data(procedure_nr, window_size, pre_data_path, post_data_path, image_nr, extension, toggle_plot, save_image)
    
    #     contour_x = contour[:,:,0]
    #     contour_y = contour[:,:,1]
    #     ax.plot(contour_x, contour_y, colors[i])
    
# =============================================================================
#     C: Single procedure, single image, double frame
# =============================================================================

# =============================================================================
#%%%    D: Single procedure, double image, single frame
# =============================================================================
   
    # procedure_nr = 2
    # image_nrs = [197, 198]
    # colors = ['r', 'g']
    # frame_nr = 0
    
    # image_nr = image_nrs[0]
    # pre_data_path = data_dir + sep + pre_data_folder + sep + flame_name + sep + record_name + "_Frame" + str(frame_nr) + sep
    
    # img_raw = cv2.imread(pre_data_path + 'B%.4d' % image_nr + extension, cv2.IMREAD_ANYDEPTH)
    
    # toggle_plot = False
    # save_image = False
    
    # fig, ax = plot_image('raw image ' + str(image_nr), img_raw, brighten_factor=8)
    
    # window_size = np.ceil(2*scale) // 2 * 2 + 1 # Get closest odd number
    
    # for i, image_nr in enumerate(image_nrs):
        
    #     pre_data_path = data_dir + sep + pre_data_folder + sep + flame_name + sep + record_name + "_Frame" + str(frame_nr) + sep
    #     post_data_path = data_dir + sep + post_data_folder + sep + flame_name + sep + record_name + "_Frame" + str(frame_nr) + sep
        
    #     shape, contour, contour_length_pixels = get_contour_data(procedure_nr, window_size, pre_data_path, post_data_path, image_nr, extension, toggle_plot, save_image)
    
    #     contour_x = contour[:,:,0]
    #     contour_y = contour[:,:,1]
    #     ax.plot(contour_x, contour_y, colors[i])

# =============================================================================
#     D: Single procedure, double image, single frame
# =============================================================================
    
    # os.chdir(pre_data_path)
    # for filename in os.listdir(pre_data_path):
    #     print(filename)
    #     os.rename(filename, filename.replace('B', 'B0'))
    
    # # x and y coordinates of the discretized (segmented) flame front 
    # contour_nr = image_nr - 1
    # contour_x = flame.tortuosity_data.x_list[contour_nr]
    # contour_y = flame.tortuosity_data.y_list[contour_nr]

    # ax.plot(contour_x, contour_y, color='y')
    
    # shape = (5, 5)
    # dtype = np.uint16 # data type
    # img_empty = 3000*np.ones(shape, dtype) # Destination image
    # factor = 2**0
    # img_empty_8bit = (img_empty/factor).astype(np.uint8)
    
    
    
# =============================================================================
#  code snippets    
# =============================================================================

 # print(pixel_density_threshold)
 # #%% Step 3-2:
 # pixel_density_gradient = np.gradient(img_pixel_density)
 
 # pixel_density_gradient_x = np.abs(pixel_density_gradient[0])
 # pixel_density_gradient_y = np.abs(pixel_density_gradient[1])
 # pixel_density_gradient_combined = np.abs(pixel_density_gradient_x + pixel_density_gradient_y)
 
 # plot_number_density(pixel_density_gradient_combined[::-1])
 
 # w_size = int(window_size/2)
 # kernel = np.ones((w_size, w_size),np.float32)/(w_size**2)
 # img_pixel_density_gradient_combined = cv2.filter2D(pixel_density_gradient_combined, -1, kernel)
 
 # plot_number_density(img_pixel_density_gradient_combined[::-1])
 
 # figx, axx = plt.subplots()
 # axx.grid()
 
 # distplot2 = sns.histplot(img_pixel_density_gradient_combined.flatten(), kde=True, stat='density', ax=axx)
 
 # # Extract fitted distribution
 # dist_data = distplot2.get_lines()[0].get_data()
 
 # dist_data_x = dist_data[0]
 # dist_data_y = dist_data[1]
 
 # peaks, _ = find_peaks(dist_data_y)
 
 # prominences = peak_prominences(dist_data_y, peaks)[0]
 
 # peak_coords = list(zip(peaks, prominences, dist_data_x[peaks], dist_data_y[peaks]))
 
 # peak_coords_prominence_descending = sorted(peak_coords, key = lambda x:x[1], reverse=True)
 
 # peak_coords_sorted_probability = sorted(peak_coords_prominence_descending[0:2], key = lambda x:x[3])

 # accepted_peak_coords = sorted(peak_coords_sorted_probability, key = lambda x:x[0])

 # accepted_peaks = [accepted_peak_coord[0] for accepted_peak_coord in accepted_peak_coords]
 
 # axx.plot(dist_data_x[accepted_peaks], dist_data_y[accepted_peaks], color='b', marker='x', ms=8, mew=2, ls='None')
 
 # print(np.max(dist_data_x[accepted_peaks]))
 
 # #%% Determine threshold pixel density (separating unburnt and burnt side) [value corresponding to minimum probability]
 # threshold2 = np.max(dist_data_x[accepted_peaks])
 
 # #%% Step 4: Binarize image
 # src = img_pixel_density_gradient_combined # Input image array (must be in Grayscale)
 # thresholdValue = threshold2 #140 # flame 1 and 2 (.tif) require a different threshold value compared to flame 3 (.bmp)
 # maxVal = 2**8 - 1 # Maximum value that can be assigned to a pixel. 
 # threshold_type = cv2.THRESH_BINARY # The type of thresholding to be applied. 
 # ret2, img_binary2 = cv2.threshold(src, thresholdValue, maxVal, threshold_type) 

# =============================================================================
#  code snippets    
# =============================================================================










