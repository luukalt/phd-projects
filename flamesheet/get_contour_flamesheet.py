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
import math
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from scipy.signal import find_peaks, peak_prominences
from scipy.interpolate import UnivariateSpline, CubicSpline
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

# from contour_properties import contour_segmentation
from plot_params import colormap, fontsize, fontsize_legend

figures_folder = 'figures'
if not os.path.exists(figures_folder):
        os.makedirs(figures_folder)

#%% GET CONTOUR DATA
def get_contour_data(procedure_nr, window_size, pre_record_data_path, pre_data_path, post_data_path, image_nr, extension, toggle_plot, save_image):
    
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
        shape, contour, contour_length_pixels = get_contour_procedure_bilateral_filter_method(window_size, pre_record_data_path, pre_data_path, post_data_path, image_nr, extension, color, toggle_plot, save_image)
    elif procedure_nr == 3:
        color = google_red #google_green
        shape, contour, contour_length_pixels = get_contour_procedure_bilateral_filter_method2(window_size, pre_data_path, post_data_path, image_nr, extension, color, toggle_plot, save_image)

    return shape, contour, contour_length_pixels


#%% METHOD B: BILATERAL FILTER METHOD
def get_contour_procedure_bilateral_filter_method(window_size, mask_path, record_data_path, post_data_path, image_nr, extension, color, toggle_plot, save_image): 

    #%% [1] Read raw image
    image_file = f'B{image_nr:04d}{extension}'
    image_path = os.path.join(record_data_path, image_file)
    img_raw = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
    shape = img_raw.shape
    
    print(shape)
    print(shape[0]*shape[1])
    
    image_nr_mask = 1
    image_file = f'B{image_nr_mask:04d}{extension}'
    mask_path = os.path.join(mask_path, image_file)
    mask = cv2.imread(mask_path, cv2.IMREAD_ANYDEPTH)
    mask = (mask > 0).astype(np.uint8)
    
    result = cv2.bitwise_and(img_raw, img_raw, mask=mask)
    # result_float = result.astype(float)
    # result_float[mask == 0] = np.nan
    
    # # Normalize the result excluding NaN values
    # valid_values = result_float[~np.isnan(result_float)]
    # min_valid = np.min(valid_values)
    # max_valid = np.max(valid_values)
    
    # result_normalized = (result_float - min_valid) / (max_valid - min_valid)
    # # result_normalized[mask == 0] = np.nan

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(mask)  # Display the masked image
    axs[1].imshow(img_raw)  # Display the masked image
    brighten_factor = 200
    axs[2].imshow(result, vmin=np.min(result.flatten())/brighten_factor, vmax=np.max(result.flatten())/brighten_factor)  # Display the masked image
    
    #%% [2] Normalize the signal intensity of raw image to unity based on global raw image maxima and minima
    img_normalized = cv2.normalize(result, dst=None, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    fig, ax = plt.subplots()
    brighten_factor = 1
    ax.imshow(img_normalized, vmin=np.min(img_normalized.flatten())/brighten_factor, vmax=np.max(img_normalized.flatten())/brighten_factor)  # Display the masked image
    
    #%% [3] Apply bilateral filter on normalized image
    w_size = int(window_size*1)
    filter_diameter = w_size
    sigma_color = 0.1
    sigma_space = filter_diameter/2.0
    img_bilateral = cv2.bilateralFilter(img_normalized, filter_diameter, sigma_color, sigma_space)
    
    # img_normalized = cv2.normalize(img_raw, dst=None, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # img_bilateral = cv2.bilateralFilter(img_normalized, filter_diameter, sigma_color, sigma_space)
    
    # img_bilateral = bilateral_filter(result_normalized, mask, filter_diameter, sigma_color, sigma_space) 
    # img_bilateral = bilateral_filter_own(result_normalized, filter_diameter, sigma_color, sigma_space)
    
    fig, ax = plt.subplots()
    brighten_factor = 200
    # im = ax.imshow(img_bilateral, vmin=np.min(img_bilateral.flatten())/brighten_factor, vmax=np.max(img_bilateral.flatten())/brighten_factor)  # Display the masked image
    
    
    with open(os.path.join('figures', 'bfm', 'B0001'), 'rb') as f:
            img_bilateral = pickle.load(f)
    
    im = ax.imshow(img_bilateral, vmin=np.min(img_bilateral.flatten())/brighten_factor, vmax=np.max(img_bilateral.flatten())/brighten_factor)  # Display the masked image
    
    # Add a colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Intensity', rotation=270, labelpad=15)  # Label for the colorbar    
    
    #%% [4] Obtain threshold value corresponding to a minimum probability 
    threshold_value = get_thresholding_value(img_bilateral, mask, toggle_plot)
    
    print(threshold_value)
    
    #%% [5] Binarize the bilateral filtered image 
    threshold_type, maxVal  = cv2.THRESH_BINARY, 1                             
    ret, img_binary = cv2.threshold(img_bilateral, threshold_value, maxVal, threshold_type)
    
    fig, ax = plt.subplots()
    im = ax.imshow(img_binary, vmin=0, vmax=1)  # Display the masked image
    
    img_new = cv2.bitwise_and(img_binary, img_binary, mask=mask)
    
    #%% [6] Extract largest contour == flame front
    contour, contour_length_pixels = find_and_draw_flame_contours(img_new, mask)
    
    # print(contour)
    
    #%% [7]Turn plots on or off
    brighten_factor = 8
    
    toggle_plot = True
    
    if toggle_plot:
        
        fig, axs = plt.subplots(1, 3, figsize=(10, 6))
        plt.subplots_adjust(wspace=-.3)
        plot_images(axs, img_raw, img_bilateral, brighten_factor, contour, color)
        
        # fig.tight_layout()
        # filename = f'H{flame.H2_percentage}_Re{flame.Re_D}_detection_B{image_nr}'
        # eps_path = os.path.join('figures', f"{filename}.eps")
        # fig.savefig(eps_path, format='eps', dpi=300, bbox_inches='tight')
        
        toggle_contour = False
        title = 'raw image (\#' + str(image_nr) + ')'
        title = ''
        plot_image(title, img_raw, brighten_factor, contour, toggle_contour, color)
        
        title = 'Bilateral filtered image (\#' + str(image_nr) + ')'
        toggle_contour = True
        title = ''
        brighten_factor = 128
        plot_image(title, img_bilateral, brighten_factor, contour, toggle_contour, color)
        
        toggle_contour = True
        title = 'raw image (\#' + str(image_nr) + ')' + ' with contour'
        title = ''
        plot_image(title , img_raw, brighten_factor, contour, toggle_contour, color)
        
    #%% [8] Save images with contour drawn into the raw image
    if save_image:
    
        path = os.path.join(post_data_path, 'bfm', f'w_size_{w_size}')
        save_contour_images(path, image_nr, img_raw, brighten_factor, contour, color)
    
    return shape, contour, contour_length_pixels

#%% MAIN FUNCTIONS

def get_thresholding_value(image, mask, toggle_plot):
    
    quantity = image[mask == 1]
    
    print(len(quantity))
    # quantity = image.flatten()
    
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
        
        fig, ax = plot_pixel_density_histogram(quantity, mask, percentile_99_double)
        
        ax.plot(dist_data_x, dist_data_y, color='r', ls='-', label='kernel density estimation')
        ax.plot(quantity_coord, probability_coord, color='#db3236', marker='x', ms=10, mew=2, ls='None')
        
        # # Plot the vertical line on the axis
        # ax.axvline(x=quantity_coord, ymin=0, ymax=0.6, color='#db3236', ls='--', label='minimum probability') 
        # # ax.axvline(x=0.5*(quantity_range[0] + quantity_range[-1]), ymin=0, ymax=0.6, color='k', ls='--') 
        
        # # This is in data coordinates
        # point = (quantity_coord, probability_coord)

        # trans = ax.transData.transform(point)
        # trans = ax.transAxes.inverted().transform(trans)

        # # Add text to the left of the line
        # text_left = 'products'
        # ax.text(trans[0] - 0.025, 0.5, text_left, ha='right', va='center', transform=ax.transAxes, fontsize=12)

        # # Add text to the right of the line
        # text_right = 'reactants'
        # ax.text(trans[0] + 0.025, 0.5, text_right, ha='left', va='center', transform=ax.transAxes, fontsize=12)
        
        # ax.legend(loc='best', prop={'size': 16})
        
        # fig.tight_layout()
        # # filename = f'H{flame.H2_percentage}_Re{flame.Re_D}_histogram_B{image_nr}'
        # # eps_path = os.path.join('figures', f"{filename}.eps")
        # # fig.savefig(eps_path, format='eps', dpi=300, bbox_inches='tight')
        
    # Determine threshold pixel density (separating unburnt and burnt side) [value corresponding to minimum probability]
    threshold = quantity_range[min_probability_index]
    
    return threshold

def find_and_draw_flame_contours(img_binary, mask):
    
    img_binary_8bit = (img_binary * (2 ** 8 - 1)).astype(np.uint8)
    
    # Find the contours
    src = img_binary_8bit # Input image array
    contour_retrieval = cv2.RETR_TREE # Contour retrieval mode
    contours_approx = cv2.CHAIN_APPROX_NONE
    contours_found = cv2.findContours(src, contour_retrieval, contours_approx)
    contours = imutils.grab_contours(contours_found)
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # contour = largest_contour[total_indices]
    contour_flame_front = largest_contour
    closed_contour = False
    contour_length_pixels = cv2.arcLength(contour_flame_front, closed_contour)
    
    
    ### MASK
    img_mask_8bit = (mask * (2 ** 8 - 1)).astype(np.uint8)
    
    # Find the contours
    src = img_mask_8bit # Input image array
    contour_retrieval = cv2.RETR_TREE # Contour retrieval mode
    contours_approx = cv2.CHAIN_APPROX_NONE
    contours_found = cv2.findContours(src, contour_retrieval, contours_approx)
    contours = imutils.grab_contours(contours_found)
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # contour = largest_contour[total_indices]
    contour_mask = largest_contour
    closed_contour = False
    contour_length_pixels = cv2.arcLength(contour_mask, closed_contour)
    
    # Convert contours to sets of tuples for easy comparison
    contour_mask_set = set(tuple(point[0]) for point in contour_mask)
    
    # Create z as a list to collect points
    contour = []
    
    # Iterate through x and add points to z if they are not in y_set
    for point in contour_flame_front:
        if tuple(point[0]) not in contour_mask_set:
            contour.append(point)
        else:
            break
    
    # Convert z to numpy array (if needed)
    contour = np.array(contour)
    
    contour_spline = fit_parametric_cubic_spline(contour)
    
    contour_length_pixels = cv2.arcLength(contour, closed_contour)
    
    print(type(contour_spline), contour_spline.shape, len(contour_spline))

    return contour_spline, contour_length_pixels



def bilateral_filter(image, mask, diameter, sigma_color, sigma_space):
    
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
            
            if mask[i, j] > 0:
                
                # extract the current pixel
                pixel = image[i, j]
    
                # compute the squared color distances for all neighboring pixels
                color_dists = (pixel - image[i - radius:i + radius + 1, j - radius:j + radius + 1]) ** 2
    
                # compute the weights for the color distances
                color_weights = np.exp(-color_dists / (2 * sigma_color ** 2))
    
                # compute the combined weights for all pixels
                combined_weights = color_weights * spatial_kernel
    
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
    
    # Define rectangle parameters
    x_min, y_min = 75, 400  # Lower-left corner
    width, height = 225 - 75, 575 - 400  # Width and height of the rectangle
    
    # Create a Rectangle patch
    rectangle = patches.Rectangle((x_min, y_min), width, height, edgecolor='lime', facecolor='none', linewidth=3, zorder=2)
    
    ax3.add_patch(rectangle)
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
    
    # segmented_contour_x, segmented_contour_y, segmented_contour = contour_segmentation(contour, flame.segment_length_pixels)
    
    if toggle_contour:
        contour_x = contour[:,:,0]
        contour_y = contour[:,:,1]
        ax.plot(contour_x, contour_y, color, lw=1)
        ax.plot(contour_x[0], contour_y[0], 'yx')
        ax.plot(contour_x[-1], contour_y[-1], 'gx')
        
        
    
    # if toggle_contour:
        
    #     # contour_x = segmented_contour[:,:,0]
    #     # contour_y = segmented_contour[:,:,1]
        
    #     ax.plot(contour_x, contour_y, c='y', marker='o', ms=10, ls='solid', lw=2)
        
    #     lw = 3
    #     segment_nr = 95
    #     color = 'magenta'
    #     ax.plot(contour_x[segment_nr:segment_nr + 2], contour_y[segment_nr:segment_nr + 2], c=color, marker='o', ms=10, ls='solid', lw=2)
        
    #     # Calculate the direction vector of the line segment
    #     start_x, start_y = contour_x[segment_nr], contour_y[segment_nr]
    #     dx = contour_x[segment_nr + 1] - start_x
    #     dy = contour_y[segment_nr + 1] - start_y
        
    #     # Define the length of the line to draw
    #     length = 40
        
    #     # Calculate the coordinates of the new point
    #     end_x1 = start_x + (dx / np.sqrt(dx**2 + dy**2)) * length
    #     end_y1 = start_y + (dy / np.sqrt(dx**2 + dy**2)) * length
        
    #     end_x2 = start_x - (dx / np.sqrt(dx**2 + dy**2)) * length
    #     end_y2 = start_y - (dy / np.sqrt(dx**2 + dy**2)) * length
        
    #     copy_length1 = np.abs(end_y1 - contour_y[segment_nr + 1])
    #     copy_length2 = np.abs(end_y2 - contour_y[segment_nr + 1])
        
    #     # Plot the new point
    #     ax.plot([start_x, end_x1], [start_y, end_y1], ls='dashed', color=color, lw=lw)  # Adjust marker and color as needed
    #     ax.plot([start_x, end_x2], [start_y, end_y2], ls='solid', color=color, lw=lw)  # Adjust marker and color as needed
        
    #     ax.vlines(x=contour_x[segment_nr + 1], ymin=contour_y[segment_nr + 1], ymax=end_y1, colors=color, linestyles='dashed', lw=lw)
    #     ax.vlines(x=contour_x[segment_nr + 1], ymin=contour_y[segment_nr + 1] - copy_length2, ymax=contour_y[segment_nr + 1], colors=color, linestyles='solid', lw=lw)
        
    #     # Add text relative to the axes
    #     ax.text(.52, .9, r'$\theta < 0$', fontsize=16, ha='center', color=color, transform=ax.transAxes, bbox=dict(facecolor="w", edgecolor='k', boxstyle='round'))

        
    #     segment_nr = 101
    #     color = 'magenta'
    #     ax.plot(contour_x[segment_nr:segment_nr + 2], contour_y[segment_nr:segment_nr + 2], c=color, marker='o', ms=10, ls='solid', lw=2)
        
    #     # Calculate the direction vector of the line segment
    #     start_x, start_y = contour_x[segment_nr + 1], contour_y[segment_nr + 1]
    #     dx = start_x - contour_x[segment_nr]
    #     dy = start_y - contour_y[segment_nr]
        
    #     # Define the length of the line to draw
    #     length = 40
        
    #     # Calculate the coordinates of the new point
    #     end_x1 = start_x + (dx / np.sqrt(dx**2 + dy**2)) * length
    #     end_y1 = start_y + (dy / np.sqrt(dx**2 + dy**2)) * length
        
    #     end_x2 = start_x - (dx / np.sqrt(dx**2 + dy**2)) * length
    #     end_y2 = start_y - (dy / np.sqrt(dx**2 + dy**2)) * length
        
    #     copy_length1 = np.abs(end_y1 - contour_y[segment_nr + 1])
    #     copy_length2 = np.abs(end_y2 - contour_y[segment_nr + 1])
        
    #     # Plot the new point
    #     ax.plot([start_x, end_x1], [start_y, end_y1], ls='solid', color=color, lw=lw)  # Adjust marker and color as needed
    #     ax.plot([start_x, end_x2], [start_y, end_y2], ls='dashed', color=color, lw=lw)  # Adjust marker and color as needed
        
    #     ax.vlines(x=contour_x[segment_nr + 1], ymin=start_y, ymax=end_y1 + copy_length1/2, colors=color, linestyles='solid', lw=lw)
    #     ax.vlines(x=contour_x[segment_nr + 1], ymin=start_y - copy_length1, ymax=start_y, colors=color, linestyles='dashed', lw=lw)
        
    #     # Add text relative to the axes
    #     ax.text(.35, .1, r'$\theta > 0$', fontsize=16, ha='center', color=color, transform=ax.transAxes, bbox=dict(facecolor="w", edgecolor='k', boxstyle='round'))
    
    # # custom_y_ticks = [0,  800]
    # # ax.set_yticks(custom_y_ticks)
    
    # ax.set_xlim(left=75, right=225)
    # ax.set_ylim(bottom=575, top=400)
    # ax.set_aspect('equal')
    
    # # Set the number of ticks you want
    # num_ticks = 5
    # ax.locator_params(axis='x', nbins=num_ticks)
    # ax.locator_params(axis='y', nbins=num_ticks)
    
    # fig.tight_layout()
    # filename = f'H{flame.H2_percentage}_Re{flame.Re_D}_detection_B{image_nr}_zoom'
    # eps_path = os.path.join('figures', f"{filename}.eps")
    # fig.savefig(eps_path, format='eps', dpi=300, bbox_inches='tight')
    
    # plt.title(title)
    # plt.imshow(image, cmap="gray", vmin=np.min(image.flatten())/brighten_factor, vmax=np.max(image.flatten())/brighten_factor)
    # plt.xlabel('pixels')
    # plt.ylabel('pixels')
    
    # if toggle_contour:
    #     contour_x = contour[:,:,0]
    #     contour_y = contour[:,:,1]
    #     plt.plot(contour_x, contour_y, color)
    
    # return fig
    
def plot_pixel_density_histogram(quantity, mask, x_lim_right):
    
    fig, ax = plt.subplots(figsize=(6, 6))
    # fig, ax = plt.subplots()
    
    ax.grid()
    
    # quantity = image.flatten()
    
    y, x, _ = ax.hist(quantity, bins='auto', density=True, color='lightblue', edgecolor='k') 
    
    ax.set_xlim(0, .01)
    # ax.set_ylim(0, 100)
    
    # custom_x_ticks = [.0, .02, .04, .06, .08, .1]
    # custom_x_tick_labels =  [f'{tick:.2f}' for tick in custom_x_ticks] # Replace with your desired tick labels
    # ax.set_xticks(custom_x_ticks)
    # ax.set_xticklabels(custom_x_tick_labels)  # Use this line to set custom tick labels
    
    # custom_y_ticks = [0, 25, 50, 75, 100]
    # ax.set_yticks(custom_y_ticks)
    
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
    
    # toggle_contour = False
    # plot_image('', img_raw, brighten_factor, contour, toggle_contour, color)
    # plt.savefig(filename + '_raw.png', dpi=dpi)
    # plt.clf()
    
    toggle_contour = True
    plot_image('', img_raw, brighten_factor, contour, toggle_contour, color)
    plt.savefig(filename + '.png', dpi=dpi,  bbox_inches='tight')
    # plt.clf()
    
    # Using cv2.imwrite() method
    # Saving the image
    # cv2.imwrite(file_name + '_raw.png', img)
    # cv2.imwrite(file_name + '.png', img)
    
    # fig1.savefig(file_name + '_raw.png', dpi=dpi) 
    # fig2.savefig(file_name + '.png', dpi=dpi) 

def fit_parametric_cubic_spline(coords, num_points=5):
    
    print(coords.shape, type(coords))
    step = len(coords) // (num_points * 2)  # Calculate the step size
    
    # Select coordinates with step
    selected_coords = coords[::step].squeeze(axis=1)  # Remove the single-dimensional entries
    
    # Check if the last coordinate is already included
    if len(coords) % step != 0:
        selected_coords = np.vstack([selected_coords, coords[-1].squeeze()])
        
    s = np.arange(len(selected_coords))
    
    s_interp = np.linspace(s.min(), s.max(), num=1000)
    
    print(coords[:,:,0])
    
    x_spline = CubicSpline(s, selected_coords[:,0], bc_type='not-a-knot')
    y_spline = CubicSpline(s, selected_coords[:,1], bc_type='not-a-knot')

    x_interp = x_spline(s_interp)
    y_interp = y_spline(s_interp)
    
    # Calculate the arc length
    dx_interp = np.diff(x_interp)
    dy_interp = np.diff(y_interp)
    
    spline_coords = np.column_stack((x_interp, y_interp))
    
    spline_coords = spline_coords[:, np.newaxis, :] 
    
    return spline_coords
      
#%% MAIN

if __name__ == "__main__":
    
    google_red = '#db3236'
    google_green = '#3cba54'
    google_blue = '#4885ed'
    google_yellow = '#f4c20d'
    
    cv2.destroyAllWindows()
    plt.close("all")
    
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
    # mask_path = os.path.join(data_dir, f'flamesheet_2d_day{day_nr:03}', pre_record_name, 'MaskCreateGeometric_01', 'MakePermanentImgMask', 'AboveBelow', 'Correction', f'Frame{frame_nr}', 'Export_01')
    mask_path = os.path.join(data_dir, f'flamesheet_2d_day{day_nr:03}', 'Masks')
    
    post_data_path = 'post_data'

    image_nr = 5
    
    record_data_path
    toggle_plot = True
    save_image = False
    
    procedure_nr = 2
    
    shape, contour, contour_length_pixels = get_contour_data(procedure_nr, window_size, mask_path, record_data_path, post_data_path, image_nr, extension, toggle_plot, save_image)









