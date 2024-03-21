# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 23:31:16 2024

@author: luuka
"""

#%% IMPORT PACKAGES
import numpy as np

#%%  FUNCTIONS

def process_df(df, D_in, offset_to_wall_center, offset):
    """
    Process the DataFrame by shifting and normalizing coordinates.

    :param df: DataFrame to process.
    :param D_in: Diameter for normalization.
    :param offset_to_wall_center: Offset for x-coordinate shifting.
    :param offset: Offset for y-coordinate shifting.
    :return: Processed DataFrame.
    """
    
    df['x_shift [mm]'] = df['x [mm]'] - (D_in/2 - offset_to_wall_center)
    df['y_shift [mm]'] = df['y [mm]'] + offset

    df['x_shift_norm'] = df['x_shift [mm]']/D_in
    df['y_shift_norm'] = df['y_shift [mm]']/D_in

    df['x_shift [m]'] = df['x_shift [mm]']*1e-3
    df['y_shift [m]'] = df['y_shift [mm]']*1e-3

    return df

def contour_correction(flame, contour_nr, r_left_raw, r_right_raw, x_bottom_raw, x_top_raw, window_size_r_raw, window_size_x_raw, frame_nr=0):
    
    segmented_contour =  flame.frames[frame_nr].contour_data.segmented_contours[contour_nr]
    
    segmented_contour_r = segmented_contour[:, 0, 0]
    segmented_contour_x = segmented_contour[:, 0, 1]
    
    # x and y coordinates of the discretized (segmented) flame front 
    contour_r_corrected = segmented_contour_r*window_size_r_raw + r_left_raw
    contour_x_corrected = segmented_contour_x*window_size_x_raw + x_top_raw
    
    # Non-dimensionalize coordinates by pipe diameter
    contour_r_corrected /= 1 #D_in
    contour_x_corrected /= 1 #D_in
    
    contour_r_corrected_array = np.array(contour_r_corrected)
    contour_x_corrected_array = np.array(contour_x_corrected)
    
    # Combine the x and y coordinates into a single array of shape (n_coords, 2)
    contour_corrected_coords = np.array([contour_r_corrected_array, contour_x_corrected_array]).T

    # Create a new array of shape (n_coords, 1, 2) and assign the coordinates to it
    contour_corrected = np.zeros((len(contour_r_corrected_array), 1, 2))
    contour_corrected[:, 0, :] = contour_corrected_coords
    
    return contour_corrected  

# =============================================================================
#  Start intersect function
# =============================================================================
"""
Give, two x,y curves this gives intersection points,
autor: Sukhbinder
5 April 2017


Based on: http://uk.mathworks.com/matlabcentral/fileexchange/11837-fast-and-robust-curve-intersections
"""



def intersection(x1, y1, x2, y2):
    """
INTERSECTIONS Intersections of curves.
   Computes the (x,y) locations where two curves intersect.  The curves
   can be broken with NaNs or have vertical segments.

usage:
x,y=intersection(x1,y1,x2,y2)

    Example:
    a, b = 1, 2
    phi = np.linspace(3, 10, 100)
    x1 = a*phi - b*np.sin(phi)
    y1 = a - b*np.cos(phi)

    x2=phi
    y2=np.sin(phi)+2
    x,y=intersection(x1,y1,x2,y2)

    plt.plot(x1,y1,c='r')
    plt.plot(x2,y2,c='g')
    plt.plot(x,y,'*k')
    plt.show()

    """
    
    def _rect_inter_inner(x1, x2):
        n1 = x1.shape[0]-1
        n2 = x2.shape[0]-1
        X1 = np.c_[x1[:-1], x1[1:]]
        X2 = np.c_[x2[:-1], x2[1:]]
        S1 = np.tile(X1.min(axis=1), (n2, 1)).T
        S2 = np.tile(X2.max(axis=1), (n1, 1))
        S3 = np.tile(X1.max(axis=1), (n2, 1)).T
        S4 = np.tile(X2.min(axis=1), (n1, 1))
        return S1, S2, S3, S4


    def _rectangle_intersection_(x1, y1, x2, y2):
        S1, S2, S3, S4 = _rect_inter_inner(x1, x2)
        S5, S6, S7, S8 = _rect_inter_inner(y1, y2)

        C1 = np.less_equal(S1, S2)
        C2 = np.greater_equal(S3, S4)
        C3 = np.less_equal(S5, S6)
        C4 = np.greater_equal(S7, S8)

        ii, jj = np.nonzero(C1 & C2 & C3 & C4)
        return ii, jj

    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)

    ii, jj = _rectangle_intersection_(x1, y1, x2, y2)
    n = len(ii)

    dxy1 = np.diff(np.c_[x1, y1], axis=0)
    dxy2 = np.diff(np.c_[x2, y2], axis=0)

    T = np.zeros((4, n))
    AA = np.zeros((4, 4, n))
    AA[0:2, 2, :] = -1
    AA[2:4, 3, :] = -1
    AA[0::2, 0, :] = dxy1[ii, :].T
    AA[1::2, 1, :] = dxy2[jj, :].T

    BB = np.zeros((4, n))
    BB[0, :] = -x1[ii].ravel()
    BB[1, :] = -x2[jj].ravel()
    BB[2, :] = -y1[ii].ravel()
    BB[3, :] = -y2[jj].ravel()

    for i in range(n):
        try:
            T[:, i] = np.linalg.solve(AA[:, :, i], BB[:, i])
        except:
            T[:, i] = np.Inf

    in_range = (T[0, :] >= 0) & (T[1, :] >= 0) & (
        T[0, :] <= 1) & (T[1, :] <= 1)

    xy0 = T[2:, in_range]
    xy0 = xy0.T
    return xy0[:, 0], xy0[:, 1]

# =============================================================================
#  End intersect function
# =============================================================================
