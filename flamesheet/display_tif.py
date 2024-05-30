# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 12:22:11 2022

@author: luuka
"""

import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

cv2.destroyAllWindows()
plt.close("all")

def showInMovedWindow(winname, img, x, y):
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, x, y)   # Move it to (x,y)
    cv2.imshow(winname, img)
    

x_screen = 0
y_screen = 0

# filename = "imgBackground.tif"
image_nr = str("3320")
filename = "Export\B" + str(image_nr)+ ".tif" 
img_raw = cv2.imread(filename, cv2.IMREAD_ANYDEPTH) #cv2.IMREAD_ANYDEPTH)
# showInMovedWindow('raw ' + str(image_nr), img_raw, 0, 0)

#%% Convert image to graycsale: Normalize
img_gray_16bit = cv2.normalize(img_raw, dst=None, alpha=0, beta=2**16-1, norm_type=cv2.NORM_MINMAX)
showInMovedWindow('grayscale_16bit', img_gray_16bit, x_screen, y_screen)

# Dividing by 16 seems counter intuitive. One would expect a division by 256 (= 16*16), but this somehow "helps" this algorithm. 
# img_gray = (img_gray/16).astype(np.uint8) 

img_gray_8bit = (img_gray_16bit/256).astype(np.uint8)
showInMovedWindow('grayscale_8bit', img_gray_8bit, x_screen, y_screen)

factor = 2**2
brighten = cv2.addWeighted(img_gray_8bit, factor, img_gray_8bit, 0, 0)
showInMovedWindow('brighten', brighten, x_screen, y_screen)

ksize = 9
area = ksize*ksize 
kernel = np.ones((ksize,ksize),np.float32)
avg1 = cv2.filter2D(img_gray_8bit/255, -1, kernel)
avg1_area = avg1/area
showInMovedWindow('Averaging 1', avg1_area, x_screen, y_screen)

data = 255*avg1_area# Now scale by 255
img = data.astype(np.uint8)
showInMovedWindow("Averaging 1 - uint8", img, x_screen, y_screen)

src = img_gray_8bit
threshold1 = 0.001
threshold2 = 0.002
edges = cv2.Canny(src, threshold1, threshold2) #, apertureSize, L2gradient)
showInMovedWindow("Canny 1", img, x_screen, y_screen)

# print(avg1_area.dtype)























