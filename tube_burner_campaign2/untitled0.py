# -*- coding: utf-8 -*-
"""
Created on Mon May 13 17:14:44 2024

@author: luuka
"""

import numpy as np

alpha_tan = .43
norm = np.pi

# Calculate the angle in degrees
angle_radians = alpha_tan * norm
angle_degrees = np.degrees(angle_radians)

print("Angle in degrees:", angle_degrees)


# Define the angle in degrees
angle_degrees = 10.5  # Change this to your desired angle

# Convert angle from degrees to radians
angle_radians = np.radians(angle_degrees)

# Define the norm
norm = np.pi

# Calculate alpha_tan
alpha_tan = angle_radians / norm

print("alpha_tan:", alpha_tan + .5)
