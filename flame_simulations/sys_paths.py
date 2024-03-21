#%% IMPORT STANDARD PACKAGES
import os
import sys

#%% Project parent folder
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

#%% ADD SYS PATHS
# Get a list of all directories within the parent folder
sub_directories = [f.path for f in os.scandir(parent_directory) if f.is_dir()]

# Add each subfolder to sys.path
for sub_directory in sub_directories:
    sys.path.append(sub_directory)
# -*- coding: utf-8 -*-

