# -*- coding: utf-8 -*-
"""
Created on Wed May 24 16:42:50 2017

@author: student
"""

import os
import shutil

def moveImages(directory_images,df_output_list):
    '''
    Copies selected images to new folder called Selected in image directory
    '''
    output_directory = os.path.join(directory_images,'Selected')
    # Create directory if needed
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    # Copy selected images
    for image_name in df_output_list:
        full_file_name = os.path.join(directory_images,image_name)
        if (os.path.isfile(full_file_name)):
            print('Copying Image ' + str(image_name))
            shutil.copy(full_file_name, output_directory)
    
    return