# -*- coding: utf-8 -*-
"""
Created on Wed May 17 15:27:22 2017

@author: student
"""

import numpy as np
import pandas as pd
import glob
import os

def readMetaData(directory_images,directory_metadata):
    '''
    readMetaData searches for matching metadata for all images in the given directory.
    It returns a pandas DataFrame with all image paths and available mark_1_pva metadata in csv
    '''

    #directory_images = 'E:/21_Sept_2015_WAMI_Flight_1/2015.09.21_15.39.54'
    #directory_metadata = 'E:/Bags/20150921_Flight1'
    
    # Get image names
    image_names = [os.path.basename(x) for x in glob.glob(directory_images + '/*.jpg')]
    df_img = pd.DataFrame(image_names,columns=['image_name']) # Initialize dataframe
    
    # Get frame numbers
    image_numbers = [i[5:10] for i in image_names]
    df_img['image_number'] = image_numbers # Add image numbers to dataframe
    df_img['image_number'] = pd.to_numeric(df_img['image_number']) # Convert to int so we can match with sequence
    
    # Load metadata
    metadata_file = [os.path.basename(x) for x in glob.glob(directory_metadata + '/*mark_1_pva.csv')]
    df_meta = pd.read_csv(directory_metadata + '/' + metadata_file[0])
    
    # Merge data with image file names based on sequence number
    df_all = pd.merge(df_img,df_meta,left_on='image_number',right_on='field.header.seq',how='left')
    
    return df_all

if __name__ == "__main__":
    directory_images = 'E:/21_Sept_2015_WAMI_Flight_1/2015.09.21_15.39.54'
    directory_metadata = 'E:/Bags/20150921_Flight1'
    df_all = readMetaData(directory_images,directory_metadata)
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # 3D Line Plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plt.plot(df_all['field.lat'],df_all['field.lon'],df_all['field.height'])
    
    # 3D Scatter Plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(df_all['field.lat'],df_all['field.lon'],df_all['field.height'], s=1)
