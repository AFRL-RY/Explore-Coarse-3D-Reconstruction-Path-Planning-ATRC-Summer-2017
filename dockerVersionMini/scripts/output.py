# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 11:30:17 2017

@author: student
"""
import pandas as pd
import os
import numpy as np
import shutil
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import subprocess
from subprocess import Popen
import logging
from PIL import Image
import glob

def startLogging(directory_images,timestamp):
    output_directory = os.path.join(directory_images,'Results'+timestamp)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    logging.basicConfig(filename=os.path.join(output_directory,'runInfo.log'),level=logging.INFO)
    return output_directory

def writeFrameList(directory_images,metaData,camIdx,output_directory):
    '''
    Write names of selected frames to csv
    '''
    df_output_list = metaData['image_name'][camIdx]
    output_directory = os.path.join(output_directory,'Selected')
    # Create directory if needed
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    filepath = os.path.join(output_directory, 'selected_frame_list.txt')
    print('Writing Frame List...')
    df_output_list.to_csv(filepath,index=False)
    return df_output_list

def geoTagImages(directory_images,metaData,cameras,camIdx,output_directory):
    '''
    This function tags the selected frames with lat lon data from the metadata file.
    This should be run after moveImages.
    Requires that ExifTool be installed and added to the path.
    '''
    # Get required metadata
    positionData = metaData[['image_name','field.lat','field.lon','field.height']]
    positionData = positionData.iloc[camIdx,:]
    dirPath = os.path.join(output_directory,'Selected')
#    positionData['image_name'] =  dirPath + '/' + positionData['image_name'].values
    positionData.columns = ['SourceFile','GPSLatitude','GPSLongitude','GPSAltitude']
    
    # Write to file in Selected folder
    filepath = os.path.join(output_directory,'Selected', 'position_data_latlon.csv')
    print('Writing Position Data File')
    positionData.to_csv(filepath,index=False,sep=',')
    
    # Call ExifTool to geotag images
    print('GeoTagging images')
    directory = os.path.join(output_directory,'Selected')
    p = Popen(['exiftool','-csv='+filepath,'-gpslatituderef=N','-gpslongituderef=W','-gpsaltituderef=above','-gpstrackref=T','-overwrite_original',directory], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Write GPS data to file in UTM for VisualSFM
    # Unpack cameras
    selectedCams = cameras.iloc[camIdx,:]
    gpsData = pd.concat([positionData['SourceFile'],selectedCams['ce'],selectedCams['cn'],selectedCams['cd']],axis=1)
    filepath = os.path.join(output_directory,'Selected', 'position_data_utm.txt')
    gpsData.to_csv(filepath,index=False,header=False,sep=' ')
#    out, err = p.communicate()
#    print(err)
    
    return
    
def moveImages(directory_images,df_output_list,output_directory):
    '''
    Copies selected images to new folder called Selected in image directory
    '''
    print('Copying Images...')
    output_directory = os.path.join(output_directory,'Selected')
    # Create directory if needed
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    # Copy selected images
    for image_name in df_output_list:
        full_file_name = os.path.join(directory_images,image_name)
        if (os.path.isfile(full_file_name)):
            shutil.copy(full_file_name, output_directory)
    
    return

def shrinkImages(directory_images,df_output_list,output_directory):
    '''
    Saves a 25% size copy of the selected images in a subfolder
    '''
    print('Resizing Images')
    smallOut = os.path.join(output_directory,'Selected','small')
    if not os.path.exists(smallOut):
        os.makedirs(smallOut)
    for infile in glob.glob(os.path.join(output_directory,'Selected','*.jpg')):
        im = Image.open(infile)
        ImageWidth,ImageHeight = im.size
        size = int(ImageWidth/4.0),int(ImageHeight/4.0)
        im.thumbnail(size)
        im.save(os.path.join(smallOut,os.path.basename(infile)),"JPEG")

def writePairList(directory_images,c_min,df_output_list,matches,output_directory):
    '''
    Writes a list of overlapping image pairs for VisualSFM
    matches is the number of images to match to each frame.
    '''
    print('Calculating Pair List...')
    # This should correspond to the amount of overlap between images
    distMat = squareform(pdist(c_min, lambda u, v: np.count_nonzero((u==True)&(v==True))))
    frameList = []
    matchList = []
    # Loop through each image and select the images with the most overlap
    for i, row in enumerate(distMat):
        row[i] = 0 # Make sure doesn't select itself
        # Get m matches for each image
        for m in range(matches):
            bestMatch = np.argmax(row)
            matchList.append(bestMatch)
            frameList.append(i)
            row[bestMatch] = 0 # Don't select the same match twice
            distMat[bestMatch,i] = 0 # Don't select the inverse
    # Add this list to the original list of frames i.e. "Image A, Image B"
    frames = pd.DataFrame(df_output_list)
    frames.reset_index(inplace=True,drop=True)
    pairList = pd.DataFrame(frames['image_name']).iloc[frameList]
    pairList.reset_index(inplace=True,drop=True)
    matches = frames['image_name'].iloc[matchList]
    matches.reset_index(inplace=True,drop=True)
    pairList['Image Match'] = matches.values
    # Write to file
    output_directory = os.path.join(output_directory,'Selected')
    # Create directory if needed
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    filepath = os.path.join(output_directory, 'selected_pair_list.txt')
    print('Writing Pair List')
    pairList.to_csv(filepath,index=False,header=False,sep=' ')
    return