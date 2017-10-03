# -*- coding: utf-8 -*-
"""
Created on Thu May 18 11:52:51 2017

@author: student
"""
import pandas as pd
import numpy as np
import utm
import collections
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from terrain import cropTerrainBB
import glob
import os
from PIL import Image

def loadCameras(directory_images,directory_metadata,metaDataType,terrain):
    '''
    loadCameras loads camera locations and orientations from a set of images and matching metadata
    This is a variation specifically for the Medina Wasl images and metadata
    INPUTS
    directory_images - Location of folder holding images
    directory_metadata - Location of folder holding matching mark_1_pva metadata in csv format
    OUTPUTS
    cn, ce, ce - camera north east and down positions (m)
    cor_n, cor_e, cor_d - camera center orientation vector, or pointing direction (m)
    cup_n, cup_e, cup_d - camera up vector, or the vector pointing out the top of the camera (m)
    alphax - camera horizontal field of view (degrees)
    alphay - camera vertical field of view (degrees)
    df_all - camera metadata read from mark_1_pva.csv
    '''
    # Load metadata
    if(metaDataType=='Muscat'):
        df_all = readMetaDataMuscat(directory_images,directory_metadata)
    elif(metaDataType=='Medina'):
        df_all = readMetaDataMedina(directory_images,directory_metadata)
    else:
        print('Invalid Metadata Type.')
        sys.exit()
        
        
    # Convert lat/lon to UTM coordinates
    lat = df_all['field.lat'].values
    lon = df_all['field.lon'].values
    cn = []
    ce = []
    for row in np.c_[lat,lon]:
        east,north,zone,zone_letter = utmConversion(row)
        cn.append(north)
        ce.append(east)
    cn = np.asarray(cn)
    ce = np.asarray(ce)
    cd = df_all['field.height'].values
    
    if(metaDataType=='Muscat'):
        # Load sensor parameters from parameter file
        SensorYaw, SensorPitch, SensorRoll, ImageHeight, ImageWidth, FocalLength = getSensorParamsMuscat(directory_images,directory_metadata)
        df_all['sensor_yaw'] = SensorYaw
        df_all['sensor_pitch'] = SensorPitch
        df_all['sensor_roll'] = SensorRoll
        df_all['focal_length'] = FocalLength
        df_all['ImageHeight'] = ImageHeight
        df_all['ImageWidth'] = ImageWidth
        
        # Get field of view angles
        alphax_val = np.rad2deg(2*np.arctan(ImageWidth/(2*FocalLength)))
        alphay_val = np.rad2deg(2*np.arctan(ImageHeight/(2*FocalLength)))
        alphax = np.ones(len(df_all))*alphax_val
        alphay = np.ones(len(df_all))*alphay_val
    elif(metaDataType=='Medina'):
        ImageHeight, ImageWidth = getSensorParamsMedina(directory_images)
        df_all['ImageHeight'] = ImageHeight
        df_all['ImageWidth'] = ImageWidth
        alphax = df_all['HorizontalFOV']
        alphay = df_all['VerticalFOV']
        #TODO: Check on focal length for Medina data
    
    # Loop through all cameras and compute their orientations
    cor_n = np.zeros(len(df_all))
    cor_e = np.zeros(len(df_all))
    cor_d = np.zeros(len(df_all))
    cup_n = np.zeros(len(df_all))
    cup_e = np.zeros(len(df_all))
    cup_d = np.zeros(len(df_all))
    if(metaDataType=='Muscat'):
        for i,row in enumerate(df_all['field.azimuth'].values):
            Yaw = np.deg2rad(df_all['field.azimuth'][i])
            Pitch = np.deg2rad(df_all['field.pitch'][i])
            Roll = np.deg2rad(df_all['field.roll'][i])
            SensorYaw = df_all['sensor_yaw'][i]
            SensorPitch = df_all['sensor_pitch'][i]
            SensorRoll = df_all['sensor_roll'][i]
            ImageHeight = df_all['ImageHeight'][i]
            ImageWidth = df_all['ImageWidth'][i]
            FocalLength = df_all['focal_length'][i]
        
            projectionContext = collections.namedtuple('projectionContext',' \
            Yaw, Pitch, Roll, SensorYaw, SensorPitch, SensorRoll, ImageHeight, \
            ImageWidth,FocalLength')
            
            pc = projectionContext(Yaw,Pitch,Roll,
                               SensorYaw,SensorPitch,SensorRoll,ImageHeight,
                               ImageWidth,FocalLength)
            
            cor_n[i],cor_e[i],cor_d[i],cup_n[i],cup_e[i],cup_d[i] = cameraOrientations(pc)
    elif(metaDataType=='Medina'):
        # Get average height of ground in flight area
        nn_cropped,ee_cropped,dd_cropped = cropTerrainBB(terrain, cn, ce, 0)
        #TODO: This should probably be on a per image basis
        ground_z = np.mean(dd_cropped)
        for i,row in df_all.iterrows():
            # Get four corners
            tl = row[['image corners tl_latitude','tl_longitude']].values
            tr = row[['tr_latitude','tr_longitude']].values
            bl = row[['bl_latitude','bl_longitude']].values
            br = row[['br_latitude','br_longitude']].values
            
            # Convert four corners to UTM
            tle,tln,z,zl = utmConversion(tl)
            tre,trn,z,zl = utmConversion(tr)
            ble,bln,z,zl = utmConversion(bl)
            bre,brn,z,zl = utmConversion(br)
            
            # Get camera location
            cam = np.r_[cn[i],ce[i],cd[i]]
            
            # Get image center point
            center = np.r_[np.mean([tln,trn,bln,brn]),np.mean([tle,tre,ble,bre]),ground_z]
            # Get point on side of image
            side = np.r_[np.mean([trn,brn]),np.mean([tre,bre]),ground_z]
            
            # Vector from center of image to camera location
            cor_vector = center - cam
            cor_vector = cor_vector/np.linalg.norm(cor_vector)
            
            # Vector from side of image to camera location
            cside_vector = side - cam
            cside_vector = cside_vector/np.linalg.norm(cside_vector)
            
            # Get camera up vector using cross product of side and center vectors
            cup_vector = np.cross(cor_vector,cside_vector)
            if(cup_vector[2]<0): # Make sure we get the up vector and not the down vector
                cup_vector = np.cross(cside_vector,cor_vector)
                
            cor_n[i],cor_e[i],cor_d[i] = cor_vector
            cup_n[i],cup_e[i],cup_d[i] = cup_vector
    else:
        print('Invalid MetaData Type')
        sys.exit()

    # Pack camera information
    cameras = pd.DataFrame({'cn':cn,
                            'ce':ce,
                            'cd':cd,
                            'cor_n':cor_n,
                            'cor_e':cor_e,
                            'cor_d':cor_d,
                            'cup_n':cup_n,
                            'cup_e':cup_e,
                            'cup_d':cup_d,
                            'alphax':alphax,
                            'alphay':alphay})

    return cameras,df_all

def utmConversion(row):
    '''
    Converts from lat lon coordinates to UTM coordinates
    '''
    
    east, north, zone, zone_letter = utm.from_latlon(row[0],row[1])
    
    return east, north, zone, zone_letter

def cameraOrientations(pc,plotCorners=False):
    '''
    Applies rotation matrices to calculate the vector along which a camera is pointing
    '''
    # form the rotation matrix for the platform roll, pitch, yaw
    pr = np.array([[1, 0, 0],
                    [0, np.cos(pc.Roll), np.sin(pc.Roll)],
                    [0, -np.sin(pc.Roll), np.cos(pc.Roll)]])
    pp = np.array([[np.cos(pc.Pitch), 0, -np.sin(pc.Pitch)],
                    [0, 1, 0],
                    [np.sin(pc.Pitch), 0, np.cos(pc.Pitch)]])
    py = np.array([[np.cos(pc.Yaw), np.sin(pc.Yaw), 0],
                    [-np.sin(pc.Yaw), np.cos(pc.Yaw), 0],
                    [0, 0, 1]])
    R_ned2ins = pr.dot(pp).dot(py)

    # form the rotation matrix for the sensor relative roll, pitch, yaw
    sr = np.array([[1, 0, 0],
                  [0, np.cos(pc.SensorRoll), np.sin(pc.SensorRoll)],
                  [0, -np.sin(pc.SensorRoll), np.cos(pc.SensorRoll)]])
    sp = np.array([[np.cos(pc.SensorPitch), 0, -np.sin(pc.SensorPitch)],
                  [0, 1, 0],
                  [np.sin(pc.SensorPitch), 0, np.cos(pc.SensorPitch)]])
    sy = np.array([[np.cos(pc.SensorYaw), np.sin(pc.SensorYaw), 0],
                  [-np.sin(pc.SensorYaw), np.cos(pc.SensorYaw), 0],
                  [0, 0, 1]])
    R_ins2sensor = sr.dot(sp).dot(sy)
    
    # rotation matrix that defines the transformation between the sensor and
    # camera frame
    R_sensor2cam = np.array([[0,1,0],
                            [0,0,1],
                            [1,0,0]])
    
    # full rotation matrix from NED to the camera frame
    R_ned2cam = R_sensor2cam.dot(R_ins2sensor).dot(R_ned2ins)

    # camera matrix
    K = np.array([[pc.FocalLength,0,pc.ImageWidth/2],
                 [0,pc.FocalLength,pc.ImageHeight/2],
                 [0,0,1]])
    
    # Get camera direction vector
    cor_vector = np.transpose(R_ned2cam).dot(np.linalg.inv(K)).dot([pc.ImageWidth/2.0, pc.ImageHeight/2.0, 1])
    
    # Get side vector
    cside_vector = np.transpose(R_ned2cam).dot(np.linalg.inv(K)).dot([pc.ImageWidth, pc.ImageHeight/2.0, 1])
    
    # Get top vector
#    ctop_vector = np.transpose(R_ned2cam).dot(np.linalg.inv(K)).dot([pc.ImageWidth/2.0, pc.ImageHeight, 1])
    
    # Get camera up vector
    cup_vector = np.cross(cor_vector,cside_vector)
    if(-cup_vector[2]<0): # Make sure we get the up vector and not the down vector
        cup_vector = np.cross(cside_vector,cor_vector)
        
#    upperLeft = np.transpose(R_ned2cam).dot(np.linalg.inv(K)).dot([0, 0, 1])
#    upperRight = np.transpose(R_ned2cam).dot(np.linalg.inv(K)).dot([pc.ImageWidth, 0, 1])
#    lowerRight = np.transpose(R_ned2cam).dot(np.linalg.inv(K)).dot([pc.ImageWidth, pc.ImageHeight, 1])
#    lowerLeft = np.transpose(R_ned2cam).dot(np.linalg.inv(K)).dot([0, pc.ImageHeight, 1])
    
    # Get normalized camera direction vector
    cor_n = cor_vector[0]/np.linalg.norm(cor_vector)
    cor_e = cor_vector[1]/np.linalg.norm(cor_vector)
    cor_d = -cor_vector[2]/np.linalg.norm(cor_vector)
    
    # Get normalized camera up vector
    cup_n = cup_vector[0]/np.linalg.norm(cup_vector)
    cup_e = cup_vector[1]/np.linalg.norm(cup_vector)
    cup_d = -cup_vector[2]/np.linalg.norm(cup_vector)
    
#    if(plotCorners==True):
#        upperLeft = np.transpose(R_ned2cam).dot(np.linalg.inv(K)).dot([0, 0, 1])
#        upperRight = np.transpose(R_ned2cam).dot(np.linalg.inv(K)).dot([pc.ImageWidth, 0, 1])
#        lowerRight = np.transpose(R_ned2cam).dot(np.linalg.inv(K)).dot([pc.ImageWidth, pc.ImageHeight, 1])
#        lowerLeft = np.transpose(R_ned2cam).dot(np.linalg.inv(K)).dot([0, pc.ImageHeight, 1])
#        return cor_n,cor_e,cor_d,cup_n,cup_e,cup_d,upperLeft,upperRight,lowerRight,lowerLeft
    
    return cor_n,cor_e,cor_d,cup_n,cup_e,cup_d

def readMetaDataMuscat(directory_images,directory_metadata):
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

def readMetaDataMedina(directory_images,directory_metadata):
    '''
    readMetaData searches for matching metadata for all images in the given directory.
    It returns a pandas DataFrame with all image paths and available mark_1_pva metadata in csv
    '''

#    directory_images = r'E:\movies\Mosque\MosqueImages'
#    directory_metadata = r'E:\movies\Mosque\MosqueMetaData'
    
    # First get a list of the unique video file names the images are drawn from
    # Get image names
    image_names = [os.path.basename(x) for x in glob.glob(directory_images + '/*.jpg')]
    # Get video file names
    vid_names = list(set([i.split('.')[0] for i in image_names]))
    
    df_all = pd.DataFrame()
    for vid in vid_names:
        # Get image names
        image_names = [os.path.basename(x) for x in glob.glob(directory_images + '/'+ vid +'*.jpg')]
        df_img = pd.DataFrame(image_names,columns=['image_name']) # Initialize dataframe
        
        # Get frame numbers
        image_numbers = [i[-10:-4] for i in image_names]
        df_img['image_number'] = image_numbers # Add image numbers to dataframe
        df_img['image_number'] = pd.to_numeric(df_img['image_number']) # Convert to int so we can match with sequence
        
        # Load metadata files and concatenate into one
        metadata_file = [os.path.basename(x) for x in glob.glob(directory_metadata + '/'+ vid +'*.csv')]
        df_meta = pd.read_csv(directory_metadata + '/' + metadata_file[0])
        # Drop rows without four corners data
        df_meta = df_meta[df_meta.tr_latitude.notnull()]
        # Drop rows with zero field of view
        df_meta = df_meta[df_meta.HorizontalFOV>0]
        df_meta = df_meta[df_meta.VerticalFOV>0]
        
        # Merge data with image file names based on sequence number
        df_vid_all = pd.merge(df_img,df_meta,left_on='image_number',right_on='frame_number',how='inner')
    
        # Concatenate to full list
        df_all = df_all.append(df_vid_all,ignore_index=True)
    
    # Change column names to match Muscatatuck data
    df_all.rename(columns={'sensor latitude':'field.lat','sensor longitude':'field.lon','sensor altitude':'field.height'},inplace=True)
    df_all.rename(columns={'platform yaw':'field.azimuth','platform pitch':'field.pitch','platform roll':'field.roll'},inplace=True)
    df_all.rename(columns={'sensor relative yaw':'sensor_yaw','sensor relative pitch':'sensor_pitch','sensor relative roll':'sensor_roll'},inplace=True)

    return df_all

def getSensorParamsMuscat(directory_images,directory_metadata):
    '''
    Reads the sensor yaw, pitch, roll and focal ratio from parameters file.
    Also returns image height and width from first image in image directory
    '''

    
    # Get image size
    image_names = glob.glob(directory_images + '/*.jpg')
    im = Image.open(image_names[0])
    ImageWidth,ImageHeight = im.size
    
    # Read Parameters from Parameter Output file
    parameter_file = glob.glob(directory_metadata + '/*ParameterOutput.csv')[0]
    with open(parameter_file, 'r') as inF:
        for line in inF:
            # Search for Focal Length
            if 'CameraFocalLength' in line:
                # Split the number out from everything else
                before_keyword, keyword, after_keyword = line.partition('CameraFocalLength')
                temp1 = after_keyword.split(',')
                temp2 = temp1[0].split(':')
                FocalLength = float(temp2[1])
                break
        for line in inF:
            # Search for Sensor Roll Pitch and Yaw
            if 'imu_to_camera_pitch' in line:
                # Split the pitch out from everything else
                before_keyword, keyword, after_keyword = line.partition('imu_to_camera_pitch')
                temp1 = after_keyword.split(',')
                temp2 = temp1[0].split(':')
                SensorPitch = float(temp2[1])
                # Split the roll out from everything else
                before_keyword, keyword, after_keyword = line.partition('imu_to_camera_roll')
                temp1 = after_keyword.split(',')
                temp2 = temp1[0].split(':')
                SensorRoll = float(temp2[1])
                # Split the yaw out from everything else
                before_keyword, keyword, after_keyword = line.partition('imu_to_camera_yaw')
                temp1 = after_keyword.split(',')
                temp2 = temp1[0].split(':')
                SensorYaw = float(temp2[1])
                break
    # Convert to radians
    SensorYaw   = np.deg2rad(SensorYaw)
    SensorPitch = np.deg2rad(SensorPitch)
    SensorRoll = np.deg2rad(SensorRoll)
    
    return SensorYaw, SensorPitch, SensorRoll, ImageHeight, ImageWidth, FocalLength

def getSensorParamsMedina(directory_images):
    '''
    Returns image height and width from first image in image directory
    '''
    
    # Get image size
    image_names = glob.glob(directory_images + '/*.jpg')
    image_names.extend(glob.glob(directory_images + '/*.png'))
    im = Image.open(image_names[0])
    ImageWidth,ImageHeight = im.size
    
    return ImageHeight, ImageWidth

if __name__ == "__main__":
    print('Test')
#    directory_images = 'E:/21_Sept_2015_WAMI_Flight_1/2015.09.21_15.39.54'
#    directory_metadata = 'E:/Bags/20150921_Flight1'
#    cn,ce,cd,cor_n,cor_e,cor_d,alphax,alphay,df_all,cup_n,cup_e,cup_d = loadCameras(directory_images,directory_metadata)
#    
#    import matplotlib.pyplot as plt
#    from mpl_toolkits.mplot3d import Axes3D
#    from plotAll import plotSection
    
#    crange = range(10)
#    plotSection(cn,ce,cd,cor_n,cor_e,cor_d,cup_n,cup_e,cup_d,crange)
    
    # Plot camera positions
#    fig = plt.figure()
#    ax = fig.add_subplot(1, 1, 1, projection='3d')
#    ax.scatter(cn,ce,cd, s=1)
    
    # Plot camera vectors
#    fig = plt.figure()
#    ax = fig.add_subplot(1, 1, 1, projection='3d')
#    plt.quiver(cn,ce,cd,cor_n,cor_e,cor_d,color='g',pivot='tail',length=75)

    