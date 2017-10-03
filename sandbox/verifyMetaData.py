# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 10:46:04 2017

@author: student
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from terrain import calcNormals, loadTerrain, cropTerrainBB, cropTerrainKML, cropTerrainFC, addDome
from loadCameras import loadCameras
import yaml
import gdal, osr
import glob
import utm
import os
import matplotlib.image as mpimg

from cartopy import config
import cartopy.crs as ccrs

# Load configuration file
with open('./configFiles/configMedinaV1.yml','r') as ymlfile:
    cfg = yaml.load(ymlfile)
    
# Load Terrain Data
print('Loading terrain data...')
directory_terrain = cfg['dirs']['directory_terrain']
nn,ee,dd = loadTerrain(directory_terrain)

# Load cameras from image metadata
print('Loading image locations and poses...')
directory_images = cfg['dirs']['directory_images']
directory_metadata = cfg['dirs']['directory_metadata']
metaDataType = cfg['dirs']['metadata_type']
vloc,vor,vup,alphas,metaData = loadCameras(directory_images,directory_metadata,metaDataType,nn,ee,dd)
cn,ce,cd = vloc # Camera locations
cor_n,cor_e,cor_d = vor # Camera orientations
cup_n,cup_e,cup_d = vup # Camera up vectors
alphax,alphay = alphas # Field of view values

# load Satellite
# Open geotiff files (merge if multiple)
ds = gdal.BuildVRT('',glob.glob(directory_terrain + '/*.tif'))

## GDAL affine transform parameters, According to gdal documentation xoff/yoff are image left corner, a/e are pixel wight/height and b/d is rotation and is zero if image is north up. 
xoff, a, b, yoff, d, e = ds.GetGeoTransform()
def pixel2coord(x, y,a,b,xoff,yoff,d,e):
    """Returns global coordinates from pixel x, y coords"""
    xp = a * x + b * y + xoff
    yp = d * x + e * y + yoff
    return(xp, yp)

# Get corner coordinates
longitude,latitude = pixel2coord(0,0,a,b,xoff,yoff,d,e)

# Transform from lat lon to UTM Coordinates
def utm_getZone(longitude):
    return (int(1+(longitude+180.0)/6.0))

def utm_isNorthern(latitude):
    if (latitude < 0.0):
        return 0;
    else:
        return 1;

# Define target SRS 
utm_cs = osr.SpatialReference()
utm_cs.SetWellKnownGeogCS('WGS84')
utm_cs.SetUTM(utm_getZone(longitude),utm_isNorthern(latitude));
dst_wkt = utm_cs.ExportToWkt()
error_threshold = 0.125  # error threshold --> use same value as in gdalwarp
resampling = gdal.GRA_NearestNeighbour

# Call AutoCreateWarpedVRT() to fetch default values for target raster dimensions and geotransform
utm_ds = gdal.AutoCreateWarpedVRT( ds,
                                   None, # src_wkt : left to default value --> will use the one from source
                                   dst_wkt,
                                   resampling,
                                   error_threshold )

# Get Satellite data
satImg = ds.ReadAsArray()   
    
satImg = np.transpose(satImg,(1,2,0)) # Reorder to MxNx4 

# Recalculate affine transformation
xoff, a, b, yoff, d, e = ds.GetGeoTransform()

# Get Northing and Easting Grids
rows,cols,depth = satImg.shape
snn = np.zeros(satImg.shape)
see = np.zeros(satImg.shape)
snn = snn[:,:,0]
see = see[:,:,0]
for row in  range(0,rows):
    for col in  range(0,cols): 
        east,north = pixel2coord(col,row,a,b,xoff,yoff,d,e)
        snn[row,col] = north
        see[row,col] = east

# Crop satellite image
pad = 100
# Find bounding box
cam_max_east = ce.max()
cam_min_east = ce.min()
cam_max_north = cn.max()
cam_min_north = cn.min()

# Find rows and columns to crop to
crop_cols = np.where((see[0,:]>(cam_min_east-pad)) & (see[0,:]<(cam_max_east+pad)))
crop_rows = np.where((snn[:,0]>(cam_min_north-pad)) & (snn[:,0]<(cam_max_north+pad)))

# Tuple to 1D arrays so we can use them to select the right subset
crop_cols = crop_cols[0]
crop_rows = crop_rows[0]

# Crop rows
satImg_cropped = satImg[crop_rows,:,:]

# Crop columns
satImg_cropped = satImg_cropped[:,crop_cols,:]

print('Begin inspection')
numberOfComparisons = 3
sampledData = metaData.sample(numberOfComparisons)
for i in range(len(sampledData)):
    row = sampledData.iloc[i,:]
    fig1 = plt.figure()
    ax = plt.axes(projection=ccrs.UTM('11S'))

    img_extent = (cam_min_east, cam_max_east, cam_min_north, cam_max_north)
    
    ax.set_xmargin(0.01)
    ax.set_ymargin(0.01)
    
    ax.imshow(satImg_cropped, origin='upper', extent=img_extent, transform=ccrs.UTM('11S'))
    ax.set_title('Test Image ' + str(i) + ' Four Corners')
    plt.show()
    
    tl_lat = row['image corners tl_latitude']
    tl_lon = row['tl_longitude']
    tr_lat = row['tr_latitude']
    tr_lon = row['tr_longitude']
    bl_lat = row['bl_latitude']
    bl_lon = row['bl_longitude']
    br_lat = row['br_latitude']
    br_lon = row['br_longitude']
    
    ax.scatter(tl_lon,tl_lat,transform=ccrs.Geodetic())
    ax.scatter(tr_lon,tr_lat,transform=ccrs.Geodetic())
    ax.scatter(bl_lon,bl_lat,transform=ccrs.Geodetic())
    ax.scatter(br_lon,br_lat,transform=ccrs.Geodetic())
    
    image_name = row.image_name
    image_name_full = os.path.join(cfg['dirs']['directory_images'],image_name)
    
    fig2 = plt.figure()
    img=mpimg.imread(image_name_full)
    imgplot = plt.imshow(img)
    plt.title('Test Image ' + str(i))
    plt.show()

def cropImage(satImg,minEast,maxEast,minNorth,maxNorth):
    # Find rows and columns to crop to
    crop_cols = np.where((see[0,:]>(cam_min_east-pad)) & (see[0,:]<(cam_max_east+pad)))
    crop_rows = np.where((snn[:,0]>(cam_min_north-pad)) & (snn[:,0]<(cam_max_north+pad)))
    
    # Tuple to 1D arrays so we can use them to select the right subset
    crop_cols = crop_cols[0]
    crop_rows = crop_rows[0]
    
    # For some reason numpy won't select rows and columns by index at the same time.
    # Crop rows
    satImg_cropped = satImg[crop_rows,:,:]
    
    # Crop columns
    satImg_cropped = satImg_cropped[:,crop_cols,:]
    
    return satImg_cropped