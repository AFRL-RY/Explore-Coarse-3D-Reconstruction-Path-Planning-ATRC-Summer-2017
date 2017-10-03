# -*- coding: utf-8 -*-
"""
Created on Wed May 17 15:52:29 2017

@author: student
"""

import gdal, osr
import numpy as np
import glob

# Terrain Directory
directory_terrain = 'terrain'

# Open DTED files (merge if multiple)
ds = gdal.BuildVRT('',glob.glob(directory_terrain + '/*.dt2'))

# GDAL affine transform parameters, According to gdal documentation xoff/yoff are image left corner, a/e are pixel wight/height and b/d is rotation and is zero if image is north up. 
xoff, a, b, yoff, d, e = ds.GetGeoTransform()
def pixel2coord(x, y,a,b,xoff,yoff,d,e):
    """Returns global coordinates from pixel x, y coords"""
    xp = a * x + b * y + xoff
    yp = d * x + e * y + yoff
    return(xp, yp)

# Get upper left corner coordinates to find UTM zone
longitude,latitude = pixel2coord(0,0,a,b,xoff,yoff,d,e)

# Helper functions for UTM transform
def utm_getZone(longitude):
    return (int(1+(longitude+180.0)/6.0))

def utm_isNorthern(latitude):
    if (latitude < 0.0):
        return 0;
    else:
        return 1;

# Find new UTM coordinate system
utm_cs = osr.SpatialReference()
utm_cs.SetWellKnownGeogCS('WGS84')
utm_cs.SetUTM(utm_getZone(longitude),utm_isNorthern(latitude));
dst_wkt = utm_cs.ExportToWkt()
error_threshold = 0.125  # error threshold --> use same value as in gdalwarp
resampling = gdal.GRA_NearestNeighbour

# Transform from WGS84 to UTM coordinates
utm_ds = gdal.AutoCreateWarpedVRT( ds,
                                   None, # src_wkt : left to default value --> will use the one from source
                                   dst_wkt,
                                   resampling,
                                   error_threshold )

# Recalculate affine transformation
xoff, a, b, yoff, d, e = utm_ds.GetGeoTransform()

# Get elevation data
dd = utm_ds.ReadAsArray()
dd[dd<=0] = np.mean(dd[dd>0]) # Set NoData values to average

# Get Northing and Easting Grids
rows,cols = dd.shape
nn = np.zeros(dd.shape)
ee = np.zeros(dd.shape)
for row in  range(0,rows):
    for col in  range(0,cols): 
        east,north = pixel2coord(col,row,a,b,xoff,yoff,d,e)
        nn[row,col] = north
        ee[row,col] = east

#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#
#fig = plt.figure()
#ax = fig.add_subplot(1, 1, 1, projection='3d')
#
##terrain_subsample = terrain.sample(frac=0.01)
#
## Flatten 2D grids to 1D vectors
#nu = np.ravel(nn)
#eu = np.ravel(ee)
#du = np.ravel(dd)
#
#ax.scatter(nu,eu,du,s=1)
