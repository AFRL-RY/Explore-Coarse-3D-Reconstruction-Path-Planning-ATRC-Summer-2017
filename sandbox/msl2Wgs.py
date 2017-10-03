# -*- coding: utf-8 -*-
"""
Created on Tue May 23 10:06:02 2017

@author: student
"""

import numpy as np

def msl2Wgs(lat,lon):
    '''
    Calculates adjustment factor to change vertical datum of elevation data 
    from MSL (E96) to WGS84.  Returns offset value in meters
    To Convert from MSL to WGS84
        alt_WGS = alt_MSL + offset
    To Convert from WGS84 to MSL
        alt_MSL = alt_WGS - offset
    Adapted from Location.cpp (Clark Taylor)
    '''
    # Warning, this will return if right at the North/South pole
    if ( lat <= -90.0 or lat > 90.0 or lon <= -360.0 or lon >= 360.0): 
        return 0;
    
    if(lon < 0.0):
        my_lon = lon + 360.0
    else:
        my_lon = lon

    lon_idx = int(my_lon * 2.0)
    lat_idx = int((lat - 90.0) * -2.0)

    lon_diff = (my_lon * 2.0) - lon_idx
    lat_diff = ((lat - 90.0) *-2.0) - lat_idx
    
    # Load adjustment data from csv
    msl_adjust = np.genfromtxt('msl_adjust.csv',delimiter=',')

    offset = (msl_adjust[lat_idx,lon_idx] * (1-lon_diff) * (1-lat_diff)) + \
    (msl_adjust[lat_idx + 1,lon_idx] * (1.0 - lon_diff) * lat_diff) + \
    (msl_adjust[lat_idx + 1,lon_idx + 1] * (lon_diff) * lat_diff) + \
    (msl_adjust[lat_idx,lon_idx + 1] * (lon_diff) * (1.0 - lat_diff))
    
    return offset