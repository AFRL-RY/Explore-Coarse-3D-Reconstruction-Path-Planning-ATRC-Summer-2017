# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 22:54:12 2016

@author: oacom
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import griddata
import gdal, osr
import glob
from scipy.stats import multivariate_normal
import xml.etree.ElementTree as ET
import utm
from plyfile import PlyData

def genTerrain(nmax,emax,dmax,res):
    '''
    Generates random 'terrain' grid for testing
    INPUTS
    nmax: maximum north value
    emax: maximum east value
    dmax: maximum elevation
    res: number of grid points per axis
    OUTPUT
    Returns north east and elevation values as three 2D grids
    '''

    # Generate grid ranges and elevation values
    n = np.linspace(0,nmax,res)
    e = np.linspace(0,emax,res)
    d = np.random.rand(res,res)*dmax
    
    # Convert grid ranges to 2D grid
    nn,ee = np.meshgrid(n,e)
    dd = d

    terrain = pd.DataFrame
    terrain.nn = nn
    terrain.ee = ee
    terrain.dd = dd
    
    return terrain

def loadTerrain(directory_terrain,cfg):
    '''
    Loads a terrain data file using GDAL
    INPUTS
    directory_terrain: Path to a folder containing one or more DTED files in .dt2 format
    OUTPUT
    Returns north east and elevation values as three 2D grids
    '''
    
    if(cfg['terrain']['source']=='DTED'):
        
        # Open DTED files (merge if multiple)
        ds = gdal.BuildVRT('',glob.glob(directory_terrain + '/*.dt2'))
        
        # GDAL affine transform parameters, According to gdal documentation xoff/yoff are image left corner, a/e are pixel wight/height and b/d is rotation and is zero if image is north up. 
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
                return 0
            else:
                return 1
        
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
        
        # Recalculate affine transformation
        xoff, a, b, yoff, d, e = utm_ds.GetGeoTransform()
        
        # Get elevation data
        dd = utm_ds.ReadAsArray()    
        
        # Calculate vertical datum adjustment from MSL to WGS84
        msl2Wgs_adjustment = msl2Wgs(latitude,longitude)
        dd = dd + msl2Wgs_adjustment
        
        # Get Northing and Easting Grids
        rows,cols = dd.shape
        nn = np.zeros(dd.shape)
        ee = np.zeros(dd.shape)
        for row in  range(0,rows):
            for col in  range(0,cols): 
                east,north = pixel2coord(col,row,a,b,xoff,yoff,d,e)
                nn[row,col] = north
                ee[row,col] = east
    elif(cfg['terrain']['source']=='PLY'):
        filename = glob.glob(directory_terrain + '/*.ply')[0]
        nn,ee,dd = loadply(filename)

    terrain = pd.DataFrame
    terrain.nn = nn
    terrain.ee = ee
    terrain.dd = dd

    return terrain

def fixNoData(terrain):
    '''
    Replaces NoData values in input DTED using inpainting
    '''
    dd = terrain.dd
    dd[dd<=0] = np.nan
    dd = replace_nans(dd)
    return dd
   
def interpTerrain(terrain,ifactor):
    '''
    Interpolates terrain to a desired resolution using a piecewise cubic
    '''
    # Unpack terrain
    nn = terrain.nn
    ee = terrain.ee
    dd = terrain.dd

    # Flatten 2D grids to 1D vectors
    nu = np.ravel(nn)
    eu = np.ravel(ee)
    du = np.ravel(dd)
    
    # Put data points in required vector format
    points = np.c_[nu,eu]
    values = du.T
    # Create new grid (complex numbers are to make the new grid inclusive of the end points)
    nn, ee = np.mgrid[min(nu):max(nu):complex(0,nn.shape[0]*ifactor), min(eu):max(eu):complex(0,ee.shape[0])*ifactor]
    # Interpolate elevations
    dd = griddata(points, values, (nn, ee), method='cubic')

    # Repack terrain
    terrain.nn = nn
    terrain.ee = ee
    terrain.dd = dd

    return terrain

def cropTerrain(terrain, cameras, cfg):
    if (cfg['crop']['cropType'] == 'KML'):
        pad = cfg['crop']['pad']  # extra border size around flight area (m)
        terrain = cropTerrainKML(terrain, pad, cfg['crop']['path_POI'])
    elif (cfg['crop']['cropType'] == 'FC'):
        pad = cfg['crop']['pad']  # extra border size around flight area (m)
        tl = cfg['crop']['FC_tl']
        tr = cfg['crop']['FC_tr']
        bl = cfg['crop']['FC_bl']
        br = cfg['crop']['FC_br']
        terrain = cropTerrainFC(terrain, pad, tl, tr, bl, br)
#        terrain = interpTerrain(terrain, 10)
#        pad = 10
#        terrain = cropTerrainFC(terrain, pad, tl, tr, bl, br)
    elif (cfg['crop']['cropType'] == 'BB'):
        pad = cfg['crop']['pad']  # extra border size around flight area (m)
        terrain = cropTerrainBB(terrain, cameras.cn, cameras.ce, pad)
    elif (cfg['crop']['cropType'] == 'INTERP'):
        pad = cfg['crop']['pad']  # extra border size around flight area (m)
        terrain = cropTerrainKML(terrain, pad, cfg['crop']['path_POI'])
        terrain = interpTerrain(terrain, 10)
        pad = 0
        terrain = cropTerrainKML(terrain, pad, cfg['crop']['path_POI'])
    else:
        print('Invalid Crop Type.')
    return terrain

def cropTerrainBB(terrain, cn, ce, pad):
    '''
    Crops terrain to the bounding box of the flight area, plus a user specified padding on each side.
    INPUTS
    nn: Terrain north values
    ee: Terrain east values
    dd: Terrain elevation values
    cn: Camera north values
    ce: Camera east values
    buffer: Padding on bounding box crop in each direction (m)
    OUTPUTS
    nn_cropped,ee_cropped,dd_cropped: 2D grids of cropped north, east and elevation data
    '''

    # Find bounding box
    cam_max_east = ce.max()
    cam_min_east = ce.min()
    cam_max_north = cn.max()
    cam_min_north = cn.min()
    
    # Find rows and columns to crop to
    crop_cols = np.where((terrain.ee[0,:]>(cam_min_east-pad)) & (terrain.ee[0,:]<(cam_max_east+pad)))
    crop_rows = np.where((terrain.nn[:,0]>(cam_min_north-pad)) & (terrain.nn[:,0]<(cam_max_north+pad)))
    
    # Tuple to 1D arrays so we can use them to select the right subset
    crop_cols = crop_cols[0]
    crop_rows = crop_rows[0]
    
    # For some reason numpy won't select rows and columns by index at the same time.
    # Crop rows
    nn_cropped = terrain.nn[crop_rows,:]
    ee_cropped = terrain.ee[crop_rows,:]
    dd_cropped = terrain.dd[crop_rows,:]
    
    # Crop columns
    nn_cropped = nn_cropped[:,crop_cols]
    ee_cropped = ee_cropped[:,crop_cols]
    dd_cropped = dd_cropped[:,crop_cols]
    
    # Fix NoData values if present
    if(terrain.dd.any()<0):
        dd_cropped = replace_nans(terrain)

    # Repack terrain
    terrain.nn = nn_cropped
    terrain.ee = ee_cropped
    terrain.dd = dd_cropped

    return terrain

def cropTerrainKML(terrain,pad,file_path):
    '''
    Crops terrain to the bounding box of the input KML area, plus a user specified padding on each side.
    INPUTS
    nn: Terrain north values
    ee: Terrain east values
    dd: Terrain elevation values
    file_path: path to kml file containing a single polygon marking the area of interest
    pad: Padding on bounding box crop in each direction (m)
    OUTPUTS
    nn_cropped,ee_cropped,dd_cropped: 2D grids of cropped north, east and elevation data
    '''
    # Find bounding box
    northList,eastList = readKML(file_path)
    max_east = max(eastList)
    min_east = min(eastList)
    max_north = max(northList)
    min_north = min(northList)
    
    # Find rows and columns to crop to
    crop_cols = np.where((terrain.ee[0,:]>(min_east-pad)) & (terrain.ee[0,:]<(max_east+pad)))
    crop_rows = np.where((terrain.nn[:,0]>(min_north-pad)) & (terrain.nn[:,0]<(max_north+pad)))
    
    # Tuple to 1D arrays so we can use them to select the right subset
    crop_cols = crop_cols[0]
    crop_rows = crop_rows[0]
    
    # For some reason numpy won't select rows and columns by index at the same time.
    # Crop rows
    nn_cropped = terrain.nn[crop_rows,:]
    ee_cropped = terrain.ee[crop_rows,:]
    dd_cropped = terrain.dd[crop_rows,:]
    
    # Crop columns
    nn_cropped = nn_cropped[:,crop_cols]
    ee_cropped = ee_cropped[:,crop_cols]
    dd_cropped = dd_cropped[:,crop_cols]
    
    # Fix NoData values if present
    if(terrain.dd.any()<0):
        dd_cropped = replace_nans(terrain)

    # Repack terrain
    terrain.nn = nn_cropped
    terrain.ee = ee_cropped
    terrain.dd = dd_cropped

    return terrain

def cropTerrainFC(terrain,pad,tl,tr,bl,br):
    '''
    Crops terrain given the four image corners in north and east, plus a user specified padding on each side.
    INPUTS
    nn: Terrain north values
    ee: Terrain east values
    dd: Terrain elevation values
    tl,tr,bl,br: Image four corners, each containing an north and an east value
    pad: Padding on bounding box crop in each direction (m)
    OUTPUTS
    nn_cropped,ee_cropped,dd_cropped: 2D grids of cropped north, east and elevation data
    '''
    # Find bounding box
    latList = [tl[0],tr[0],bl[0],br[0]]
    lonList = [tl[1],tr[1],bl[1],br[1]]
    eastList = np.zeros(4)
    northList = np.zeros(4)
    for i in range(4):
        eastList[i], northList[i], zone, zone_letter = utm.from_latlon(latList[i],lonList[i])
    max_east = max(eastList)
    min_east = min(eastList)
    max_north = max(northList)
    min_north = min(northList)
    
    # Find rows and columns to crop to
    crop_cols = np.where((terrain.ee[0,:]>(min_east-pad)) & (terrain.ee[0,:]<(max_east+pad)))
    crop_rows = np.where((terrain.nn[:,0]>(min_north-pad)) & (terrain.nn[:,0]<(max_north+pad)))
    
    # Tuple to 1D arrays so we can use them to select the right subset
    crop_cols = crop_cols[0]
    crop_rows = crop_rows[0]
    
    # For some reason numpy won't select rows and columns by index at the same time.
    # Crop rows
    nn_cropped = terrain.nn[crop_rows,:]
    ee_cropped = terrain.ee[crop_rows,:]
    dd_cropped = terrain.dd[crop_rows,:]
    
    # Crop columns
    nn_cropped = nn_cropped[:,crop_cols]
    ee_cropped = ee_cropped[:,crop_cols]
    dd_cropped = dd_cropped[:,crop_cols]
    
    # Fix NoData values if present
    if(terrain.dd.any()<0):
        dd_cropped = replace_nans(terrain)

    # Repack terrain
    terrain.nn = nn_cropped
    terrain.ee = ee_cropped
    terrain.dd = dd_cropped

    return terrain
    
def calcNormals(terrain):
    '''
    Calculates normals of terrain points
    INPUTS
    nn: north values
    ee: east values
    dd: elevation values
    OUTPUT
    Returns 3xn array of normal vectors
    '''
    # Construct convolution matrices to calculate the two gradient vectors that cross 
    # each terrain point using a modified first order central difference
    xconv = np.array([[0,0,0],
                     [-1,0,1],
                     [0,0,0]])
    yconv = np.array([[0,-1,0],
                     [0,0,0],
                     [0,1,0]])
    # Calculate gradient vectors for north, east and elevation 
    nnx = signal.convolve2d(terrain.nn, xconv, boundary='symm', mode='same')
    nny = signal.convolve2d(terrain.nn, yconv, boundary='symm', mode='same')
    eex = signal.convolve2d(terrain.ee, xconv, boundary='symm', mode='same')
    eey = signal.convolve2d(terrain.ee, yconv, boundary='symm', mode='same')
    ddx = signal.convolve2d(terrain.dd, xconv, boundary='symm', mode='same')
    ddy = signal.convolve2d(terrain.dd, yconv, boundary='symm', mode='same')
    
    # Prepare vectors for cross product
    xs = np.c_[nnx.ravel(),eex.ravel(),ddx.ravel()].T
    ys = np.c_[nny.ravel(),eey.ravel(),ddy.ravel()].T
               
    # Calculate normals using the cross product of the two gradient vectors
    normals = np.cross(ys,xs,axis=0)
    if(np.mean(normals[2,:])<0):
        normals = np.cross(xs,ys,axis=0)
    normals = normals/np.linalg.norm(normals,axis=0)
    norm_n = normals[0,:].T
    norm_e = normals[1,:].T
    norm_d = normals[2,:].T
    
    normals = pd.DataFrame({'norm_n':norm_n,
                            'norm_e':norm_e,
                            'norm_d':norm_d})
    
    return normals

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
        return 0
    
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

def readKML(kml_filepath):
    '''
    This function takes the filepath of a kml containing a single polygon,
    and returns lists of the north and east coordinates of the vertices.
    '''

    
#    kml_filepath = './POI/muscat.kml'
    
    tree = ET.parse(kml_filepath) 
    #lineStrings = tree.findall('.//{http://www.opengis.net/kml/2.2}LinearRing')
    namespace = tree.getroot().tag
    namespace = namespace[namespace.find("{")+1:namespace.find("}")]
    lineStrings = tree.findall('.//{' + str(namespace) + '}LinearRing')
    
    eastList = []
    northList =[]
    for attributes in lineStrings:
        for subAttribute in attributes:
            if subAttribute.tag == '{http://www.opengis.net/kml/2.2}coordinates':
                for line in subAttribute.text.split():
                    lon,lat,alt = line.split(',')
                    east, north, zone, zone_letter = utm.from_latlon(float(lat),float(lon))
                    eastList.append(east)
                    northList.append(north)
    return northList, eastList

def addDome(terrain,A):
    n0 = np.mean(terrain.nn)
    e0 = np.mean(terrain.ee)
    sigma_n = (np.max(terrain.nn) - n0)*0.75
    sigma_e = (np.max(terrain.ee) - e0)*0.75
    terrain.dd = terrain.dd + A*np.exp(-((terrain.nn-n0)**2/(2*sigma_n**2)+(terrain.ee-e0)**2/(2*sigma_e**2)))
    return terrain

def makeGaussian(size, sigma):
    x, y = np.mgrid[0:size:1, 0:size:1]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    rv = multivariate_normal(mean=[size/2,size/2], cov=[[sigma,0],[0,sigma]])
    return rv.pdf(pos)/np.sum(rv.pdf(pos))

def replace_nans(array, max_iter=50, tol=0.05, kernel_radius=2, kernel_sigma=2, method='idw'):
    # Original inpainting code (replace_nans) by Davide Lasagna https://github.com/gasagna/openpiv-python/blob/master/openpiv/src/lib.pyx
    # Cython removed and Gaussian kernel code added by opit (https://github.com/astrolitterbox)
    """Replace NaN elements in an array using an iterative image inpainting algorithm.
    The algorithm is the following:
    1) For each element in the input array, replace it by a weighted average
    of the neighbouring elements which are not NaN themselves. The weights depends
    of the method type. If ``method=localmean`` weight are equal to 1/( (2*kernel_size+1)**2 -1 )
    2) Several iterations are needed if there are adjacent NaN elements.
    If this is the case, information is "spread" from the edges of the missing
    regions iteratively, until the variation is below a certain threshold.
    Parameters
    ----------
    array : 2d np.ndarray
    an array containing NaN elements that have to be replaced
    max_iter : int
    the number of iterations
    kernel_size : int
    the size of the kernel, default is 1
    method : str
    the method used to replace invalid values. Valid options are
    `localmean`, 'idw'.
    Returns
    -------
    filled : 2d np.ndarray
    a copy of the input array, where NaN elements have been replaced.
    """
    kernel_size = kernel_radius*2+1
    filled = np.empty( [array.shape[0], array.shape[1]])
    kernel = np.empty( (2*kernel_size+1, 2*kernel_size+1))

    # indices where array is NaN
    inans, jnans = np.nonzero( np.isnan(array) )

    # number of NaN elements
    n_nans = len(inans)

    # arrays which contain replaced values to check for convergence
    replaced_new = np.zeros( n_nans)
    replaced_old = np.zeros( n_nans)

    # depending on kernel type, fill kernel array
    if method == 'localmean':

        print('kernel_size', kernel_size)
        for i in range(kernel_size):
            for j in range(kernel_size):
                kernel[i,j] = 1
        print(kernel, 'kernel')

    elif method == 'idw':
        kernel = makeGaussian(kernel_size, kernel_sigma)
        print(kernel.shape, 'kernel')
    else:
        raise ValueError( 'method not valid.')

    # fill new array with input elements
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            filled[i,j] = array[i,j]

    # make several passes
    # until we reach convergence
    for it in range(max_iter):
        # for each NaN element
        for k in range(n_nans):
            i = inans[k]
            j = jnans[k]

            # initialize to zero
            filled[i,j] = 0.0
            n = 0

            # loop over the kernel
            for I in range(kernel_size):
                for J in range(kernel_size):

                    # if we are not out of the boundaries
                    if i+I-kernel_radius < array.shape[0] and i+I-kernel_radius >= 0:
                        if j+J-kernel_radius < array.shape[1] and j+J-kernel_radius >= 0:

                            # if the neighbour element is not NaN itself.
                            if filled[i+I-kernel_radius, j+J-kernel_radius] == filled[i+I-kernel_radius, j+J-kernel_radius]:

                                # do not sum itself
                                if I-kernel_radius != 0 and J-kernel_radius != 0:

                                    # convolve kernel with original array
                                    filled[i,j] = filled[i,j] + filled[i+I-kernel_radius, j+J-kernel_radius]*kernel[I, J]
                                    n = n + 1*kernel[I,J]

            # divide value by effective number of added elements
            if n != 0:
                filled[i,j] = filled[i,j] / n
                replaced_new[k] = filled[i,j]
            else:
                filled[i,j] = np.nan

        # check if mean square difference between values of replaced
        #elements is below a certain tolerance
        print('tolerance', np.mean( (replaced_new-replaced_old)**2 ))
        if np.mean( (replaced_new-replaced_old)**2 ) < tol:
            break
        else:
            for l in range(n_nans):
                replaced_old[l] = replaced_new[l]
    return filled

def loadply(file):
    plydata = PlyData.read(file)
    x = np.array(plydata['vertex']['x'])
    y = np.array(plydata['vertex']['y'])
    z = np.array(plydata['vertex']['z'])
    # Downsample
    c = np.random.choice(len(x),int(len(x)/10))
    x = x[c]
    y = y[c]
    z = z[c]
    
    # Interpolation limits
    tx = np.linspace(x.min(),x.max(),int(np.sqrt(len(x))))
    ty = np.linspace(y.min(),y.max(),int(np.sqrt(len(y))))
    XI,YI = np.meshgrid(tx,ty)
    
    # Grid Data
    ZI = griddata(np.array([x.ravel(),y.ravel()]).T,z.ravel(),(XI,YI),method='nearest',rescale=True)

    return YI,XI,ZI
