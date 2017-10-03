# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 22:56:36 2016

@author: oacom
"""
import numpy as np
import pandas as pd

def generateCameras(terrain, normals, cfg):
    '''
    Calculates positions and orientations of initial population of cameras
    based off of the surface normals
    
    INPUTS
    nn: Terrain north values
    ee: Terrain east values
    dd: Terrain elevation values
    norm_n: North component of surface normals
    norm_e: East component of surface normals
    norm_d: Vertical component of surface normals
    elevation: Desired flight elevation (m)
    OUTPUTS
    Returns camera position vectors cn, ce, cd, and camera orientation vectors
    cor_n, cor_e, and cor_d
    '''
    # Convert 2D grid values to 1D vectors
    nu = np.ravel(terrain.nn)
    eu = np.ravel(terrain.ee)
    du = np.ravel(terrain.dd)
    
    # Unpack normals
    norm_n = normals.norm_n.values
    norm_e = normals.norm_e.values
    norm_d = normals.norm_d.values
    
    # Calculate camera positions
    elevation = cfg['flightPath']['elevation']
    cn = nu + norm_n * (elevation-du)/norm_d
    ce = eu + norm_e * (elevation-du)/norm_d
    cd = du + norm_d * (elevation-du)/norm_d
    
    # Calculate camera orientation vectors (just reflection of surface normal)
    cor_n = -norm_n
    cor_e = -norm_e
    cor_d = -norm_d

    # Up vector for cameras
    cup_n = np.zeros(cn.shape)
    cup_e = np.zeros(cn.shape)
    cup_d = np.ones(cn.shape)

    # Fields of view
    alphax = np.ones(cn.shape) * cfg['flightPath']['alphax']
    alphay = np.ones(cn.shape) * cfg['flightPath']['alphay']
    
    # Only return cameras that are reasonable close
    offset = (elevation-du)/norm_d
    idx = offset <= (elevation-terrain.dd.mean())*3

    # Pack camera information
    # Pack camera information
    cameras = pd.DataFrame({'cn':cn[idx],
                            'ce':ce[idx],
                            'cd':cd[idx],
                            'cor_n':cor_n[idx],
                            'cor_e':cor_e[idx],
                            'cor_d':cor_d[idx],
                            'cup_n':cup_n[idx],
                            'cup_e':cup_e[idx],
                            'cup_d':cup_d[idx],
                            'alphax':alphax[idx],
                            'alphay':alphay[idx]})
    
    return cameras