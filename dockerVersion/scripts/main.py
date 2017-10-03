# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 23:15:48 2016

@author: oacom
"""
import numpy as np
import time
from terrain import calcNormals, loadTerrain, cropTerrain, addDome
from plotting import plotResults
from checkPoints import computeVisibility
from sortAngles import sortAnglesOverlap
from greedySort import greedySort5
from loadCameras import loadCameras
from output import moveImages, writeFrameList, writePairList, geoTagImages, startLogging, shrinkImages
import yaml
import logging
import datetime
import shutil
from initCameras import generateCameras
from findPath import findPath
from analyze import analyzeResults
import sys

def run_main(cfg_file_path,directory_terrain=None):
    
    start = time.clock()
    
    # Load configuration file
#    cfg_file_path = './configFiles/configMuscat.yml'
    with open(cfg_file_path,'r') as ymlfile:
        cfg = yaml.load(ymlfile)
        
    # Start logging
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    directory_images = cfg['dirs']['directory_images']
    output_directory = startLogging(directory_images,timestamp)
    
    # Load terrain from file
    start_sub = time.clock()
    print('Loading terrain data...')
    if(directory_terrain == None):
        directory_terrain = cfg['dirs']['directory_terrain']
    terrain = loadTerrain(directory_terrain,cfg)
    print('Done...Time: ' + str(time.clock()-start_sub))
    logging.info('Terrain: '+ str(time.clock()-start_sub))
    
    # Load cameras from image metadata
    if(cfg['run']['mode']=='SelectImages'):
        start_sub = time.clock()
        print('Loading image locations and poses...')
        directory_images = cfg['dirs']['directory_images']
        directory_metadata = cfg['dirs']['directory_metadata']
        metaDataType = cfg['dirs']['metadata_type']
        cameras,metaData = loadCameras(directory_images,directory_metadata,metaDataType,terrain)
        print('Done...Time: ' + str(time.clock()-start_sub))
        logging.info('Images: ' + str(time.clock()-start_sub))
    else:
        # Assign dummy values to cams until they are generated later.
        cameras = []
    
    # Crop Terrain to chosen area
    start_sub = time.clock()
    print('Cropping Terrain...')
    terrain = cropTerrain(terrain, cameras, cfg)
    print('Done...Time: ' + str(time.clock()-start_sub))
    logging.info('Crop: ' + str(time.clock()-start_sub))
    
    # Add dome to target area
    if(cfg['dome']['enable']):
        terrain = addDome(terrain,cfg['dome']['height'])
        # TODO: Explore different/multiple domes
    
    # Terrain normals
    start_sub = time.clock()
    print('Calculating Normals...')
    normals = calcNormals(terrain)
    print('Done...Time: ' + str(time.clock()-start_sub))
    logging.info('Normals: ' + str(time.clock()-start_sub))
    
    # Create Cameras projected from surface
    if(cfg['run']['mode']=='PlanFlight'):
        cameras = generateCameras(terrain, normals, cfg)
    
    # Compute camera point visibility matrix
    start_sub = time.clock()
    print('Computing visibility matrix...')
    visibility, pointSpread = computeVisibility(terrain, cameras, cfg)
    # TODO: Threading/parallel processing for this part.  
    print('Done...Time: ' + str(time.clock()-start_sub))
    logging.info('Visibility: ' + str(time.clock()-start_sub))
    
    # Sort visible points by angle
    start_sub = time.clock()
    print('Sorting points by angle...')
    vis_cors = np.c_[visibility,cameras[['cn','ce','cd','cor_n','cor_e','cor_d']].values] # Combine all camera info needed
    visibilityAngle = np.apply_along_axis(sortAnglesOverlap,1,vis_cors,normals=normals,terrain=terrain,cfg=cfg)
    print('Done...Time: ' + str(time.clock()-start_sub))
    logging.info('Angles: ' + str(time.clock()-start_sub))
    
    # Camera selection
    start_sub = time.clock()
    print('Selecting images...')
    c_min, numCams, camIdx = greedySort5(visibilityAngle.copy(),cfg['selection']['tol'],pointSpread,cfg)
    c_min = c_min.astype(bool)
    print('Done...Time: ' + str(time.clock()-start_sub))
    logging.info('Selection: ' + str(time.clock()-start_sub))
    
    # Find Shortest Path
    if(cfg['run']['mode']=='PlanFlight'):
        print('Finding Shortest Path...')
        start_sub = time.clock()
        flightPath, order = findPath(camIdx,cameras,cfg)
        print('Done...Time: ' + str(time.clock()-start_sub))
        logging.info('Shortest Path: ' + str(time.clock()-start_sub))
    else:
        flightPath = [] # Dummy list for plotting
        
    # Analyze Results
    pointsVisibleInitial, pointsVisibleFinal, pointsNotVisibleInitial = analyzeResults(visibility, c_min, terrain.dd, cfg)    
    
    # Plotting
    time_noplot = time.clock()-start # Get time before plotting
    print('Plotting...')
    start_sub = time.clock()
    if(cfg['plotting']['enable']==True):
        plotResults(terrain,cameras,normals,camIdx,pointsVisibleInitial,pointsVisibleFinal,flightPath,cfg)
    print('Done...Time: ' + str(time.clock()-start_sub))
    logging.info('Plotting: ' + str(time.clock()-start_sub))
    
    # Ouput camera lists and copy images
    if(cfg['output']['enable'] and cfg['run']['mode']=='SelectImages'):
        df_output_list = writeFrameList(directory_images,metaData,camIdx,output_directory)
        moveImages(directory_images,df_output_list,output_directory)
        geoTagImages(directory_images,metaData,cameras,camIdx,output_directory)
        writePairList(directory_images,c_min,df_output_list,cfg['output']['pair_list_matches'],output_directory)
        shutil.copy2(cfg_file_path,output_directory)   
        shrinkImages(directory_images,df_output_list,output_directory)
    
    # Print Results
    print('All Complete!')
    print('Terrain Points: ' + str(terrain.nn.size))
    logging.info('Terrain Points: ' + str(terrain.nn.size))
    print('Starting Cameras: ' + str(cameras.cn.size))
    logging.info('Starting Cameras: ' + str(cameras.cn.size))
    print('Ending Cameras: ' + str(numCams))
    logging.info('Ending Cameras: ' + str(numCams))
    if(cfg['plotting']['enable']):
        print('Starting Terrain Points Visible: ' +str(terrain.nn.size-len(pointsNotVisibleInitial)))
        print('Ending Terrain Points Visible: ' + str(np.sum(pointsVisibleFinal > 0)))
    print('Solution Time: ' + str(time_noplot))
    logging.info('Solution Time: ' + str(time_noplot))
    if(cfg['plotting']['enable']):
        print('Solution Time with plotting: ' + str(time.clock()-start))
    
    return output_directory    
    
if __name__ == "__main__":
    run_main(sys.argv[1])