# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 23:15:48 2016

@author: oacom
"""
import numpy as np
from terrain import genTerrain, interpTerrain, calcNormals, loadTerrain
from initCameras import generateCameras
from plotting import plotAll
from checkPoints import checkPoints
from sortAngles import sortAngles
from greedySort import greedySort
import time
from loadCameras import loadCameras

start = time.clock()

np.random.seed(0)

# Create random terrain
nn,ee,dd = genTerrain(100,100,50,5)
# OR Load terrain from file
#nn,ee,dd = loadTerrain('terrain/output.mean.tif')

# Smooth terrain
nn,ee,dd = interpTerrain(nn,ee,dd,2)

# Terrain normals
norm_n,norm_e,norm_d = calcNormals(nn,ee,dd)

# Initialize cameras from normals
cn,ce,cd,cor_n,cor_e,cor_d = generateCameras(nn, ee, dd, norm_n, norm_e, norm_d, 50)
# OR Load cameras from image metadata
#directory_images = 'E:/21_Sept_2015_WAMI_Flight_1/2015.09.21_15.39.54'
#directory_metadata = 'E:/Bags/20150921_Flight1'
#cn,ce,cd,cor_n,cor_e,cor_d = loadCameras(directory_images,directory_metadata)

# Compute camera point visibility matrix
cams_cors = np.c_[cn,ce,cd,cor_n,cor_e,cor_d]
P = np.c_[nn.flatten(),ee.flatten(),dd.flatten()].T
#start_v = time.clock()
# rows are cameras cols are points
visibility = np.apply_along_axis(checkPoints,1,cams_cors,P=P,f=200,n=5,alphax=np.radians(30),alphay=np.radians(30))
#print('Visibility Time: ' + str(time.clock()-start_v))
#fig = plt.figure()
#ax = plt.gca()
#ax.imshow(visibility,cmap='Greys',interpolation='nearest')
# What does this covariance/correspondence plot mean?

# Sort visible points by angle
vis_cors = np.c_[visibility,cor_n,cor_e,cor_d] # Combine all camera info needed
visibilityAngle = np.apply_along_axis(sortAngles,1,vis_cors,norm_n=norm_n,norm_e=norm_e,norm_d=norm_d)
#
# Camera selection
c_min, numCams, camIdx = greedySort(visibilityAngle,0.90)

# Plotting
time_noplot = time.clock()-start
visStack = np.sum(np.r_[c_min[:,0:100],c_min[:,100:200],c_min[:,200:300]],axis=0)
plotAll(1,1,1,1,1,nn,ee,dd,cn,ce,cd,norm_n,norm_e,norm_d,cor_n,cor_e,cor_d,camIdx,visStack)

# Ouput
print('Terrain Points: ' + str(nn.size))
print('Starting Cameras: ' + str(cn.size))
print('Ending Cameras: ' + str(numCams))
print('Terrain Points covered by 3 Cameras: ' + str(np.sum(visStack>=3)))
print('Terrain Points covered by 2 Cameras: ' + str(np.sum(visStack==2)))
print('Terrain Points covered by 1 Cameras: ' + str(np.sum(visStack==1)))
print('Terrain Points covered by 0 Cameras: ' + str(np.sum(visStack==0)))
print('Solution Time: ' + str(time_noplot))
print('Solution Time with plotting: ' + str(time.clock()-start))