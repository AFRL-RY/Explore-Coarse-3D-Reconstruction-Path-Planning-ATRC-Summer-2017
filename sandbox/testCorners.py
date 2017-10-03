# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 10:49:53 2017

@author: student
"""

# Plot Corners

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

row = metaData.iloc[0,:]

# Get four corners
tl = row[['image corners tl_latitude','tl_longitude']]
tr = row[['tr_latitude','tr_longitude']]
bl = row[['bl_latitude','bl_longitude']]
br = row[['br_latitude','br_longitude']]

# Convert four corners to UTM
tle,tln,z,zl = utmConversion(tl)
tre,trn,z,zl = utmConversion(tr)
ble,bln,z,zl = utmConversion(bl)
bre,brn,z,zl = utmConversion(br)

cam = np.r_[cn[0],ce[0],cd[0]]
nn_cropped,ee_cropped,dd_cropped = cropTerrain(nn,ee,dd,cn,ce,0)
ground_z = np.mean(dd_cropped)
center = np.r_[np.mean([tln,trn,bln,brn]),np.mean([tle,tre,ble,bre]),ground_z]
side = np.r_[np.mean([trn,brn]),np.mean([tre,bre]),ground_z]

cor_vector = center - cam
cor_vector = cor_vector/np.linalg.norm(cor_vector)

cside_vector = side - cam
cside_vector = cside_vector/np.linalg.norm(cside_vector)

cup_vector = np.cross(cor_vector,cside_vector)
if(cup_vector[2]<0): # Make sure we get the up vector and not the down vector
    cup_vector = np.cross(cside_vector,cor_vector)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')

ax.scatter(cam[0],cam[1],cam[2])
ax.scatter([tln,trn,bln,brn],[tle,tre,ble,bre],ground_z)
ax.scatter(center[0],center[1],center[2])
ax.scatter(side[0],side[1],side[2])
ax.quiver(cam[0],cam[1],cam[2],cor_vector[0],cor_vector[1],cor_vector[2],color='g',pivot='tail',length=1500)
ax.quiver(cam[0],cam[1],cam[2],cside_vector[0],cside_vector[1],cside_vector[2],color='b',pivot='tail',length=1500)
ax.quiver(cam[0],cam[1],cam[2],cup_vector[0],cup_vector[1],cup_vector[2],color='r',pivot='tail',length=1500)
