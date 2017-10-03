# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:15:59 2017

@author: student
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from checkPoints import checkPointsOverlap, checkPoints
from sortAngles import sortAnglesOverlap, recordAngles
from numpy.linalg import norm
from terrain import calcNormals
from greedySort import greedySort3, greedySort5
from plotting import plotFourCorners

m = 10
angles = np.linspace(0,2*np.pi,m)
pt = np.array([np.cos(angles),np.sin(angles)])*10

fig = plt.figure()
plt.scatter(pt[0,:],pt[1,:])

cn = pt[0,:]
ce = pt[1,:]
cd = np.ones(cn.shape)*5

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')

nn = np.array([0,0.1])
ee = np.array([0,0.1])
dd = np.array([0,0.1])

ax.scatter(cn,ce,cd)
ax.scatter(nn,ee,dd)

cor_n = (nn-cn)*np.random.uniform(0.5,1.5,10)
cor_e = (ee-ce)*np.random.uniform(0.5,1.5,10)
cor_d = (dd-cd)*np.random.uniform(0.5,1.5,10)

cor = np.c_[cor_n,cor_e,cor_d]
cor = cor/norm(cor,axis=1).reshape(m,1)

cor_n = cor[:,0]
cor_e = cor[:,1]
cor_d = cor[:,2]

cup_n = np.zeros(cn.shape)
cup_e = np.zeros(cn.shape)
cup_d = np.ones(cn.shape)

ax.quiver(cn,ce,cd,cor_n,cor_e,cor_d)
ax.quiver(cn,ce,cd,cup_n,cup_e,cup_d)

alphax_adj = np.ones(cn.shape)*np.deg2rad(15)
alphay_adj = np.ones(cn.shape)*np.deg2rad(15)

cams_cors = np.c_[cn,ce,cd,cor_n,cor_e,cor_d,cup_n,cup_e,cup_d,alphax_adj,alphay_adj]
for i in range(len(cams_cors)):
    plotFourCorners(cams_cors[i,:],f=15,n=1,ax=ax)
# List of terrain points
P = np.c_[nn,ee,dd].T
visibility = np.apply_along_axis(checkPointsOverlap,1,cams_cors,P=P,f=15,n=1)


norm_n = np.array([0])
norm_e = np.array([0])
norm_d = np.array([1])

vis_cors = np.c_[visibility,cn,ce,cd,cor_n,cor_e,cor_d] # Combine all camera info needed
visibilityAngle = np.apply_along_axis(sortAnglesOverlap,1,vis_cors,norm_n=norm_n,norm_e=norm_e,norm_d=norm_d,nn=nn,ee=ee,dd=dd)
#angleMatrix = np.apply_along_axis(recordAngles,1,vis_cors,norm_n=norm_n,norm_e=norm_e,norm_d=norm_d,nn=nn,ee=ee,dd=dd)
#
#c_min1, numCams1, camIdx1 = greedySort3(visibilityAngle,1)

c_min2, numCams2, camIdx2 = greedySort5(visibilityAngle,1)

def plotCheckPoints(CN,P,f,n,ax):
    '''
    This function checks to see which points in an array are visible from a given
    camera location and orientation
    INPUTS
    CN: Camera location (3x1) and Camera orientation vector (3x1)
    P: Array of N points (3xN)
    f: Camera far plane (meters)
    n: Camera near plane (meters)
    alphax: Camera horizontal angle of view
    alphay: Camera vertical angle of view
    OUTPUT
    visibility: logical vector of length m where m is number of terrain points
    '''
    # Camera location
    C = CN[0:3]
    
    # Camera orientation
    N = CN[3:6]
    
    # Camera up vector
    U = CN[6:9]
    
    # Camera field of view
    alphax = CN[9]
    alphay = CN[10]
    
    # Copy of points for later
    P_orig = P.copy().T
    
    # Define camera space axis    
    w = -N
#    y = np.array([0,0,1])
    y = U
    u = np.cross(y,w)/np.linalg.norm(np.cross(y,w))
    v = np.cross(w,u)
    t = np.array([0,0,0]) - C
    
    # Define transform from cartesian space to camera space
    E = np.array([[u[0],v[0],w[0],0],
                  [u[1],v[1],w[1],0],
                  [u[2],v[2],w[2],0],
                  [np.dot(u.T,t),np.dot(v.T,t),np.dot(w.T,t),1]])
    # Add fourth dimension of ones for homogeneous coordinate system
    P = np.r_[P,np.ones([1, P.shape[1]])]
    
    # Transform points to camera space
    P = np.dot(P.T,E)
    
    # Define transform from camera space to image volume space
    A_vt = np.array([[1/np.tan(alphax),0,0,0],
                  [0,1/np.tan(alphay),0,0],
                  [0,0,(f+n)/float((f-n)),-1],
                  [0,0,2*f*n/float((f-n)),0]])
    
    # Transform points to image volume space
    P = np.dot(P,A_vt)
    
    # Normalize values by last column
    end = P[:,-1]
    P = P/end[:,None]
    
    # Visible points have all values within range -1 to 1
    visiblePoints = np.all((P >= -1)*(P <= 1),axis=1)
    
    VP = np.array([[1,1,1,1],
                   [1,1,-1,1],
                   [1,-1,1,1],
                   [1,-1,-1,1],
                   [-1,1,1,1],
                   [-1,1,-1,1],
                   [-1,-1,1,1],
                   [-1,-1,-1,1]])
    
    VPt = np.dot(np.dot(VP,np.linalg.inv(A_vt)),np.linalg.inv(E))
    end2 = VPt[:,-1]
    VPt2 = VPt/end2[:,None]
    sc = ax.scatter(VPt2[:,0],VPt2[:,1],VPt2[:,2])
    ax.plot(VPt2[[0,1],0],VPt2[[0,1],1],VPt2[[0,1],2],color=sc.get_edgecolor()[0])
    ax.plot(VPt2[[2,3],0],VPt2[[2,3],1],VPt2[[2,3],2],color=sc.get_edgecolor()[0])
    ax.plot(VPt2[[4,5],0],VPt2[[4,5],1],VPt2[[4,5],2],color=sc.get_edgecolor()[0])
    ax.plot(VPt2[[6,7],0],VPt2[[6,7],1],VPt2[[6,7],2],color=sc.get_edgecolor()[0])
    
#    points = P_orig[visiblePoints,:]
#    camera = C
#    if(len(points)>4):
#        visibleIndex = np.where(visiblePoints==True)[0]
#        nonHidden = hpr(points,camera,1)
##        hidden = np.logical_not(nonHidden)
#        visibleHidden = visibleIndex[nonHidden]
#        visiblePoints[:]=False
#        visiblePoints[visibleHidden]=True

    return visiblePoints