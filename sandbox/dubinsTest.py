# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 11:56:15 2017

@author: student
"""

import dubins
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist,squareform
from TSChristofides import Christofides


qlist = np.zeros([len(cn[camIdx]),3])

for i, row in enumerate(qlist):
    theta = np.arctan2(cor_e[camIdx][i],cor_n[camIdx][i])
    theta = theta + 3*np.pi/2.0
    qlist[i,:] = np.array([cn[camIdx][i],ce[camIdx][i],theta])
    
fig = plt.figure()
plt.scatter(cn[camIdx],ce[camIdx])
plt.quiver(cn[camIdx],ce[camIdx],cor_n[camIdx],cor_e[camIdx])
plt.title('Before')
for i, row in enumerate(qlist[:-1]):
    q0 = qlist[i,:]
    q1 = qlist[i+1,:]
    turning_radius = 50
    step_size = 5
    qs, _ = dubins.path_sample(q0, q1, turning_radius, step_size)
    qs = np.asarray(qs)
    plt.plot(qs[:,0],qs[:,1])

def dubins_dist(a,b):
    turning_radius = 50
    step_size = 5
    qs, _ = dubins.path_sample(a, b, turning_radius, step_size)
    qs = np.asarray(qs)
    dist = 0
    for i in range(len(qs)-1):
        dist += np.sqrt((qs[i,0]-qs[i+1,0])**2+(qs[i,1]-qs[i+1,1])**2)
    return dist

#distances = squareform(pdist(qlist,dubins_dist))
#    
#distances_upper = np.triu(distances)

TSP = Christofides(qlist)

qlist = qlist[TSP.finalSolution]

fig = plt.figure()
plt.scatter(cn[camIdx],ce[camIdx])
plt.quiver(cn[camIdx],ce[camIdx],cor_n[camIdx],cor_e[camIdx])
plt.title('After')
for i, row in enumerate(qlist[:-1]):
    q0 = qlist[i,:]
    q1 = qlist[i+1,:]
    turning_radius = 50
    step_size = 5
    qs, _ = dubins.path_sample(q0, q1, turning_radius, step_size)
    qs = np.asarray(qs)
    plt.plot(qs[:,0],qs[:,1])