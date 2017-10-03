# -*- coding: utf-8 -*-
"""
Created on Mon May 15 10:51:58 2017

% Python port of original code by Sagi Katz
% sagikatz@tx.technion.ac.il
% For more information, see "Direct Visibility of Point Sets", Katz S., Tal
% A. and Basri R., SIGGRAPH 2007, ACM Transactions on Graphics, Volume 26, Issue 3, August 2007.

% HPR - Using HPR ("Hidden Point Removal) method, approximates a visible subset of points 
% as viewed from a given viewpoint.
% Usage:
% visiblePtInds=HPR(p,C,param)
%
% Input:
% p - NxD D dimensional point cloud.
% C - 1xD D dimensional viewpoint.
% param - parameter for the algorithm. Indirectly sets the radius.
% This should be larger for dense clouds
%
% Output:
% visiblePtInds - indices of p that are visible from C.

@author: student
"""

import numpy as np
from scipy.spatial import ConvexHull

def hpr(p,c,param):
    
    dim = np.size(p,1)
    numPts = np.size(p,0)
    p = p - np.tile(c,(numPts, 1)) # Move C to the origin
    normp=np.sqrt(np.sum(p*p, axis=1)) # Calculate ||p||
    normp=np.reshape(normp,(len(normp),1))
#    param = 1
    R=np.tile(max(normp)*(10**param),(numPts, 1)) # Sphere radius
    P=p+2*np.tile(R-normp,(1, dim))*p/np.tile(normp,(1, dim)) # Spherical flipping
    hull=ConvexHull(np.r_[P,np.zeros((1,dim))]) # convex hull
    visiblePtInds=np.unique(hull.vertices)
    visiblePtInds = visiblePtInds[visiblePtInds!=numPts]
    
    return visiblePtInds