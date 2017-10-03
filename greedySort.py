# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 14:22:26 2016

@author: oacom
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')
from scipy.optimize import fsolve
from scipy.spatial.distance import cdist

def greedySort(visibility,tol):
    '''
    Given a binary matrix with rows representing cameras and columns representing
    terrain points, this function greedily picks cameras that can see the most points
    until full or maximum coverage is achieved.
    INPUT
    visibility: nxm logical matrix where n is cameras and m is points
    tol: tolerance value (0-1) of acceptable percentage of points to cover
    OUTPUT
    visibility: minimum camera set
    numCams: number of cameras in minimum set
    idx: indexes of minimum set cameras in original camera set.
    '''
    # Sort visibility matrix by cameras that see most points (ascending)
    colSum = np.sum(visibility,axis=1)
    idx = colSum.argsort()
    visibility = np.take(visibility,idx,axis=0)
    
    # Determine the maximum number of points that can be covered with these cameras
    maxCover = np.count_nonzero(np.sum(visibility,axis=0))
    
    # Greedily select cameras until max coverage is reached
    cover = 0
    numCams = 0
    while(cover < maxCover*tol):
        numCams = numCams + 1
        cover = np.count_nonzero(np.sum(visibility[-numCams:,:],axis=0))
    # Find number of cameras that meets coverage requirement
#    numCams = fsolve(coverageZero,len(visibility)*0.75,args=(visibility,maxCover,tol))
#    numCams = int(numCams)
    
    return visibility[-numCams:,:], numCams, idx[-numCams:]

def greedySort2(visibility,tol):
    '''
    Given a binary matrix with rows representing cameras and columns representing
    terrain points, this function greedily picks cameras that can see the most points
    until full or maximum coverage is achieved.
    INPUT
    visibility: nxm logical matrix where n is cameras and m is points
    tol: tolerance value (0-1) of acceptable percentage of points to cover
    OUTPUT
    visibility: minimum camera set
    numCams: number of cameras in minimum set
    idx: indexes of minimum set cameras in original camera set.
    '''
    # Remove terrain points not visible in original image set
    filtered = visibility[:,np.where(np.sum(visibility,axis=0)>0)[0]]
    
    # Sort visibility matrix by cameras that see most points (ascending)
    colSum = np.sum(filtered,axis=1)
    idx = colSum.argsort()
    filtered = np.take(filtered,idx,axis=0)
    
    # Greedily select cameras until desired coverage is reached
    numCams = 0
    # Select until all points are covered
    while(filtered.shape[1]>0):
#        print(numCams)
#        print(filtered.shape[1])
        # Pick the camera at the bottom of the visibility matrix (can see most points)
        numCams = numCams + 1
        # Remove terrain point columns that are already covered from matrix
        filtered = filtered[:,np.sum(filtered[-numCams:],axis=0)==0]
        # Re-sort the visibility matrix for non chosen cameras
        colSum = np.sum(filtered[:-numCams,:],axis=1)
        # Re-sort camera indices to match
        idx = np.r_[idx[colSum.argsort()],idx[-numCams:]]
        # Apply sort to visiblity matrix
        filtered = np.take(filtered,idx,axis=0)
        
#    fig = plt.figure()
#    ax = plt.gca()
#    ax.imshow(filtered,cmap='Greys',interpolation='nearest')
    
    return visibility[idx[-numCams:],:], numCams, idx[-numCams:]

def greedySort3(visibility,tol):
    '''
    Given a binary matrix with rows representing cameras and columns representing
    terrain points, this function greedily picks cameras that can see the most points
    until full or maximum coverage is achieved.
    INPUT
    visibility: nxm logical matrix where n is cameras and m is points
    tol: tolerance value (0-1) of acceptable percentage of points to cover
    OUTPUT
    visibility: minimum camera set
    numCams: number of cameras in minimum set
    idx: indexes of minimum set cameras in original camera set.
    '''
    # Save copy of original matrix for the end
    visibilityOriginal = visibility.copy()
    
    # Max points covered by the initial image set
    maxCover = np.sum(np.sum(visibility,axis=0)>0)

    # Initialize
    covered = 0
    idx = []
    # Loop until desired number of points are covered by the image set
    while(maxCover*tol - covered > 0):
        colSum = np.sum(visibility,axis=1) # Sum visible points from each camera
        mostPoints = np.argmax(colSum) # Get index of camera that sees most points
        idx.append(mostPoints) # Add camera index to list of chosen cameras
        pointsCovered = visibility[mostPoints,:]==1 # Get the indices of points covered by that camera
        covered = covered + np.sum(pointsCovered) # Update total number of points covered
        visibility[:,pointsCovered]=0 # Discard points already covered
    
    # Get final number of images
    numCams = len(idx)
    
    return visibilityOriginal[idx,:], numCams, idx

def greedySort4(visibility,tol):
    '''
    Given a binary matrix with rows representing cameras and columns representing
    terrain points, this function greedily picks cameras that can see the most points
    until full or maximum coverage is achieved.
    This one adds an overlap constraint between images
    INPUT
    visibility: nxm logical matrix where n is cameras and m is points
    tol: tolerance value (0-1) of acceptable percentage of points to cover
    OUTPUT
    visibility: minimum camera set
    numCams: number of cameras in minimum set
    idx: indexes of minimum set cameras in original camera set.
    '''
    # Save copy of original matrix for the end
    visibilityOriginal = visibility.copy()
    
    # Max points covered by the initial image set
    maxCover = np.sum(np.sum(visibility,axis=0)>0)

    # Initialize
    covered = 0
    idx = []
    stop = False
    # Loop until desired number of points are covered by the image set
    while(maxCover*tol - covered > 0):
        colSum = np.sum(visibility,axis=1) # Sum visible points from each camera
        i = 1
        done = False
        while(done==False and i<=(len(colSum)-len(idx))):
            mostPoints = np.argsort(colSum)[-i] # Get index of camera that sees most points
            print(i)
            print(mostPoints)
            # Overlap is impossible
            if(i == len(colSum)-len(idx)):
                print('Unable to find overlapping image!')
                stop = True
                break
            # Overlap Check
            if(len(idx)>0):
                xa = visibilityOriginal[mostPoints,:]
                xb = visibilityOriginal[idx,:]
                dist = cdist([xa],xb, lambda u, v: np.count_nonzero((u==True)&(v==True)))
                if(dist.max()>=colSum[mostPoints]/3.0):
                    done = True
                else:
                    i = i + 1
            else:
                done = True
        if(stop==True):
            break
        idx.append(mostPoints) # Add camera index to list of chosen cameras
        pointsCovered = visibility[mostPoints,:]==1 # Get the indices of points covered by that camera
        covered = covered + np.sum(pointsCovered) # Update total number of points covered
        visibility[:,pointsCovered]=0 # Discard points already covered
        print(maxCover*tol - covered)
    
    # Get final number of images
    numCams = len(idx)
    
    return visibilityOriginal[idx,:], numCams, idx

def greedySort5(visibility,tol,pointSpread,cfg):
    '''
    Given a binary matrix with rows representing cameras and columns representing
    terrain points, this function greedily picks cameras that can see the most points
    until full or maximum coverage is achieved.
    This one uses the gaussian distribution to promote centered images
    INPUT
    visibility: nxm logical matrix where n is cameras and m is points
    tol: tolerance value (0-1) of acceptable percentage of points to cover
    OUTPUT
    visibility: minimum camera set
    numCams: number of cameras in minimum set
    idx: indexes of minimum set cameras in original camera set.
    '''
    # Save copy of original matrix for the end
    visibilityOriginal = visibility.copy()
    
    overlap_ratio = cfg['overlap_enhancement']['ratio']
    
    # Max points covered by the initial image set
    maxCover = np.sum(np.sum(visibility,axis=0)>overlap_ratio)

    # Initialize
    covered = 0
    idx = []
    # Loop until desired number of points are covered by the image set
    while(maxCover*tol - covered > 0):
        colSum = np.sum(visibility,axis=1) # Sum visible points from each camera
        colSum = colSum*pointSpread # Promote images with better spatial distribution of points
        mostPoints = np.argmax(colSum) # Get index of camera that sees most points
        idx.append(mostPoints) # Add camera index to list of chosen cameras
        pointsCovered = np.sum(visibility[idx,:],axis=0)>overlap_ratio # Get the indices of points covered by that camera
        covered = covered + np.sum(pointsCovered) # Update total number of points covered
        visibility[:,pointsCovered]=0 # Discard points already covered
#        print(covered)
    
    # Get final number of images
    idx = list(set(idx))
    numCams = len(idx)
    
    return visibilityOriginal[idx,:], numCams, idx

def coverageZero(numCams,visibility,maxCover,tol):
    '''
    Helper function for coverage root finding
    Finds number of cameras where coverage is equal to the maximum possible coverage
    plus a tolerance.
    '''
    cover = np.count_nonzero(np.sum(visibility[-int(numCams):,:],axis=0))
    zero = cover - maxCover*tol
    return zero

#    # Sort visibility matrix by cameras that see most points (ascending)
#    colSum = np.sum(visibility,axis=1)
#    idxRow = colSum.argsort()
##    visibility = np.take(visibility,idxRow,axis=0)
##    rowSum = np.sum(visibility,axis=0)
##    idxCol = rowSum.argsort()
##    visibility = np.take(visibility,idx,axis=1)
##    import matplotlib.pyplot as plt
##    fig = plt.figure()
##    ax = plt.gca()
##    ax.imshow(visibility,cmap='Greys',interpolation='nearest')
#    
#    # Determine the maximum number of points that can be covered with these cameras
#    maxCover = np.count_nonzero(np.sum(visibility,axis=0))
#    
#    # Greedily select cameras until max coverage is reached
#    cover = 0
#    idxList = []
#    visibilitySort = visibility.copy()
#    while(cover < maxCover*tol):
#        idxList.append(idxRow[-1])
#        cover = np.count_nonzero(np.sum(visibility[idxList,:],axis=0))
#        print(cover)
#        # Take out points that are already 3 covered and re-sort
#        coveredPoints = np.sum(visibility[idxList,:],axis=0)
#        visibilitySort = visibility[:,coveredPoints<3]
#        colSum = np.sum(visibilitySort,axis=1)
#        idxNew = colSum.argsort()
#        visibilitySort = np.take(visibilitySort,idxNew,axis=0)
#        idxRow = np.take(idxRow,idxNew)
#    
#    return visibility[np.unique(idxList),:], len(np.unique(idxList)), np.unique(idxList)