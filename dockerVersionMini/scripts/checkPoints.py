# -*- coding: utf-8 -*-
"""
Created on Sat Dec 03 17:06:23 2016

@author: oacom
"""

import numpy as np
import sys

def computeVisibility(terrain, cameras, cfg):
    '''
    This function checks to see which points in an array are visible from a given
    camera location and orientation.  Preps data for checkPoints.
    '''
    if (cfg['angle_reduction']['enable']):
        # Adjust FOV to increase overlap
        ratio = cfg['angle_reduction']['visibility']
        cameras['alphax_adj'] = np.radians(cameras.alphax * ratio)
        cameras['alphay_adj'] = np.radians(cameras.alphay * ratio)
    else:
        # If no reduction, just change to radians
        cameras['alphax_adj']  = np.radians(cameras.alphax)
        cameras['alphay_adj'] = np.radians(cameras.alphay)
    # Build a matrix of the needed information for the visiblity check
    cams_cors = cameras[['cn','ce','cd','cor_n','cor_e','cor_d','cup_n','cup_e','cup_d','alphax_adj','alphay_adj']].values
    # List of terrain points
    P = np.c_[terrain.nn.flatten(), terrain.ee.flatten(), terrain.dd.flatten()].T
    # Load camera near and far planes
    f = cfg['cam_range']['far'] * max(cameras.cd)
    n = cfg['cam_range']['near']
    # Perform the visiblity check
    # visibility - rows are cameras, columns are terrain points
    visibility = np.apply_along_axis(checkPointsOverlap, 1, cams_cors, P=P, f=f, n=n, cfg=cfg)
    pointSpread = visibility[:, -1]
    visibility = visibility[:, :-1]
    return visibility, pointSpread

def checkPoints(CN,P,f,n,cfg):
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
    
    points = P_orig[visiblePoints,:]
    camera = C
    if(cfg['terrain']['source']=='PLY'):
        hpr_param = 3
    else:
        hpr_param = 1
    if(len(points)>4):
        visibleIndex = np.where(visiblePoints==True)[0]
        nonHidden = hpr(points,camera,hpr_param)
#        hidden = np.logical_not(nonHidden)
        visibleHidden = visibleIndex[nonHidden]
        visiblePoints[:]=False
        visiblePoints[visibleHidden]=True

    return visiblePoints

def checkPointsOverlap(CN,P,f,n,cfg):
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
    
    points = P_orig[visiblePoints,:]
    camera = C
    if(cfg['terrain']['source']=='PLY'):
        hpr_param = 3
    else:
        hpr_param = 1
    if(len(points)>4):
        visibleIndex = np.where(visiblePoints==True)[0]
        nonHidden = hpr(points,camera,hpr_param)
#        hidden = np.logical_not(nonHidden)
        visibleHidden = visibleIndex[nonHidden]
        visiblePoints[:]=False
        visiblePoints[visibleHidden]=True
        
    # Gaussian distribution around center point to promote overlap
    visiblePointsFloat = visiblePoints.astype(float)
    ind = np.where(visiblePoints==True)
    if(visiblePoints[visiblePoints==1].size>0):
        for i in ind:
            visiblePointsFloat[i] = np.exp(-(P[i,0]**2/(2*0.4**2)+P[i,1]**2/(2*0.4**2)))
            
    # Check spatial distribution of visible points in the image.  1 - Strong distribution
    # 0 - Weak distribution
    spread = 0
    div = 5
    for i in range(div):
        for j in range(div):
            testH = (P[:,0]>=(i*2/div-1))*(P[:,0]<((i+1)*2/div-1))
            testV = (P[:,1]>=(j*2/div-1))*(P[:,1]<((j+1)*2/div-1))
            spread = spread + bool(np.count_nonzero(testH*testV))
    spread = spread/9.0
            
    visiblePointsFloat = np.append(visiblePointsFloat,spread)
    
    if(len(visiblePointsFloat)==0):
        print('No Images Selected')
        sys.exit(1)

    return visiblePointsFloat

def hpr(p,c,param):
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
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Camera and Normal
    C = np.array([0,0,1])
    N = np.array([1.0,0.0,0.0])
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_xlim(-3,3)
    ax.set_ylim(-3,3)
    ax.set_zlim(-1,3)
    
    plt.quiver(C[0],C[1],C[2],N[0],N[1],N[2],color='r',pivot='tail',length=1)
    
    # Point to look at
    P = np.array([2.5,0,1])
    
    ax.scatter(P[0],P[1],P[2])
    
    A = P
    w = -N
    y = np.array([0,0,1])
    u = np.cross(y,w)/np.linalg.norm(np.cross(y,w))
    v = np.cross(w,u)
    t = np.array([0,0,0]) - C
    
    # Transform to camera space
    E = np.array([[u[0],v[0],w[0],0],
                  [u[1],v[1],w[1],0],
                  [u[2],v[2],w[2],0],
                  [np.dot(u.T,t),np.dot(v.T,t),np.dot(w.T,t),1]])
    P = np.r_[P,1]
    
    Pt1 = np.dot(P,E)
    
    # Horizontal view angle
    alphax = np.radians(30)
    # Vertical view angle
    alphay = np.radians(30)
    # Far plane
    f = 2
    # Near plane
    n = 1
    # Transform to image volume space
    A_vt = np.array([[1/np.tan(alphax),0,0,0],
                  [0,1/np.tan(alphay),0,0],
                  [0,0,(f+n)/float((f-n)),-1],
                  [0,0,2.0*f*n/float((f-n)),0]])
    Pt2 = np.dot(Pt1,A_vt)
    Pt2n = Pt2/Pt2[-1]
    print(Pt2n)
    if((Pt2n >= -1).all() and (Pt2n <= 1).all()):
        print('Point Spotted')
    else:
        print('Nothing to See')
        
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
    ax.scatter(VPt2[:,0],VPt2[:,1],VPt2[:,2])
    
    # Multiple points
    # Point to look at
    P2 = np.array([[2,1],[0,0],[1,0.5]])
    
    ax.scatter(P2[0],P2[1],P2[2])
    
    w = -N
    y = np.array([0,0,1])
    u = np.cross(y,w)/np.linalg.norm(np.cross(y,w))
    v = np.cross(w,u)
    t = np.array([0,0,0]) - C
    
    # Transform to camera space
    E = np.array([[u[0],v[0],w[0],0],
                  [u[1],v[1],w[1],0],
                  [u[2],v[2],w[2],0],
                  [np.dot(u.T,t),np.dot(v.T,t),np.dot(w.T,t),1]])
    P2 = np.r_[P2,np.ones([1, len(P2[0])])]
    
    P2t1 = np.dot(P2.T,E)
    
    # Horizontal view angle
    alphax = np.radians(30)
    # Vertical view angle
    alphay = np.radians(30)
    # Far plane
    f = 2
    # Near plane
    n = 1
    # Transform to image volume space
    A_vt = np.array([[1/np.tan(alphax),0,0,0],
                  [0,1/np.tan(alphay),0,0],
                  [0,0,(f+n)/float((f-n)),-1],
                  [0,0,2*f*n/float((f-n)),0]])
    P2t2 = np.dot(P2t1,A_vt)
    end = P2t2[:,-1]
    P2t2n = P2t2/end[:,None]
    print(P2t2n)
    visiblePoints = np.all((P2t2n >= -1)*(P2t2n <= 1),axis=1)
    
    # Hidden point removal
#    points = P2t2n[visiblePoints,:]
#    camera = np.array([[0,0,5,1]]) # Approximate the camera location
#    if(len(points)>3):
#        visiblePoints = hpr(points,camera,1)