# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 23:04:37 2016

@author: oacom
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import dubins

def plotResults(terrain,cameras,normals,camIdx,visStatsInitial,visStatsFinal,flightPath,cfg):
    '''
    Master plotting function
    '''
    
    # Unpack terrain
    nn = terrain.nn
    ee = terrain.ee
    dd = terrain.dd

    # Unpack cameras
    cn = cameras.cn
    ce = cameras.ce
    cd = cameras.cd
    cor_n = cameras.cor_n
    cor_e = cameras.cor_e
    cor_d = cameras.cor_d
    cup_n = cameras.cup_n
    cup_e = cameras.cup_e
    cup_d = cameras.cup_d
    
    # Unpack normals
    norm_n = normals.norm_n
    norm_e = normals.norm_e
    norm_d = normals.norm_d

    if (cfg['angle_reduction']['enable']):
        # Adjust FOV to increase overlap
        ratio = cfg['angle_reduction']['visibility']
        alphax = np.radians(cameras.alphax * ratio)
        alphay = np.radians(cameras.alphay * ratio)
    else:
        # If no reduction, just change to radians
        alphax = np.radians(cameras.alphax)
        alphay = np.radians(cameras.alphay)

    if (cfg['plotting']['frustums']):
        # Plot Image frustums
        f = cfg['cam_range']['far'] * max(cd)
        n = cfg['cam_range']['near']
        ax = plotAll(1, 0, 0, 0, 0, 0, terrain, cn, ce, cd, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        cams_cors_fc = np.c_[
            cn[camIdx], ce[camIdx], cd[camIdx], cor_n[camIdx], cor_e[camIdx], cor_d[camIdx], cup_n[camIdx], cup_e[
                camIdx], cup_d[camIdx], alphax[camIdx], alphay[camIdx]]
        for i in range(len(cams_cors_fc)):
            plotFrustums(cams_cors_fc[i, :], f, n, ax)
        ax.set_xlim(min(nn.min(), cn[camIdx].min()) - 2, max(nn.max(), cn[camIdx].max()) + 2)
        ax.set_ylim(min(ee.min(), ce[camIdx].min()) - 2, max(ee.max(), ce[camIdx].max()) + 2)
        set_axes_equal(ax)
    if (cfg['plotting']['initialImages']):
        # Plot Terrain and cameras
        ax = plotAll(1, 0, 1, 0, 0, 0, terrain, cn, ce, cd, norm_n, norm_e, norm_d, cor_n, cor_e, cor_d, camIdx,
                     visStatsFinal, visStatsInitial)
        plt.title('Initial Images')
    if (cfg['plotting']['finalImages']):
        # Plot Terrain and selected cameras
        plotAll(1, 0, 0, 1, 0, 0, terrain, cn, ce, cd, norm_n, norm_e, norm_d, cor_n, cor_e, cor_d, camIdx,
                visStatsFinal, visStatsInitial)
        plt.title('Selected Images')
    if (cfg['plotting']['initialCoverage']):
        # Plot Initial cameras and point visibility
        plotAll(0, 0, 1, 0, 0, 1, terrain, cn, ce, cd, norm_n, norm_e, norm_d, cor_n, cor_e, cor_d, camIdx,
                visStatsFinal, visStatsInitial)
        plt.title('Initial Terrain Coverage')
    if (cfg['plotting']['finalCoverage']):
        # Plot Final Selected cameras and point visibility
        plotAll(0, 0, 0, 1, 1, 0, terrain, cn, ce, cd, norm_n, norm_e, norm_d, cor_n, cor_e, cor_d, camIdx,
                visStatsFinal, visStatsInitial)
        plt.title('Final Terrain Coverage')
    if (cfg['plotting']['flightPath'] and cfg['run']['mode']=='PlanFlight'):
            fig = plt.figure()
            plt.scatter(cn[camIdx],ce[camIdx])
            plt.quiver(cn[camIdx],ce[camIdx],cor_n[camIdx],cor_e[camIdx])
            plt.title('Shortest Flight Path')
            for i, row in enumerate(flightPath[:-1]):
                q0 = flightPath[i,:]
                q1 = flightPath[i+1,:]
                turning_radius = cfg['flightPath']['min_turn']
                step_size = 5
                qs, _ = dubins.path_sample(q0, q1, turning_radius, step_size)
                qs = np.asarray(qs)
                plt.plot(qs[:,0],qs[:,1])
    plt.show()
    return

def plotAll(plotTerrain,plotNormals,plotCameras,plotSelected,plotCoveredFinal,plotCoveredInitial,terrain,cn,ce,cd,norm_n,norm_e,norm_d,cor_n,cor_e,cor_d,camIdx,visStatsFinal,visStatsInitial):
    """
    Plotting function for AVP
    INPUTS
    plotTerrain: Binary switch
    plotNormals: Binary switch
    plotCameras: Binary switch
    plotSelected: Binary switch
    plotCoveredFinal: Binary switch (Final terrain coverage)
    plotCoveredInitial: Binary switch (Initial terrain Coverage)
    """
    # Unpack terrain
    nn = terrain.nn
    ee = terrain.ee
    dd = terrain.dd
    
    # Initialize plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_xlim(min(nn.min(),cn[camIdx].min())-2,max(nn.max(),cn[camIdx].max())+2)
    ax.set_ylim(min(ee.min(),ce[camIdx].min())-2,max(ee.max(),ce[camIdx].max())+2)
#    ax.set_zlim(dd.min()-20,dd.max()+20)
    ax = plt.gca()
    ax.set_xlabel('N (m)')
    ax.set_ylabel('E (m)')
    ax.set_zlabel('Elevation (m)')
    
    # Flatten 2D grids to 1D vectors
    nu = np.ravel(nn)
    eu = np.ravel(ee)
    du = np.ravel(dd)
    
    if(plotTerrain):
        ax.plot_surface(nn,ee,dd,rstride=1, cstride=1, cmap=cm.summer,
                       linewidth=0, antialiased=False)
        ax.set_xlim(nn.min()-2,nn.max()+2)
        ax.set_ylim(ee.min()-2,ee.max()+2)

    if(plotNormals):
        plt.quiver(nu,eu,du,norm_n,norm_e,norm_d,color='r',pivot='tail',length=25)
    
    if(plotCameras):
        plt.quiver(cn,ce,cd,cor_n,cor_e,cor_d,color='g',pivot='tail',length=25)
        ax.set_xlim(min(nn.min(),cn.min())-2,max(nn.max(),cn.max())+2)
        ax.set_ylim(min(ee.min(),ce.min())-2,max(ee.max(),ce.max())+2)
        
    if(plotSelected):
        ax.scatter(cn[camIdx],ce[camIdx],cd[camIdx])
        ax.set_xlim(min(nn.min(),cn[camIdx].min())-2,max(nn.max(),cn[camIdx].max())+2)
        ax.set_ylim(min(ee.min(),ce[camIdx].min())-2,max(ee.max(),ce[camIdx].max())+2)
        
    if(plotCoveredFinal):
        p = ax.scatter(nu,eu,du,c=visStatsFinal,cmap=cm.RdYlGn,s=1)
#        ax.scatter(nu[visStatsFinal >= 3],eu[visStatsFinal >= 3],du[visStatsFinal >= 3],color='green',s=1)
#        ax.scatter(nu[visStatsFinal == 2],eu[visStatsFinal == 2],du[visStatsFinal == 2],color='yellow',s=1)
#        ax.scatter(nu[visStatsFinal == 1],eu[visStatsFinal == 1],du[visStatsFinal == 1],color='yellow',s=1)
#        ax.scatter(nu[visStatsFinal == 0],eu[visStatsFinal == 0],du[visStatsFinal == 0],color='red',s=1)
#        ax.scatter(nu[visStatsFinal == -1],eu[visStatsFinal == -1],du[visStatsFinal == -1],color='0.5',s=1)
        fig.colorbar(p)
        
    if(plotCoveredInitial):
        p = ax.scatter(nu,eu,du,c=visStatsInitial,cmap=cm.RdYlGn,s=1)
#        ax.scatter(nu[visStatsInitial >= 3],eu[visStatsInitial >= 3],du[visStatsInitial >= 3],color='green',s=1)
#        ax.scatter(nu[visStatsInitial == 2],eu[visStatsInitial == 2],du[visStatsInitial == 2],color='yellow',s=1)
#        ax.scatter(nu[visStatsInitial == 1],eu[visStatsInitial == 1],du[visStatsInitial == 1],color='yellow',s=1)
#        ax.scatter(nu[visStatsInitial == 0],eu[visStatsInitial == 0],du[visStatsInitial == 0],color='red',s=1)
#        ax.scatter(nu[visStatsInitial == -1],eu[visStatsInitial == -1],du[visStatsInitial == -1],color='0.5',s=1)
        fig.colorbar(p)
    set_axes_equal(ax)
    return ax

def plotSection(cn,ce,cd,cor_n,cor_e,cor_d,cup_n,cup_e,cup_d,crange):
    '''
    Plots a portion of the camera positions
    '''
    
    # Initialize plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_aspect('equal')
    plt.quiver(cn[crange],ce[crange],cd[crange],cor_n[crange],cor_e[crange],cor_d[crange],color='b',pivot='tail',normalize='True')
    plt.quiver(cn[crange],ce[crange],cd[crange],cup_n[crange],cup_e[crange],cup_d[crange],color='g',pivot='tail',normalize='True')
    
    ax = plt.gca()
    set_axes_equal(ax)
    
    ax.set_xlim(cn[crange].min()-2,cn[crange].max()+2)
    ax.set_ylim(ce[crange].min()-2,ce[crange].max()+2)
    ax.set_zlim(cd[crange].min()-2,cd[crange].max()+2)
    

def plotFrustums(CN,f,n,ax):
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
#    P_orig = P.copy().T
    
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
#    P = np.r_[P,np.ones([1, P.shape[1]])]
    
    # Transform points to camera space
#    P = np.dot(P.T,E)
    
    # Define transform from camera space to image volume space
    A_vt = np.array([[1/np.tan(alphax),0,0,0],
                  [0,1/np.tan(alphay),0,0],
                  [0,0,(f+n)/float((f-n)),-1],
                  [0,0,2*f*n/float((f-n)),0]])
    
#    # Transform points to image volume space
#    P = np.dot(P,A_vt)
#    
#    # Normalize values by last column
#    end = P[:,-1]
#    P = P/end[:,None]
#    
#    # Visible points have all values within range -1 to 1
#    visiblePoints = np.all((P >= -1)*(P <= 1),axis=1)
    
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
    ax.scatter(C[0],C[1],C[2],color=sc.get_edgecolor()[0])
    
#    points = P_orig[visiblePoints,:]
#    camera = C
#    if(len(points)>4):
#        visibleIndex = np.where(visiblePoints==True)[0]
#        nonHidden = hpr(points,camera,1)
##        hidden = np.logical_not(nonHidden)
#        visibleHidden = visibleIndex[nonHidden]
#        visiblePoints[:]=False
#        visiblePoints[visibleHidden]=True
    set_axes_equal(ax)
    return

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])