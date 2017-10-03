# -*- coding: utf-8 -*-

import numpy as np

def sortAngles(vis_cors_row,norm_n,norm_e,norm_d,terrain):
    '''
    This function sorts the visible points in the histogram by camera viewing angle
    This lets us specify that points must be covered from a variety of angles
    INPUTS
    vis_cors: Contains one row of the visibility histogram matrix, and the 
            orientation of the corresponding camera [-3:]
    norm_n: North component of surface normals
    norm_e: East component of surface normals
    norm_d: Vertical component of surface normals
    OUTPUTS
    visibilityRowAngles: Histogram row duplicated to account for each angle bin
    '''
    # Camera location vector
    cam = vis_cors_row[-6:-3]
    
    # Camera orientation vector
    cor = vis_cors_row[-3:]
    
    # Visibility for the camera
    vis = vis_cors_row[:-6]
    
    # Flatten 2D grids to 1D vectors
    nu = np.ravel(terrain.nn)
    eu = np.ravel(terrain.ee)
    du = np.ravel(terrain.dd)
    
    # Select surface normals for visible points
    ind = np.where(vis==True)[0]
    norms = np.c_[norm_n[ind],norm_e[ind],norm_d[ind]]
    points = np.c_[nu[ind],eu[ind],du[ind]]
    
    # Set up visibility bins
    visbin1 = np.zeros(vis.shape, dtype=bool) # Normal 1
    visbin2 = np.zeros(vis.shape, dtype=bool) # Normal 2
    visbin3 = np.zeros(vis.shape, dtype=bool) # Normal 3
    
    # Additional angle projection bins
    visbin4 = np.zeros(vis.shape, dtype=bool) # Heading 1
    visbin5 = np.zeros(vis.shape, dtype=bool) # Heading 2
    visbin6 = np.zeros(vis.shape, dtype=bool) # Heading 3
    visbin7 = np.zeros(vis.shape, dtype=bool) # Heading 4
    visbin8 = np.zeros(vis.shape, dtype=bool) # Heading 5
    
    # Step through the visible points
    angles = np.zeros(len(ind[:]))
    for index, row in enumerate(norms):
        # Calculate the angle of incidence
        angle = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(cor,-row)),np.dot(cor,-row)))
        angles[index] = angle
        #### Use angle to put True value in correct index of each bin
        if(abs(angle) <= 15):
            visbin1[ind[index]] = True
        elif(abs(angle) <= 30):
            visbin2[ind[index]] = True
        elif(abs(angle) <= 45):
            visbin3[ind[index]] = True
            
        # Additional heading angles
        row_points = points[index,:]
        diff = row_points - cam
        angle = np.rad2deg(np.arctan2(diff[1],diff[0])) + 180
        if(angle >= 0 and angle < 72):
            visbin4[ind[index]] = True
        if(angle >= 72 and angle < 72*2):
            visbin5[ind[index]] = True
        if(angle >= 72*2 and angle < 72*3):
            visbin6[ind[index]] = True
        if(angle >= 72*3 and angle < 72*4):
            visbin7[ind[index]] = True
        if(angle >= 72*4 and angle <= 72*5):
            visbin8[ind[index]] = True
        
    
    # Combine all visibility bins into one big row for this camera
    visibilityRowAngles = np.r_[visbin1,visbin2,visbin3,visbin4,visbin5,visbin6,visbin7,visbin8]
    return visibilityRowAngles

def sortAnglesOverlap(vis_cors_row,normals,terrain,cfg):
    '''
    This function sorts the visible points in the histogram by camera viewing angle
    This lets us specify that points must be covered from a variety of angles
    INPUTS
    vis_cors: Contains one row of the visibility histogram matrix, and the 
            orientation of the corresponding camera [-3:]
    norm_n: North component of surface normals
    norm_e: East component of surface normals
    norm_d: Vertical component of surface normals
    OUTPUTS
    visibilityRowAngles: Histogram row duplicated to account for each angle bin
    '''
    # Camera location vector
    cam = vis_cors_row[-6:-3]
    
    # Camera orientation vector
    cor = vis_cors_row[-3:]
    
    # Visibility for the camera
    vis = vis_cors_row[:-6]
    
    # Flatten 2D grids to 1D vectors
    nu = np.ravel(terrain.nn)
    eu = np.ravel(terrain.ee)
    du = np.ravel(terrain.dd)
    
    # Select surface normals for visible points
    ind = np.where(vis>0)[0]
    norms = np.c_[normals.norm_n[ind],normals.norm_e[ind],normals.norm_d[ind]]
    points = np.c_[nu[ind],eu[ind],du[ind]]
    
    # Set up visibility bins
    totalBins = cfg['angle_sorting']['vertical']+cfg['angle_sorting']['horizontal']
    visbin = np.zeros([len(vis),totalBins])
#    visbin1 = np.zeros(vis.shape) # Normal 1
#    visbin2 = np.zeros(vis.shape) # Normal 2
#    visbin3 = np.zeros(vis.shape) # Normal 3
#    
#    # Additional angle projection bins
#    visbin4 = np.zeros(vis.shape) # Heading 1
#    visbin5 = np.zeros(vis.shape) # Heading 2
#    visbin6 = np.zeros(vis.shape) # Heading 3
#    visbin7 = np.zeros(vis.shape) # Heading 4
#    visbin8 = np.zeros(vis.shape) # Heading 5
    
    # Step through the visible points
    angles = np.zeros(len(ind[:]))
    for index, row in enumerate(norms):
        # Calculate the angle of incidence
        angle = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(cor,-row)),np.dot(cor,-row)))
        angles[index] = angle
        #### Use angle to put visibility value in correct index of each bin
#        if(abs(angle) <= 15):
#            visbin1[ind[index]] = vis[ind][index]
#        elif(abs(angle) <= 30):
#            visbin2[ind[index]] = vis[ind][index]
#        elif(abs(angle) <= 45):
#            visbin3[ind[index]] = vis[ind][index]
        
        binsize_v = cfg['angle_sorting']['max_incidence']/cfg['angle_sorting']['vertical']
        for i in range(cfg['angle_sorting']['vertical']):
            if(abs(angle) >= i*binsize_v and abs(angle) < (i+1)*binsize_v):
                visbin[ind[index],i] = vis[ind][index]
            
        # Additional heading angles
        row_points = points[index,:]
        diff = row_points - cam
        angle = np.rad2deg(np.arctan2(diff[1],diff[0])) + 180
#        if(angle >= 0 and angle < 72):
#            visbin4[ind[index]] = vis[ind][index]
#        if(angle >= 72 and angle < 72*2):
#            visbin5[ind[index]] = vis[ind][index]
#        if(angle >= 72*2 and angle < 72*3):
#            visbin6[ind[index]] = vis[ind][index]
#        if(angle >= 72*3 and angle < 72*4):
#            visbin7[ind[index]] = vis[ind][index]
#        if(angle >= 72*4 and angle <= 72*5):
#            visbin8[ind[index]] = vis[ind][index]
            
        binsize_h = 360/cfg['angle_sorting']['horizontal']
        for i in range(cfg['angle_sorting']['horizontal']):    
            if(angle >= i*binsize_h and angle < (i+1)*binsize_h):
                visbin[ind[index],i+cfg['angle_sorting']['vertical']] = vis[ind][index]
        
    
    # Combine all visibility bins into one big row for this camera
#    visibilityRowAngles = np.r_[visbin1,visbin2,visbin3,visbin4,visbin5,visbin6,visbin7,visbin8]
    visibilityRowAngles2 = visbin.flatten('F')
    return visibilityRowAngles2

def recordAngles(vis_cors_row,norm_n,norm_e,norm_d,terrain):
    '''
    This function sorts the visible points in the histogram by camera viewing angle
    This lets us specify that points must be covered from a variety of angles
    INPUTS
    vis_cors: Contains one row of the visibility histogram matrix, and the 
            orientation of the corresponding camera [-3:]
    norm_n: North component of surface normals
    norm_e: East component of surface normals
    norm_d: Vertical component of surface normals
    OUTPUTS
    visibilityRowAngles: Histogram row duplicated to account for each angle bin
    '''
    # Camera location vector
    cam = vis_cors_row[-6:-3]
    
    # Camera orientation vector
    cor = vis_cors_row[-3:]
    
    # Visibility for the camera
    vis = vis_cors_row[:-6]
    
    # Flatten 2D grids to 1D vectors
    nu = np.ravel(terrain.nn)
    eu = np.ravel(terrain.ee)
    du = np.ravel(terrain.dd)
    
    # Select surface normals for visible points
    ind = np.where(vis==True)[0]
    print(ind.max())
    norms = np.c_[norm_n[ind],norm_e[ind],norm_d[ind]]
    points = np.c_[nu[ind],eu[ind],du[ind]]
    
    # Set up visibility bins
    visbin1 = np.zeros(vis.shape) # Normal 1
    visbin2 = np.zeros(vis.shape) # Normal 2
    visbin3 = np.zeros(vis.shape) # Normal 3
    
    # Additional angle projection bins
    visbin4 = np.zeros(vis.shape) # Heading 1
    visbin5 = np.zeros(vis.shape) # Heading 2
    visbin6 = np.zeros(vis.shape) # Heading 3
    visbin7 = np.zeros(vis.shape) # Heading 4
    visbin8 = np.zeros(vis.shape) # Heading 5
    
    # Step through the visible points
    angles = np.zeros(len(ind[:]))
    for index, row in enumerate(norms):
        # Calculate the angle of incidence
        angle = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(cor,-row)),np.dot(cor,-row)))
        angles[index] = angle
        #### Use angle to put True value in correct index of each bin

        visbin1[ind[index]] = angle

        visbin2[ind[index]] = angle

        visbin3[ind[index]] = angle
            
        # Additional heading angles
        row_points = points[index,:]
        diff = row_points - cam
        angle = np.rad2deg(np.arctan2(diff[1],diff[0])) + 180
        if(angle >= 0 and angle < 72):
            visbin4[ind[index]] = angle
        if(angle >= 72 and angle < 72*2):
            visbin5[ind[index]] = angle
        if(angle >= 72*2 and angle < 72*3):
            visbin6[ind[index]] = angle
        if(angle >= 72*3 and angle < 72*4):
            visbin7[ind[index]] = angle
        if(angle >= 72*4 and angle <= 72*5):
            visbin8[ind[index]] = angle
        
    
    # Combine all visibility bins into one big row for this camera
    visibilityRowAngles = np.r_[visbin1,visbin2,visbin3,visbin4,visbin5,visbin6,visbin7,visbin8]
    return visibilityRowAngles

if __name__ == "__main__":
    print('Testing')