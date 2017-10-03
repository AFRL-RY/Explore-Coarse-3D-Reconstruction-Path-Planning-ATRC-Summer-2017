# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 22:16:36 2016

@author: oacom
"""
import numpy as np
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
P = np.array([1,0,1.5])

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
alphax = np.radians(10)
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
print Pt2n
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
P2 = np.array([[2.5,1],[0,0],[1,0.5]])

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
print P2t2n
visiblePoints = np.all((P2t2n >= -1)*(P2t2n <= 1),axis=1)