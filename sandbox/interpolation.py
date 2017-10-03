import numpy as np
from scipy.interpolate import Rbf
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Load some ply data

from plyfile import PlyData
from plotting import set_axes_equal

plydata = PlyData.read('C:/Users/student/Documents/Reporting/plyTest.ply')
x = np.array(plydata['vertex']['x'])
y = np.array(plydata['vertex']['y'])
z = np.array(plydata['vertex']['z'])
# Downsample
c = np.random.choice(len(x),int(len(x)/10))
x = x[c]
y = y[c]
z = z[c]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.scatter(x,y,z,s=1,cmap=cm.summer)
set_axes_equal(ax)
ax.set_title('Downsampled')
ax.set_xlabel('North')
ax.set_ylabel('East')
ax.set_zlabel('Elevation')

# Interpolation limits
tx = np.linspace(x.min(),x.max(),int(np.sqrt(len(x))))
ty = np.linspace(y.min(),y.max(),int(np.sqrt(len(y))))
XI,YI = np.meshgrid(tx,ty)

# Grid Data
ZI = griddata(np.array([x.ravel(),y.ravel()]).T,z.ravel(),(XI,YI),method='linear',rescale=True)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.scatter(XI,YI,ZI,s=3,cmap=cm.winter)
set_axes_equal(ax)
ax.set_title('Interpolated')
ax.set_xlabel('North')
ax.set_ylabel('East')
ax.set_zlabel('Elevation')

