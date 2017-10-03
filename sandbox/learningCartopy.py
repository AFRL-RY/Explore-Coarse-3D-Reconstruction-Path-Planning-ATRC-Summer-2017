import os
import matplotlib.pyplot as plt

from cartopy import config
import cartopy.crs as ccrs

fig = plt.figure(figsize=(8, 12))

fname = 'C:/Users/student/Downloads/m_3908560_se_16_1_20140627/m_3908560_se_16_1_20140627.tif'

img = plt.imread(fname)

ax = plt.axes(projection=ccrs.PlateCarree())

LeftLon = -85.56590486701643
RightLon = -85.49660770437526
BottomLat = 38.996462
TopLat = 39.066032

img_extent = (LeftLon, RightLon, BottomLat, TopLat)

ax.set_xmargin(0.01)
ax.set_ymargin(0.01)

ax.imshow(img, origin='upper', extent=img_extent, transform=ccrs.PlateCarree())

ax.plot(-85.528768,39.049989,'bo',transform=ccrs.Geodetic())

plt.show()