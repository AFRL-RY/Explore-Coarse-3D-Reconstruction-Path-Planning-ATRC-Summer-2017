import numpy as np
import collections

def projectPixel(pc,pixelLoc):

    # form the rotation matrix for the platform roll, pitch, yaw
    pr = np.array([[1, 0, 0],
                    [0, np.cos(pc.Roll), np.sin(pc.Roll)],
                    [0, -np.sin(pc.Roll), np.cos(pc.Roll)]])
    pp = np.array([[np.cos(pc.Pitch), 0, -np.sin(pc.Pitch)],
                    [0, 1, 0],
                    [np.sin(pc.Pitch), 0, np.cos(pc.Pitch)]])
    py = np.array([[np.cos(pc.Yaw), np.sin(pc.Yaw), 0],
                    [-np.sin(pc.Yaw), np.cos(pc.Yaw), 0],
                    [0, 0, 1]])
    R_ned2ins = pr.dot(pp).dot(py)

    # form the rotation matrix for the sensor relative roll, pitch, yaw
    sr = np.array([[1, 0, 0],
                  [0, np.cos(pc.SensorRoll), np.sin(pc.SensorRoll)],
                  [0, -np.sin(pc.SensorRoll), np.cos(pc.SensorRoll)]])
    sp = np.array([[np.cos(pc.SensorPitch), 0, -np.sin(pc.SensorPitch)],
                  [0, 1, 0],
                  [np.sin(pc.SensorPitch), 0, np.cos(pc.SensorPitch)]])
    sy = np.array([[np.cos(pc.SensorYaw), np.sin(pc.SensorYaw), 0],
                  [-np.sin(pc.SensorYaw), np.cos(pc.SensorYaw), 0],
                  [0, 0, 1]])
    R_ins2sensor = sr.dot(sp).dot(sy)

    # rotation matrix that defines the transformation between the sensor and
    # camera frame
    R_sensor2cam = np.array([[0,1,0],
                            [0,0,1],
                            [1,0,0]])

    # camera matrix
    K = np.array([[pc.FocalLength,0,pc.ImageWidth/2],
                 [0,pc.FocalLength,pc.ImageHeight/2],
                 [0,0,1]])

    # full rotation matrix from NED to the camera frame
    R_ned2cam = R_sensor2cam.dot(R_ins2sensor).dot(R_ned2ins)

    # perform the projection
    geoLoc_vec = np.transpose(R_ned2cam).dot(np.linalg.inv(K)).dot([pixelLoc[0], pixelLoc[1], 1])
    PlatformAltitude = pc.LLA[2]
    GroundElevation = 185
    scale = (PlatformAltitude - GroundElevation)/geoLoc_vec[2]

    worldLoc = NED_platform + scale * geoLoc_vec

    return worldLoc


# ----------------------------------FourCornersAtterbury
# This four corners projection example uses frame 5500
# from flight 3 of the October 2016 Atterbury-Muscatatuck dataset
#
# The metadata can be found in the ''_nov_span_adis_1_pva.csv
# The frame number is found in the RawImageTopic.csv
# These are synchronized by comparing their timestamps
# Column 2 in RawImageTopic.csv is the GPS weeks, and
# column 3 is the GPS seconds
# FrameNumberTimestamp = GpsWeeks * 7 * 24 * 60 * 60 + GpsSeconds
# MetadataTimestamp = column 3 of the pva csv divided by 1e9
#
# The ground is treated as having a constant 185m (WGS84)
# elevation for simplicity

projectionContext = collections.namedtuple('projectionContext','LLA, NED, \
        Yaw, Pitch, Roll, SensorYaw, SensorPitch, SensorRoll, ImageHeight, \
        ImageWidth,FocalLength')

# the lat, lon, alt of the platform
# taken from columns 5,6 and 7 of the ''_nov_span_adis_1_pva.csv
# the LLA position is converted to North-East-Down before projection
# using a using a NED origin of [39.342827,-86.011968, 185] (directly below the aircraft at ground level)
LLA_platform = np.array([39.342827,-86.011968,524.649069]) # altitude in WGS84
NED_platform = np.array([0,0,-339.649069])

# the yaw, pitch, and roll of the platform generated from the INS
# taken from columns 11,12, and 13 of the ''_nov_span_adis_1_pva.csv
# converted to radians for use in cos() and sin() functions
Roll = -24.619859 * np.pi/180
Pitch = 0.606986  * np.pi/180
Yaw = 148.824885  * np.pi/180

# the sensor relative yaw, pitch, and roll
# not found in the csv, these values take into account the position of the
# camera relative to the INS
# converted to radians for use in cos() and sin() functions
SensorYaw   = -87.217 * np.pi/180
SensorPitch = -26.054 * np.pi/180
SensorRoll = 179.75 * np.pi/180

# the image height,width and focal length are not found in the csv
# these values are used to construct the camera matrix K
ImageHeight = 1200
ImageWidth  = 1600
FocalLength = 3650

# Locations for the four corners
upperLeft  = np.array([0,0])
upperRight = np.array([ImageWidth,0])
lowerRight = np.array([ImageWidth,ImageHeight])
lowerLeft  = np.array([0,ImageHeight])

# These results are in the North-East-Down frame. They can be converted
# back into LLA using a NED origin of [39.342827,-86.011968, 185]
pc = projectionContext(LLA_platform,NED_platform,Yaw,Pitch,Roll,
                       SensorYaw,SensorPitch,SensorRoll,ImageHeight,
                       ImageWidth,FocalLength)
worldLocUL = projectPixel(pc,upperLeft)
worldLocUR = projectPixel(pc,upperRight)
worldLocLR = projectPixel(pc,lowerRight)
worldLocLL = projectPixel(pc,lowerLeft)



