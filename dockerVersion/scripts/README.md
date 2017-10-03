## Synopsis

This is an automated view planning tool for 3D reconstruction.  The SelectImages mode selects a minimum image set from existing images using terrain data and image metadata.  The PlanFlight mode generates the optimal positions for a minimal image set and plans the shortest path through those points.

## Motivation

Generating 3D models from UAV imagery is very useful.  However, generating 3D models from lots of UAV imagery takes a **very** long time.  This tool aims to reduce the time and processing resources required by preselecting images from the available set.  Ideally, this minimal image set should be able to reconstruct the scene with little loss of quality, but with large time savings.
See [the project VDL page](https://restricted.vdl.afrl.af.mil/programs/atrpedia/dist_c/wiki/Coarse_3D_Reconstruction_Path_Planning) for more information

## Installation

The code has several prerequisites:

First make sure that the [Anaconda distribution](https://www.continuum.io/downloads) is installed with Python 2.7.  Then install the following dependencies

```
pip install utm plyfile munkres  
pip install git+git://github.com/AndrewWalker/pydubins.git  
```

The GDAL package for python is also required.  For Windows, the easiest way I found to install this is by downloading the binary for your system from [Christoph Gohlke's Unofficial Windows Binaries for Python Extension Packages](http://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal).
The package can then be installed with the following command, adapting to the specific file you downloaded:
```
pip install GDAL-X.Y.Z-cp27-none-win_XYZ.whl
```

ExifTool is required to geotag the selected images.  It can be downloaded [here](http://www.sno.phy.queensu.ca/~phil/exiftool), and must be added to the system path so that Python can find it.

## Runing the Code

Once the prerequisites are installed, running the code should be simple.  The first step is to create a new YAML configuration file containing the paths to your images, metadata, and terrain data.  An example configuration file is given below, as well as a description of the various options and parameters.  The code can then be run from the terminal as follows:

```
python main.py /path/to/configFile/configFile.yml
```

The output of the code will be a new timestamped results folder in your image directory.  This will contain:

* A folder named Selected that contains the selected images
* A folder named small with quarter size images
* A list of the selected frames
* A csv of the coordinates of the selected images in lat, lon
* A matching pairlist for VisualSFM
* A copy of the configuration file

## Config File Reference

Example configuration file
```yaml
dirs:
    # Path to folder containing DTED file
    directory_terrain: 'E:/21_Sept_2015_WAMI_Flight_1/Terrain'
    # Path to folder containing imagery
    directory_images: 'E:/2015.09.24_10.33.46'
    # Path to folder containing metadata
    directory_metadata: 'E:/Bags/20150924_Flight1a'
    # Type of metadata ('Muscat', or 'Medina')
    metadata_type: 'Muscat'
run:
    # Choose from 'SelectImages' or 'PlanFlight'
    mode: 'SelectImages'
terrain:
    # Choose from 'PLY' or 'DTED'
    source: 'DTED'
crop:
    # Crop terrain data to an area of interest
    # Type of cropping ('KML','FC','BB','INTERP')
    cropType: 'INTERP'
    # Path to kml file containing a polygon around the area of interest.
    path_POI: 'C:/Users/student/Documents/Project/POI/muscat.kml'
    # Four corners to crop to
    FC_tl: [0,1]
    FC_tr: [1,1]
    FC_bl: [0,0]
    FC_br: [1,0]
    # extra border size cropped area (m)
    pad: 50
dome:
    # Add dome to target area to promote side views
    enable: True
    # Height of added dome (m)
    height: 50
angle_sorting:
    # Vertical bins
    vertical: 3
    max_incidence: 45
    # Horizontal bins
    horizontal: 5
angle_reduction:
    # Adjust FOV to increase overlap
    enable: True
    # Adjustment for visibility checks (0-1)
    visibility: 0.5
overlap_enhancement:
    # Promote centered points
    enable: True
    ratio: 0.1
cam_range:
    # Camera far visibility limit ( far * max(cam_elevation))
    far: 1.25
    # Camera near visibility limit (m)
    near: 200
selection:
    # Coverage tolerance for greedy algorithm
    tol: 1   
plotting:
    # Do plots (with associated calculations)
    enable: True
    # Turn various plots on and off
    initialImages: True
    finalImages: True
    initialCoverage: True
    finalCoverage: True
    frustums: True
    flightPath: True
output:
    # Write selected frame list, copy images to Selected folder,
    # Geotag images, and write pair list
    enable: True
    # Number of matches between images in pair list
    pair_list_matches: 8
flightPath:
    # Desired elevation (elipsoid) for flight path (m)
    elevation: 600
    # Horizontal field of view (deg)
    alphax: 30
    # Vertical field of view (deg)
    alphay: 30
    # Minimum turn radius (m)
    min_turn: 100
```

Explanation of configuration parameters

### dirs

* directory_terrain
    * Path to folder containing terrain data.  This can either be one more more DTED .dt2 files, or a PLY file containing a georeferenced point cloud.
* directory_images
    * Path to folder containing the input image set.  These should be in .jpg format.
* directory_metadata
    * Path to folder containing metadata for the images.  The program is currently setup to support metadata from the SUSEX flights, or Predator data.
* metadata_type
    *  ('Muscat','Medina') - CHANGE THESE

### run

* mode ('SelectImages','PlanFlight')
    * Choose from selection or planning mode. 
        * Selection mode selects a subset of images from an original set.  It outputs a folder with the selected images.  
        * Planning mode plans a set of image locations to reconstruct a target, and plans a path through the points.

### terrain

* source ('DTED','PLY')
    * Select which format the input terrain data is in.
        * DTED: Use DTED data from a .dt2 file (Can be obtained from USGS Earth Explorer STRM layers)
        * PLY: Georeferenced point cloud file in .ply format

### crop

* cropType ('KML','FC','BB','INTERP')
    * This parameter sets how the input terrain data is cropped to the area of interest.  
        *  KML:  [Crop](https://restricted.vdl.afrl.af.mil/programs/atrpedia/dist_c/wiki/Coarse_3D_Reconstruction_Path_Planning#Point_of_Interest_Cropping) according to a polygon in a kml file.  This file can be defined in Google Earth by drawing a polygon around the target, right clicking, and using save as .kml.  Note that the terrain is cropped to a rectangle bounding the polygon, not the polygon itself.  For DTED data INTERP is recommended instead of KML.
        *  FC:  Crop according to four corners data, or four coordinates.  Requires that the FC parameters be set below.
        *  BB:  Crop to the bounding box of the input image set.  This can result in large terrain areas and slow processing, and should be used with care.
        *  INTERP:  Extension of KML cropping that first [increases the density](https://restricted.vdl.afrl.af.mil/programs/atrpedia/dist_c/wiki/Coarse_3D_Reconstruction_Path_Planning#DTED_Interpolation_and_Densification) of the original terrain data, and then crops.  This avoids cropping to zero data when the input terrain is sparse.  Recommended for use with DTED data.
* path_POI
    * Path to kml file containing a polygon defining the region of interest.  Used when cropType == 'KML' or 'INTERP'.
* FC_tl
    * Top left corner of four corners area in North, East UTM coordinates
* FC_tr
    * Top right corner of four corners area in North, East UTM coordinates
* FC_bl
    * Bottom left corner of four corners area in North, East UTM coordinates
* FC_br
    * Bottom right corner of four corners area in North, East UTM coordinates
* pad
    * Extra padding (in meters) to add around selection area.  Used in KML, FC, and BB.  INTERP uses a pad of 0.

### dome

* enable ('True','False')
    * [Adds a dome](https://restricted.vdl.afrl.af.mil/programs/atrpedia/dist_c/wiki/Coarse_3D_Reconstruction_Path_Planning#Week_4) to the cropped terrain to promote side views of the target when there is limited data available.  The dome is 2D Gaussian centered on the mean of the cropped terrain, with a standard deviation in each direction of one half the width of the cropped terrain in that direction.  Most useful with DTED data.
* height
    * Amplitude (in meters) of the added dome shape.  Larger amplitudes give greater angles on the sides of the dome, and better side views, but can also lead to extraneous cameras being selected due to the exaggerated height of the terrain.  15-20 meters usually works well.

### angle_sorting
The algorithm will attempt to select images that view each terrain point from each of these angles.

* vertical
    * Number of [vertical bins](https://restricted.vdl.afrl.af.mil/programs/atrpedia/dist_c/wiki/Coarse_3D_Reconstruction_Path_Planning#Week_11) to use in the angle visibility matrix.  Angle ranges are created from the surface normal to the max incidence angle.
* max_incidence
    * Max angle (in degrees) away from the surface normal that the algorithm will attempt to find views for.  45-60 degrees is a good range, since image matching becomes difficult for SFM algorithms at larger angles.
* horizontal
    * Number of horizontal bins to use in the angle visibility matrix.  This divides the 360 degree heading range into sections, and attempts to view each terrain point with at least one image in each section.

### angle_reduction

* enable (True,False)
    * Reduces field of view values read from image metadata to give a conservative version of what each image can see.  Useful to ensure that the target is within the field of view when position and pose metadata is less accurate than desired.
* visibility (0-1)
    * Ratio by which to reduce the horizontal and vertical fields of view.  0.5 usually gives good results.

### overlap_enhancement

* enable
    * [Adds a Gaussian scoring function](https://restricted.vdl.afrl.af.mil/programs/atrpedia/dist_c/wiki/Coarse_3D_Reconstruction_Path_Planning#Overlap_Enhancement) to the visibility tests in addition to a strict yes/no test.  Helps to increase image overlap at the target location.  In effect, this makes it so that terrain points that are only viewed at the edge of an image require views from additional images to be considered covered.
* ratio (0-1)
    * The scoring for all images must sum to at least this threshold for the terrain point to be considered covered.  0.1 tends to work well for many cases.  CHANGE THIS TO THRESHOLD

### cam_range

* far (ratio)
    * Camera far plane for visibility tests.  Far plane is calculated as ( far * max(cam_elevation)).  1.25 is usually sufficient.
* near (meters)
    * Camera near plane for visibility tests (in meters).

### selection

* tol (0-1)
    * Greedily select the images that can see the most points from the most angles until all target points are visible from all angles covered in the original image set, multiplied by tolerance.  Higher tolerance value selects more images.  A lower tolerance can be useful when a large number of images are required to cover the last few terrain points.

### plotting

* enable (True,False)
    * Enable plotting
* initialImages (True,False)
    * Plot locations of initial image set
* finalImages (True,False)
    * Plot locations of chosen image set
* initialCoverage (True,False)
    * Plot terrain coverage by initial image set.  Values are from the gaussian scoring function.  Higher values = More coverage.
* finalCoverage (True,False)
    * Plot terrain coverage by selected image set.
* frustums (True,False)
    * Plot frustums of selected images in 3D.  Not especially useful since pythons 3D plotting doesn't respect depth order when drawing.
* flightPath (True,False)
    * Plot planned flight path (in planning mode.)

### output

* enable (True,False)
    * Enable data output.  
        * Creates results folder in image directory, 
        * Writes list of chosen frames, 
        * Copies chosen images to Selected folder, 
        * Geotags chosen images using ExifTool, 
        * Writes a pair list for VisualSFM based on maximum overlap,
        * Copies the configuration file to the results folder,
        * Creates a quarter size copy of the selected images for fast 3D processing
* pair_list_matches
    * Number of matches between images in pair list (Default: 8)

### flightPath
These options are used in PlanFlight mode.  

* elevation
    * Desired elevation (elipsoid) for flight path (in meters).  Images locations are found by projecting from the surface normal to the intersection with this elevation.
* alphax
    * Horizontal field of view for planned images (in degrees).
* alphay
    * Vertical field of view for planned images (in degrees).
* min_turn
    * Minimum turning radius for planned Dubins path.

## Contributors

To contribute to this project, contact @benjamin.heiner

