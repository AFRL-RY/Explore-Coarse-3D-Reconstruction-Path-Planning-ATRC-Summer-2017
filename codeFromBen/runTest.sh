#!/bin/bash
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
IMAGES_DIR=../ExampleData/Data/2015.09.21_15.39.54/
METADATA_DIR=../ExampleData/MetaData/20150921_Flight1/
echo "python SUSEXSampleReader.py --directory-images $IMAGES_DIR --directory-metadata $METADATA_DIR"
python SUSEXSampleReader.py --directory-images $IMAGES_DIR --directory-metadata $METADATA_DIR --save-dir ../ExampleData/Processed/
