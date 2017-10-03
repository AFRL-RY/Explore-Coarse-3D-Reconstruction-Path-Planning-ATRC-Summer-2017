# -*- coding: utf-8 -*-

from main import run_main
from subprocess import call
import os
import time

start_time = time.clock()

colmap_dir = '/user_data/colmap/build/src/exe/'
print(colmap_dir)

# First Run
print('Iteration 1')
project_directory = run_main('./configFiles/configMuscatAWSLoop1.yml')

# COLMAP
print('Starting Semi-Dense Reconstruction')
call([colmap_dir+'/feature_extractor','--database_path',project_directory+'/database.db','--image_path',project_directory+'/Selected/small','--SiftGPUExtraction.index','0'])
call([colmap_dir+'/exhaustive_matcher','--database_path',project_directory+'/database.db','--SiftMatching.use_gpu','1'])

if not os.path.exists(project_directory+'/sparse'):
        os.makedirs(project_directory+'/sparse')
        
call([colmap_dir+'/mapper','--database_path',project_directory+'/database.db','--image_path',project_directory+'/Selected/small','--export_path',project_directory+'/sparse',
      '--Mapper.ba_local_max_num_iterations','2','--Mapper.ba_global_max_num_iterations','2','--Mapper.ba_global_images_ratio','1.2',
      '--Mapper.ba_global_points_ratio','1.2','--Mapper.ba_global_max_refinements','2'])
call([colmap_dir+'/model_aligner','--input_path',project_directory+'/sparse/0','--ref_images_path',project_directory+'/Selected/position_data_utm.txt','--output_path',project_directory+'/sparse/0','--robust_alignment_max_error','1'])

if not os.path.exists(project_directory+'/dense'):
        os.makedirs(project_directory+'/dense')

call([colmap_dir+'/image_undistorter','--image_path',project_directory+'/Selected/small','--input_path',project_directory+'/sparse/0','--output_path',project_directory+'/dense','--output_type','COLMAP'])

call([colmap_dir+'/dense_stereo','--workspace_path',project_directory+'/dense/','--workspace_format','COLMAP','--DenseStereo.geom_consistency','false',
      '--DenseStereo.window_radius','4','--DenseStereo.num_samples','2','--DenseStereo.num_iterations','3'])
call([colmap_dir+'/dense_fuser','--workspace_path',project_directory+'/dense/','--workspace_format','COLMAP','--input_type','geometric','--output_path',project_directory+'/dense/cloud.ply','--DenseFusion.check_num_images','2'])

# Second Run
print('Iteration 2')
run_main('./configFiles/configMuscatAWSLoop2.yml',project_directory+'/dense/')

print('Final Time: ' + str(time.clock()-start_time))
