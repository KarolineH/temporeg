import os
import pheno4d_util as util
import numpy as np
from LeafSurfaceReconstruction.leaf_axis_determination import LeafAxisDetermination
from LeafSurfaceReconstruction.helper_functions import *
import leaf_alignment
import pc_to_mesh

# inputs
raw_data_directory = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D')
aligned_leaf_directory = os.path.join(raw_data_directory, '_processed', 'aligned')
mesh_directory = os.path.join(raw_data_directory, '_processed', 'meshed')
outline_directory = os.path.join(raw_data_directory, '_processed', 'outline')

# alignment of leaves
plants = leaf_alignment.find_plant_locations(raw_data_directory)
leaf_alignment.isolate_and_align_leaves(plants, aligned_leaf_directory) # isolate and align leaves, save to file

# meshing, smoothing
pc_to_mesh.mesh_and_smooth(aligned_leaf_directory, mesh_directory, maxLength=1, smoothIterations=20, smoothFactor=0.2)

# extracting the outline (using Blender)
if not os.path.exists(outline_directory):
    os.makedirs(outline_directory)
command_string = 'blender --background --python boundary_extraction.py -- ' + str(mesh_directory) + ' ' + str(outline_directory)
os.system(command_string)
