import os
from packages import pheno4d_util as util
import numpy as np
from packages.LeafSurfaceReconstruction.leaf_axis_determination import LeafAxisDetermination
from packages.LeafSurfaceReconstruction.helper_functions import *
from packages import leaf_alignment
from packages import pc_to_mesh
from packages.z_smoothing import z_smoothing_operation
from packages import outline_sampling

# inputs
raw_data_directory = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D')
aligned_leaf_directory = os.path.join(raw_data_directory, '_processed', 'aligned')
z_smoothed_directory = os.path.join(raw_data_directory, '_processed', 'z_smoothed')
mesh_directory = os.path.join(raw_data_directory, '_processed', 'meshed')
outline_directory = os.path.join(raw_data_directory, '_processed', 'outline')
pca_input_directory = os.path.join(raw_data_directory, '_processed', 'pca_input')

# alignment of leaves
plants = leaf_alignment.find_plant_locations(raw_data_directory)
leaf_alignment.isolate_and_align_leaves(plants, aligned_leaf_directory) # isolate and align leaves, save to file

# z smoothing
z_smoothing_operation(aligned_leaf_directory, z_smoothed_directory, radius=1.2, smoothIterations=8, smoothFactor=0.2)

# meshing, smoothing
pc_to_mesh.mesh_and_smooth(z_smoothed_directory, mesh_directory, maxLength=1, smoothIterations=20, smoothFactor=0.2)

# extracting the outline (using Blender)
if not os.path.exists(outline_directory):
    os.makedirs(outline_directory)
command_string = 'blender --background --python boundary_extraction.py -- ' + str(mesh_directory) + ' ' + str(outline_directory)
os.system(command_string)

# Sampling the pca inputs from the outline
outline_sampling.sampling_pca_input(outline_directory, pca_input_directory, n = 500)
