import os
import pheno4d_util as util
import numpy as np
from LeafSurfaceReconstruction.leaf_axis_determination import LeafAxisDetermination
from LeafSurfaceReconstruction.helper_functions import *

def find_plant_locations(directory, annotated_only = True, crop = "Tomato"):
    '''
    Loads plant scans from the Pheno4D data set
    '''
    all_files, annotated_files = util.get_file_locations(directory)
    all_plants = [entry for entry in all_files if crop in entry[0]] #consider all tomato files
    annotated_plants = [entry for entry in annotated_files if crop in entry[0]] #consider all annotated tomato files
    if annotated_only:
        return annotated_plants
    else:
        return all_plants

def isolate_and_align_leaves(data, out_directory):
    '''
    - Takes a list of lists, containing file locations of the scan sequences for individual plants
    - Isolates the leaves
    - Aligns their axes according to https://github.com/oceam/LeafSurfaceReconstruction
    - Saves them to .ply files
    '''
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)

    leaf_ids = {}
    for i, plant_series in enumerate(data):
        for j, time_step in enumerate(plant_series): # perform this for inividual scans
            points,labels,plant_id = util.open_file(time_step)
            leaves = isolate_leaves(points, labels) # leaves w.r.t. plant origin
            for leaf in leaves:
                leafAxisDetermination = LeafAxisDetermination(leaf[0])
                w_axis, l_axis, h_axis = leafAxisDetermination.process()
                # translate the leaf point cloud to its centroid, rather than plant emergence point
                pc = leaf[0] - np.mean(leaf[0], axis=0)
                # translate to extracted leaf coordinate system
                pc = transform_axis_pointcloud(pc, w_axis, l_axis, h_axis)
                id = 'plant' + str(i) + '_step' + str(j) + '_leaf' + str(int(leaf[1][0])) + '.ply'
                filename = os.path.join(out_directory, id)
                util.save_as_ply(pc, filename)

def isolate_leaves(points, labels):
    '''Takes one individual full plant point cloud,
    returns only leaves, given in coordinates with respect to the plant origin point'''
    organs = util.split_into_organs(points, labels)
    leaves = []
    instance_labels_lst = []

    # find the center of the soil patch, as a proxy for plant emergence point, make that the new origin
    plant_origin = np.mean(organs[0][0], axis=0)
    shifted_organs = [((organ[0] - plant_origin),organ[1]) for organ in organs] #isolate and change basis
    return shifted_organs[2:]

if __name__== "__main__":
    raw_data_directory = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D')
    out_directory = os.path.join(raw_data_directory, '_processed', 'aligned')
    plants = find_plant_locations(raw_data_directory)
    isolate_and_align_leaves(plants, out_directory)
