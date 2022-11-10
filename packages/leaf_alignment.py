import os
import packages.pheno4d_util as util
import numpy as np
from packages.LeafSurfaceReconstruction.leaf_axis_determination import LeafAxisDetermination
from packages.LeafSurfaceReconstruction.helper_functions import *
import packages.locate_emergence_point as emergence

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

def isolate_and_align_leaves(data, out_directory, transform_directory):
    '''
    - Takes a list of lists, containing file locations of the scan sequences for individual plants
    - Isolates the leaves
    - Aligns their axes according to https://github.com/oceam/LeafSurfaceReconstruction
    - Saves them to .ply files
    '''
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)
    if not os.path.exists(transform_directory):
        os.makedirs(transform_directory)

    leaf_ids = {}
    for i, plant_series in enumerate(data):
        starting_day = int(os.path.split(plant_series[0])[-1].split('_')[1]) # the date of the first scan in the series
        for j, time_step in enumerate(plant_series): # perform this for inividual scans
            day_nr = int(os.path.split(time_step)[-1].split('_')[1]) - starting_day
            points,labels,plant_id = util.open_file(time_step)
            leaves, emergence_point = isolate_leaves(points, labels) # leaves w.r.t. plant emergence point
            # to get back to original coordinate system, simply add the emergence point onto each coordinate

            for leaf in leaves:
                leafAxisDetermination = LeafAxisDetermination(leaf[0]) # takes the coordinates only, no labels needed
                w_axis, l_axis, h_axis = leafAxisDetermination.process() # give us the leaf axes
                # Some leaves are upside down at this stage (x and y both flipped) because they grow 'towards' the emergence point, not away

                centroid = np.mean(leaf[0], axis=0) # should be centroid w.r.t. emergence point
                #translate the leaf point cloud to its centroid, rather than plant emergence point
                pc = leaf[0] - centroid
                # rotate to the extracted leaf coordinate system
                pc = transform_axis_pointcloud(pc, w_axis, l_axis, h_axis)

                # Save cloud
                id = 'plant' + str(i) + '_step' + str(j) + '_day' + str(day_nr) + '_leaf' + str(int(leaf[1][0])) + '.ply'
                filename = os.path.join(out_directory, id)
                util.save_as_ply(pc, filename)

                # Save the global position and rotation information
                id = 'plant' + str(i) + '_step' + str(j) + '_day' + str(day_nr) + '_leaf' + str(int(leaf[1][0])) + '.npy'
                filename = os.path.join(transform_directory, id)
                transformation_log = np.array([centroid, w_axis, l_axis, h_axis])
                np.save(filename, transformation_log)

def isolate_leaves(points, labels):
    '''Takes one individual full plant point cloud,
    returns only leaves, given in coordinates with respect to the plant origin point'''
    organs = util.split_into_organs(points, labels) # still in the original frame of the data set
    emergence_point = emergence.pipeline(points, labels) # find the plant emergence point
    shifted_organs = [((organ[0] - emergence_point),organ[1]) for organ in organs] #isolate and change basis to emergence
    return shifted_organs[2:], emergence_point

if __name__== "__main__":
    raw_data_directory = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D')
    out_directory = os.path.join(raw_data_directory, '_processed', 'aligned')
    out_directory = os.path.join(raw_data_directory, '_processed', 'transform_log')
    plants = find_plant_locations(raw_data_directory)
    isolate_and_align_leaves(plants, out_directory, transform_dir)
