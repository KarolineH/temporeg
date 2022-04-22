import os
import pheno4d_util as util
import numpy as np
from LeafSurfaceReconstruction.leaf_axis_determination import LeafAxisDetermination
from LeafSurfaceReconstruction.helper_functions import *

def load_plants(directory, annotated_only = True, crop = "Tomato"):
    all_files, annotated_files = util.get_file_locations(directory)
    all_plants = [entry for entry in all_files if crop in entry[0]] #consider all tomato files
    annotated_plants = [entry for entry in annotated_files if crop in entry[0]] #consider all annotated tomato files
    if annotated_only:
        return annotated_plants
    else:
        return all_plants

def process_dataset(data):
    '''
    - Takes a list of lists, containing file locations of the scan sequences for individual plants
    - Isolates the leaves
    - Aligns their axes according to https://github.com/oceam/LeafSurfaceReconstruction
    '''
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
                id = 'plant' + str(i) + '_step' + str(j) + '_leaf' + str(int(leaf[1][0]))
                



            import pdb; pdb.set_trace()
            #util.save_as_ply(pc, 'test_pc.ply')


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



#     np.save(os.path.join(file_directory, 'labels.npy'), full_labels)
#     np.save(os.path.join(file_directory, 'label_IDs.npy'), label_ids)
#     print('Flattened and discretised leaf data set saved to file')
# else:
#     'Data has already been extracted and stacked. Loading data instead.'
#     data = np.load(os.path.join(file_directory, 'flattened_leaves.npy'))
#     full_labels = np.load(os.path.join(file_directory, 'labels.npy'))
#     label_ids = np.load(os.path.join(file_directory, 'label_IDs.npy'), allow_pickle=True)
#
# return data, full_labels, label_ids



if __name__== "__main__":
    raw_data_directory = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D')

    plants = load_plants(raw_data_directory)
    process_dataset(plants)
