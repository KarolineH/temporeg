
import os
import numpy as np
import copy
import leaf_matching
import leaf_encoding
import visualise
import packages.pheno4d_util as util

'''
This is a quick test to see how impactful the starting point in the outline is on feature space recognition.
It takes the same outline, duplicates it 10 times, each time shifting the starting point by one further along the outline
I added 1 very differently shaped leaf at the end to establish a reference frame for normal feature space distances.
Turns out at about 40-odd points offset (less than 10% of the way around) the differently shaped leaf looks more like the original than the exact identical leaf with starting point offset.
'''

# Load data and fit pca
directory = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D', '_processed', 'pca_input')
train_ds, test_ds, train_labels, test_labels, pca, transformed = leaf_encoding.get_encoding(train_split=0, dir=directory, location=False, rotation=False, scale=False, as_features=True)
data, labels = util.sort_examples(train_ds, train_labels) # sort

subset, sub_labels = leaf_encoding.select_subset(data, labels, plant_nr = 0, timestep=4, leaf=None)
selected_leaves = subset[[0,5],:] # leaf 2 and 7
dist = leaf_matching.make_dist_matrix(selected_leaves, selected_leaves, pca, draw=False, components=200)

# shift
stacked, additional_features = leaf_encoding.reshape_coordinates_and_additional_features(selected_leaves, nr_coordinates=500)
shifted_versions = []
loop = copy.deepcopy(stacked[0])

for i in range(10):
    shifted_loop = np.append(loop, np.reshape(loop[0], (-1, 3)), axis=0)
    shifted_loop = np.delete(shifted_loop, (0), axis=0)
    shifted_versions.append(shifted_loop)
    loop = shifted_loop

all_outlines = np.append(np.asarray(shifted_versions), stacked, axis = 0)
all_outlines = all_outlines.reshape(all_outlines.shape[0], all_outlines.shape[1]*all_outlines.shape[2])
if additional_features.shape[1] > 0:
    add_f = np.tile(additional_features[0], (11, 1))
    add_f = np.concatenate((add_f, [additional_features[1]]), axis = 0)
    all_outlines = np.concatenate((all_outlines, add_f), axis=1)

dist = leaf_matching.make_dist_matrix(all_outlines, all_outlines, pca, draw=True, components=200)

clouds = [visualise.array_to_o3d_pointcloud(outline) for outline in all_outlines]
visualise.draw2(clouds, "test_shapes", offset=True)
import pdb; pdb.set_trace()
