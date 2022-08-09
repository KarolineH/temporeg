import os
import numpy as np
import leaf_encoding
import organ_matching
import packages.pheno4d_util as util

def count_correct_pairs(data, labels):
    count = 0
    total = 0
    # seperate and loop over individual plants
    plant_splits = []
    for i in range(max(labels[:,0])):
        split = labels[:,0].searchsorted(i+1)
        plant_splits.append(split)
    plants = np.split(data, plant_splits)
    plant_labels = np.split(labels, plant_splits)

    # For each plant...
    for i,plant in enumerate(plant_labels):
        # Separate and loop over individual time steps
        time_splits = []
        for j in range(max(plant[:,1])):
            split = plant[:,1].searchsorted(j+1)
            time_splits.append(split)
        timesteps = np.split(plants[i], time_splits)
        timestep_labels = np.split(plant_labels[i], time_splits)

        # For each time step...
        for step in range(len(timestep_labels)-1):
            # get the distance matrix for each pre-post leaf pairing
            pre = timesteps[step]
            post = timesteps[step + 1]
            dist = organ_matching.make_dist_matrix(pre, post, pca, draw=False, components=200)
            match, legible_matches = organ_matching.compute_assignment(dist, timestep_labels[step], timestep_labels[step+1])
            for pair in legible_matches:
                total += 1
                if pair[0,2] == pair[1,2]:
                    count += 1
    return count, total

if __name__== "__main__":
    # Load data and fit pca
    directory = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D', '_processed', 'pca_input_maxtest')
    train_ds, test_ds, train_labels, test_labels, pca, transformed = leaf_encoding.get_encoding(train_split=0, dir=directory)

    #sort
    data, labels = util.sort_examples(train_ds, train_labels)

    score, total = count_correct_pairs(data, labels)
    print(f"{score} out of {total}")
