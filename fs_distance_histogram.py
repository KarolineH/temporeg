import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import leaf_encoding
import leaf_matching
import packages.pheno4d_util as util

def plot_feature_space_distance_histogram(same_leaf_distances, different_leaf_distances):
    fig, axs = plt.subplots(2, 1, tight_layout=True)
    axs[0].hist(same_leaf_distances, bins = 50)
    axs[0].set_title('Same leaf')
    #axs[0].set_xlim([0, 100])
    axs[0].set_xlabel('Feature space distance', fontsize=10)
    axs[0].set_ylabel('Frequency', fontsize=10)
    axs[1].hist(different_leaf_distances, bins = 50)
    axs[1].set_title('Different leaf')
    #axs[1].set_xlim([0, 100])
    axs[1].set_xlabel('Feature space distance', fontsize=10)
    axs[1].set_ylabel('Frequency', fontsize=10)
    plt.show()

def fs_distances_between_steps(centroids, centroid_labels, data, labels, PCAH, components=50):
    '''
    Loops over all given data, records the feature space distnaces of the same leaves and different leaves
    Plots a histogram of the results.
    '''

    same_leaf_distances = np.array([])
    different_leaf_distances = np.array([])
    same_leaf_distances_c = np.array([])
    different_leaf_distances_c = np.array([])

    # Compare timesteps, pairwise (before and after)
    for plant in np.unique(labels[:,0]): # for each plant
        for time_step in range(np.unique(labels[:,1]).size-1):
            before, before_labels = leaf_encoding.select_subset(data, labels, plant_nr = plant, timestep=time_step, leaf=None)
            after, after_labels = leaf_encoding.select_subset(data, labels, plant_nr = plant, timestep=(time_step+1), leaf=None)
            dist = leaf_matching.make_fs_dist_matrix(before, after, PCAH, mahalanobis_dist = True, draw=False, components=components)

            before_c = leaf_encoding.select_subset(centroids, labels, plant_nr = plant, timestep=time_step, leaf=None)[0]
            after_c = leaf_encoding.select_subset(centroids, labels, plant_nr = plant, timestep=(time_step+1), leaf=None)[0]
            dist_c = leaf_matching.make_dist_matrix(before_c, after_c, centroids, mahalanobis_dist = False, draw=False)

            leaf_nr_before = before_labels[:,3]
            leaf_nr_after = after_labels[:,3]

            for i,leaf in enumerate(np.unique(leaf_nr_before)):
                same = np.argwhere(leaf_nr_after == leaf).flatten()
                different = np.argwhere(leaf_nr_after != leaf).flatten()
                same_leaf_distances = np.append(same_leaf_distances, dist[i,same], axis=0)
                different_leaf_distances = np.append(different_leaf_distances, dist[i,different], axis=0)

                same_leaf_distances_c = np.append(same_leaf_distances_c, dist_c[i,same], axis=0)
                different_leaf_distances_c = np.append(different_leaf_distances_c, dist_c[i,different], axis=0)
                # assignment optimisation
    plot_feature_space_distance_histogram(same_leaf_distances, different_leaf_distances)
    plot_feature_space_distance_histogram(same_leaf_distances_c, different_leaf_distances_c)
    return same_leaf_distances, different_leaf_distances, same_leaf_distances_c, different_leaf_distances_c

if __name__== "__main__":
    # Load data and fit pca
    directory = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D', '_processed', 'pca_input')
    PCAH, test_ds, test_labels = leaf_encoding.get_encoding(train_split=0, random_split=False, directory=directory, standardise=True, location=False, rotation=False, scale=False, as_features=False)

    directory = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D', '_processed', 'transform_log')
    centroids, centroid_labels = leaf_matching.get_location_info(directory) # already sorted

    same_leaf_distances, different_leaf_distances, same_leaf_distances_c, different_leaf_distances_c = fs_distances_between_steps(centroids, centroid_labels, PCAH.training_data, PCAH.training_labels, PCAH, components=22)
