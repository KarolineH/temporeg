import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

from scipy.spatial import distance_matrix
import leaf_encoding
import organ_matching
import packages.pheno4d_util as util

def get_location_info(directory):
    # Load the centroids of all leaf point clouds from the transformation log files
    # leaf centroid saved as the first coordinate
    log_files = os.listdir(directory)
    logs = np.asarray([np.load(os.path.join(directory, leaf)) for leaf in log_files])
    centroids = logs[:,0,:]

    #Extract the labels from the file names, which allows for sorting and looping over the time steps.
    labels = leaf_encoding.get_labels(log_files)
    sorted_centroids, sorted_labels = util.sort_examples(centroids, labels)
    return sorted_centroids, sorted_labels

def plot_centroids(subset1, subset2):
    # Scatterplot to verify that the centroids are correct
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(2, 1, 1, projection='3d')
    scatterplot = ax.scatter(xs=subset1[:,0], ys=subset1[:,1], zs=subset1[:,2])
    scatterplot = ax.scatter(xs=0, ys=0, zs=0, c='black')

    ax2 = fig.add_subplot(2, 1, 2, projection='3d')
    scatterplot = ax2.scatter(xs=subset2[:,0], ys=subset2[:,1], zs=subset2[:,2])
    scatterplot = ax2.scatter(xs=0, ys=0, zs=0, c='black')
    plt.show()

def plot_assignment(before, after, before_labels, after_labels, legible_matches, title=None):
    fig = plt.figure(figsize=(40, 40))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    highest_leaf_nr = np.max((np.max(before_labels[:,-1]), np.max(after_labels[:,-1])))
    cmap = cm.get_cmap('rainbow')
    colours = cmap((range(0,highest_leaf_nr)/highest_leaf_nr))

    origin = ax.scatter(xs=0, ys=0, zs=0, c='black')
    before_centroids = ax.scatter(xs=before[:,0], ys=before[:,1], zs=before[:,2], c=colours[before_labels[:,-1]-1,:], marker='o')
    after_centroids = ax.scatter(xs=after[:,0], ys=after[:,1], zs=after[:,2], c=colours[after_labels[:,-1]-1,:], marker='+')

    for match in legible_matches[:,:,-1]:
        before_point = before[np.where(before_labels[:,-1] == match[0]),:].flatten()
        after_point = after[np.where(after_labels[:,-1] == match[1]),:].flatten()
        if match[0] == match[1]:
            colour = 'blue'
        else:
            colour = 'red'
        pair_line = np.stack((before_point, after_point), axis = 0)
        ax.plot(pair_line[:,0], pair_line[:,1], pair_line[:,2], c=colour)
        # make correct ones blue
        # wrong ones red
    if title is not None:
        plt.title(title)
    plt.show()

# def get_single_match_bonn_method(before, before_labels, after, after_labels):
#     # remove leaves in the before set, that have no match in the after set
#     trimmed_before = np.delete(before, np.where(np.isin(before_labels[:,-1], after_labels[:,-1]) == False), axis = 0)
#     trimmed_before_labels = np.delete(before_labels, np.where(np.isin(before_labels[:,-1], after_labels[:,-1]) == False), axis=0)
#
#     # get the distance matrix for each pre-post leaf pairing
#     dist = distance_matrix(trimmed_before, after)
#
#     # get assignment via Hunagrian Algorithm
#     match, legible_matches = organ_matching.compute_assignment(dist, before_labels, after_labels)
#     return match, legible_matches

# def get_single_match_my_method(before, before_labels, after, after_labels):
#     # remove leaves in the before set, that have no match in the after set
#     trimmed_before = np.delete(before, np.where(np.isin(before_labels[:,-1], after_labels[:,-1]) == False), axis = 0)
#     trimmed_before_labels = np.delete(before_labels, np.where(np.isin(before_labels[:,-1], after_labels[:,-1]) == False), axis=0)
#     #trimmed_before = before
#     #trimmed_before_labels = before_labels
#
#     # get the distance matrix for each pre-post leaf pairing
#     dist = organ_matching.make_dist_matrix(trimmed_before, after, pca, draw=False, components=200)
#     match, legible_matches = organ_matching.compute_assignment(dist, before_labels, after_labels)
#     return match, legible_matches

def get_score_across_dataset(centroids, centroid_labels, outline_data, outline_labels, pca, trim_missing=True, plotting=False):

    # The processed outline set potentially has fewer leaves in it than the original (centroid only) data set
    # Find only the leaves, which are present in both for a fair comparison
    indeces = np.asarray([np.argwhere((centroid_labels == leaf).all(axis=1)) for leaf in outline_labels]).flatten()
    centroids = centroids[indeces,:]
    centroid_labels = centroid_labels[indeces,:]

    # Loop over all pairs of time steps
    bonn_count = 0
    our_count = 0
    total = 0
    for plant in np.unique(outline_labels[:,0]): # for each plant
        for time_step in range(np.unique(outline_labels[:,1]).size-1): # and for each time step available in the processed data

            # select the subsets to compare
            c_before, c_before_labels = leaf_encoding.select_subset(centroids, centroid_labels, plant_nr = plant, timestep=time_step, day=None, leaf=None)
            c_after, c_after_labels = leaf_encoding.select_subset(centroids, centroid_labels, plant_nr = plant, timestep=time_step+1, day=None, leaf=None)
            o_before, o_before_labels = leaf_encoding.select_subset(outline_data, outline_labels, plant_nr = plant, timestep=time_step, day=None, leaf=None)
            o_after, o_after_labels = leaf_encoding.select_subset(outline_data, outline_labels, plant_nr = plant, timestep=time_step+1, day=None, leaf=None)

            # Remove leaves that exist only in the LATER scan. The method assumes that each leaf will be present in the next time step
            # New emerging leaves are permitted, but vanishing leaves are not expected
            if trim_missing:
                c_before = np.delete(c_before, np.where(np.isin(o_before_labels[:,-1], o_after_labels[:,-1]) == False), axis = 0)
                c_before_labels = np.delete(c_before_labels, np.where(np.isin(o_before_labels[:,-1], o_after_labels[:,-1]) == False), axis=0)
                o_before = np.delete(o_before, np.where(np.isin(o_before_labels[:,-1], o_after_labels[:,-1]) == False), axis = 0)
                o_before_labels = np.delete(o_before_labels, np.where(np.isin(o_before_labels[:,-1], o_after_labels[:,-1]) == False), axis=0)

            # Now perform the matching step using both methods

            centroid_dist = distance_matrix(c_before, c_after)
            outline_dist = organ_matching.make_dist_matrix(o_before, o_after, pca, draw=False, components=200)

            c_matches = organ_matching.compute_assignment(centroid_dist, c_before_labels, c_after_labels)[1]
            o_matches = organ_matching.compute_assignment(outline_dist, o_before_labels, o_after_labels)[1]

            # if plotting:
            #     plant_id = str(after_labels[0,:-1])
            #     plot_assignment(before, after, before_labels, after_labels, legible_matches, title=plant_id)
            # # count correct assignments (true positives)

            for pair in range(len(c_matches)):
                total += 1
                if c_matches[pair][0,-1] == c_matches[pair][1,-1]:
                    bonn_count += 1
                if o_matches[pair][0,-1] == o_matches[pair][1,-1]:
                    our_count += 1

    print(f'Bonn (centroid) method: {bonn_count}')
    print(f'Our (outline) method: {our_count}')
    print(f'total: {total}')

def testing_pipeline():
    # Load data needed for Bonn method
    directory = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D', '_processed', 'transform_log')
    centroids, centroid_labels = get_location_info(directory) # already sorted

    # Load data needed for my method
    directory = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D', '_processed', 'pca_input')
    train_ds, test_ds, train_labels, test_labels, pca, transformed = leaf_encoding.get_encoding(train_split=0, dir=dir, location=True, rotation=False, scale=True, as_features=False)

    outline_data, outline_labels = util.sort_examples(train_ds, train_labels)

    get_score_across_dataset(centroids, centroid_labels, outline_data, outline_labels, pca, trim_missing=True, plotting=False)

if __name__== "__main__":

    testing_pipeline()
