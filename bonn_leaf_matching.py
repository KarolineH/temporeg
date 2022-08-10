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
    sorted_data, sorted_labels = util.sort_examples(centroids, labels)
    return sorted_data, sorted_labels

def get_shape_info(directory):
    if directory is None:
        directory = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D', '_processed', 'pca_input')
    train_ds, test_ds, train_labels, test_labels, pca, transformed = leaf_encoding.get_encoding(train_split=0, dir=directory)


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


def get_single_match_bonn_method(before, before_labels, after, after_labels):
    # remove leaves in the before set, that have no match in the after set
    trimmed_before = np.delete(before, np.where(np.isin(before_labels[:,-1], after_labels[:,-1]) == False), axis = 0)
    trimmed_before_labels = np.delete(before_labels, np.where(np.isin(before_labels[:,-1], after_labels[:,-1]) == False), axis=0)

    # get the distance matrix for each pre-post leaf pairing
    dist = distance_matrix(trimmed_before, after)

    # get assignment via Hunagrian Algorithm
    match, legible_matches = organ_matching.compute_assignment(dist, before_labels, after_labels)
    return match, legible_matches

def get_single_match_my_method(before, before_labels, after, after_labels):
    # remove leaves in the before set, that have no match in the after set
    trimmed_before = np.delete(before, np.where(np.isin(before_labels[:,-1], after_labels[:,-1]) == False), axis = 0)
    trimmed_before_labels = np.delete(before_labels, np.where(np.isin(before_labels[:,-1], after_labels[:,-1]) == False), axis=0)

    # get the distance matrix for each pre-post leaf pairing
    dist = organ_matching.make_dist_matrix(trimmed_before, after, pca, draw=False, components=200)
    match, legible_matches = organ_matching.compute_assignment(dist, before_labels, after_labels)
    return match, legible_matches

def get_score_across_dataset(sorted_data, sorted_labels, plotting=False, bonn_method=False):
    # Loop over all pairs of time steps, evaluate with the Bonn method, get score
    count = 0
    total = 0
    for plant in np.unique(sorted_labels[:,0]):
        for time_step in range(np.unique(sorted_labels[:,1]).size-1):
            # select the subset to compare
            before, before_labels = leaf_encoding.select_subset(sorted_data, sorted_labels, plant_nr = plant, timestep=time_step, day=None, leaf=None)
            after, after_labels = leaf_encoding.select_subset(sorted_data, sorted_labels, plant_nr = plant, timestep=time_step+1, day=None, leaf=None)

            # perform pairing
            if bonn_method:
                match, legible_matches = get_single_match_bonn_method(before, before_labels, after, after_labels)
            else:
                match, legible_matches = get_single_match_my_method(before, before_labels, after, after_labels)

            if plotting:
                plant_id = str(after_labels[0,:-1])
                plot_assignment(before, after, before_labels, after_labels, legible_matches, title=plant_id)
            # count correct assignments (true positives)
            for pair in legible_matches:
                    total += 1
                    if pair[0,-1] == pair[1,-1]:
                        count += 1
    print(count)
    print(total)

if __name__== "__main__":
    # Load data for bonn method
    directory = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D', '_processed', 'transform_log')
    data, labels = get_location_info(directory)
    get_score_across_dataset(data, labels, plotting=False, bonn_method=True)

    # Load data for my pca method
    directory = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D', '_processed', 'pca_input_maxtest')
    train_ds, test_ds, train_labels, test_labels, pca, transformed = leaf_encoding.get_encoding(train_split=0, dir=directory)
    data, labels = util.sort_examples(train_ds, train_labels)
    get_score_across_dataset(data, labels, plotting=False, bonn_method=False)
