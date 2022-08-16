import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
import leaf_encoding
import packages.pheno4d_util as util

def get_location_info(directory):
    '''
    Load the centroids of all leaf point clouds from the transformation log files
    leaf centroid saved as the first coordinate
    '''
    log_files = os.listdir(directory)
    logs = np.asarray([np.load(os.path.join(directory, leaf)) for leaf in log_files])
    centroids = logs[:,0,:]

    #Extract the labels from the file names, which allows for sorting and looping over the time steps.
    labels = leaf_encoding.get_labels(log_files)
    sorted_centroids, sorted_labels = util.sort_examples(centroids, labels)
    return sorted_centroids, sorted_labels

def make_dist_matrix(data1, data2, pca, draw=True, components=50):
    '''
    Calculate the pairwise distance matrix between two data sets.
    Usually to calculate the feature space distance between the same leaves before and after a time skip.
    Can also plot the matrix as a heatmap.
    '''
    # plots the pairwise feature space  distance matrix
    all_eigenleaves = pca.components_
    query_weight1 = leaf_encoding.compress(data1, all_eigenleaves[:components], pca)
    query_weight2 = leaf_encoding.compress(data2, all_eigenleaves[:components], pca)
    dist = distance_matrix(query_weight1.T, query_weight2.T)
    if draw:
        plot_heatmap(dist, show_values=True)
    return dist

def compute_assignment(distance_matrix, label_set_1, label_set_2):
    '''
    Uses the Hungarian method or Munkres algorithm to find the best assignment or minimum weight matchin, given a cost matrix.
    '''
    assignment = linear_sum_assignment(distance_matrix)
    match = (label_set_1[assignment[0]], label_set_2[assignment[1]])
    return match, np.array(list(zip(match[0],match[1])))

def get_score_across_dataset(centroids, centroid_labels, outline_data, outline_labels, pca, trim_missing=True, plotting=False):
    '''
    Iterate over the whole given data set, calculating feature space distance and best assignment of leaves for each time skip
    using both our method (outline shape) and the Bonn method (centroid location).
    When trim_missing is active, leaves that are present before but not later are removed from the assignment process.
    Returns scores of both methods and can also plot the individual matching results of the Bonn method.
    '''
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
            outline_dist = make_dist_matrix(o_before, o_after, pca, draw=False, components=200)
            c_matches = compute_assignment(centroid_dist, c_before_labels, c_after_labels)[1]
            o_matches = compute_assignment(outline_dist, o_before_labels, o_after_labels)[1]

            if plotting:
                plant_id = str(c_after_labels[0,:-1])
                plot_bonn_assignment(c_before, c_after, c_before_labels, c_after_labels, c_matches, title=plant_id)

            # count correct assignments (true positives)
            for pair in range(len(c_matches)):
                total += 1
                if c_matches[pair][0,-1] == c_matches[pair][1,-1]:
                    bonn_count += 1
                if o_matches[pair][0,-1] == o_matches[pair][1,-1]:
                    our_count += 1

    print(f'Bonn (centroid) method: {bonn_count}')
    print(f'Our (outline) method: {our_count}')
    print(f'total: {total}')
    return bonn_count, our_count, total

def testing_pipeline(location=False, rotation=False, scale=False, as_features=False):
    # Load data needed for Bonn method
    directory = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D', '_processed', 'transform_log')
    centroids, centroid_labels = get_location_info(directory) # already sorted

    # Load data needed for my method
    directory = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D', '_processed', 'pca_input')
    train_ds, test_ds, train_labels, test_labels, pca, transformed = leaf_encoding.get_encoding(train_split=0, dir=directory, location=location, rotation=rotation, scale=scale, as_features=as_features)

    outline_data, outline_labels = util.sort_examples(train_ds, train_labels)

    get_score_across_dataset(centroids, centroid_labels, outline_data, outline_labels, pca, trim_missing=True, plotting=False)


''' PLOTTING '''

def plot_centroids(subset1, subset2):
    '''
    Scatterplot to verify that the centroids are correct
    '''
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(2, 1, 1, projection='3d')
    scatterplot = ax.scatter(xs=subset1[:,0], ys=subset1[:,1], zs=subset1[:,2])
    scatterplot = ax.scatter(xs=0, ys=0, zs=0, c='black')

    ax2 = fig.add_subplot(2, 1, 2, projection='3d')
    scatterplot = ax2.scatter(xs=subset2[:,0], ys=subset2[:,1], zs=subset2[:,2])
    scatterplot = ax2.scatter(xs=0, ys=0, zs=0, c='black')
    plt.show()

def plot_bonn_assignment(before, after, before_labels, after_labels, legible_matches, title=None):
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

def plot_heatmap(data, labels=None, ax = None, show_values=False):
    '''
    Plots a distance matrix as a heatmap, wich axis ticks, and values displayed in the cells.
    '''
    given_ax = ax
    if given_ax is None:
        fig, ax = plt.subplots()
    im = ax.imshow(data, cmap='hot')
    # Loop over matrix to add numbers in the boxes
    if show_values:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                text = ax.text(j, i, str(round(data[i, j], 3)), ha="center", va="center", color="w")
    if given_ax is None:
        fig.tight_layout()
        plt.show(block=True)
    if labels is not None:
        ax.set_xticks(labels)
        ax.set_yticks(labels)
    return ax



if __name__== "__main__":

    print(':::  No additional information')
    testing_pipeline(location=False, rotation=False, scale=False, as_features=True)

    print('::: + location (as features)')
    testing_pipeline(location=True, rotation=False, scale=False, as_features=True)
    print('::: + rotation (as features)')
    testing_pipeline(location=False, rotation=True, scale=False, as_features=True) # *
    print('::: + scale (as features)')
    testing_pipeline(location=False, rotation=False, scale=True, as_features=True)

    print('::: + location + rotation + scale (as features)')
    testing_pipeline(location=True, rotation=True, scale=True, as_features=True)
    print('::: + location + rotation (as features)')
    testing_pipeline(location=True, rotation=True, scale=False, as_features=True)
    print('::: + location + scale (as features)')
    testing_pipeline(location=True, rotation=False, scale=True, as_features=True)
    print('::: + rotation + scale (as features)')
    testing_pipeline(location=False, rotation=True, scale=True, as_features=True)

    print(':::  No additional information')
    testing_pipeline(location=False, rotation=False, scale=False, as_features=False)

    print('::: + location (intrinsic to coordinates)')
    testing_pipeline(location=True, rotation=False, scale=False, as_features=False)
    print('::: + rotation (intrinsic to coordinates)')
    testing_pipeline(location=False, rotation=True, scale=False, as_features=False)
    print('::: + scale (intrinsic to coordinates)')
    testing_pipeline(location=False, rotation=False, scale=True, as_features=False)

    print('::: + location + rotation + scale (intrinsic to coordinates)')
    testing_pipeline(location=True, rotation=True, scale=True, as_features=False)
    print('::: + location + rotation (intrinsic to coordinates)')
    testing_pipeline(location=True, rotation=True, scale=False, as_features=False)
    print('::: + location + scale (intrinsic to coordinates)')
    testing_pipeline(location=True, rotation=False, scale=True, as_features=False)
    print('::: + rotation + scale (intrinsic to coordinates)')
    testing_pipeline(location=False, rotation=True, scale=True, as_features=False)
