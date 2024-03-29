import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
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

def make_fs_dist_matrix(data1, data2, PCAH, mahalanobis_dist = True, draw=True, components=50):
    '''
    Calculate the pairwise distance matrix between two data sets.
    Usually to calculate the feature space distance between the same leaves before and after a time skip.
    Can also plot the matrix as a heatmap.
    '''
    # plots the pairwise feature space  distance matrix
    query_weight1 = PCAH.compress(data1, components)
    query_weight2 = PCAH.compress(data2, components)

    if mahalanobis_dist:
        # need VI, the inverse covariance matrix for Mahalanobis. It is calculated across the entire training set.
        # By default it would be calculated from the inputs, but only works if nr_inputs > nr_features
        training_compressed = PCAH.compress(PCAH.training_data, components)
        Vi = np.linalg.inv(np.cov(training_compressed.T))
        dist =  cdist(query_weight1,query_weight2,'mahalanobis', VI=Vi)
    else:
        # Euclidean distance
        dist = distance_matrix(query_weight1.T, query_weight2.T)
    if draw:
        plot_heatmap(dist, show_values=True)
    return dist

def make_dist_matrix(data1, data2, training_set, mahalanobis_dist = True, draw=True):
    '''
    Calculate the pairwise distance matrix between two data sets, using Mahalanobis or euclidean distance, without any compression/encoding.
    '''
    if training_set.shape[1]>1:
        if mahalanobis_dist:
            # need VI, the inverse covariance matrix for Mahalanobis. It is calculated across the entire training set.
            # By default it would be calculated from the inputs, but only works if nr_inputs > nr_features
            Vi = np.linalg.inv(np.cov(training_set.T))
            dist =  cdist(data1,data2,'mahalanobis', VI=Vi)
        else:
            dist = distance_matrix(data1, data2)
    else:
        dist = distance_matrix(data1, data2)
    if draw:
        plot_heatmap(dist, show_values=True)
    return dist

def compute_assignment(dist_mat, label_set_1, label_set_2):
    '''
    Uses the Hungarian method or Munkres algorithm to find the best assignment or minimum weight matchin, given a cost matrix.
    '''
    assignment = linear_sum_assignment(dist_mat)
    match = (label_set_1[assignment[0]], label_set_2[assignment[1]])
    return match, np.array(list(zip(match[0],match[1])))

def get_score_across_dataset(centroids, centroid_labels, outline_data, outline_labels, PCAH, components=50, add_inf=None, trim_missing=True, plotting=False, time_gap=1):
    '''
    Iterate over the whole given data set, calculating feature space distance and best assignment of leaves for each time skip
    using both our method (outline shape) and the Bonn method (centroid location).
    When trim_missing is active, leaves that are present before but not later are removed from the assignment process.
    Returns scores of both methods and can also plot the individual matching results of the Bonn method.
    '''

    # First check if additonal info was given and is not empty, else ignore
    if add_inf is not None and add_inf.shape[1] == 0:
        add_inf = None

    # The processed outline set potentially has fewer leaves in it than the original (centroid only) data set
    # Find only the leaves, which are present in both for a fair comparison
    indeces = np.asarray([np.argwhere((centroid_labels == leaf).all(axis=1)) for leaf in outline_labels]).flatten()
    centroids = centroids[indeces,:]
    centroid_labels = centroid_labels[indeces,:]

    # Loop over all pairs of time steps
    # Count all posisble true matches, the actual number of true matches, and 3 types of mistakes for each method
    bonn_counts = [0,0,0,0,0] # true matches, true mistake, open before mistake, open after mistake, open open mistake
    outline_counts = [0,0,0,0,0]
    add_inf_counts = [0,0,0,0,0]
    total_true_pairings = 0

    for plant in np.unique(outline_labels[:,0]): # for each plant
        for time_step in range(0,np.unique(outline_labels[:,1]).size-time_gap,1): # and for each time step available in the processed data

            # select the subsets to compare
            c_before, c_before_labels = leaf_encoding.select_subset(centroids, centroid_labels, plant_nr = plant, timestep=time_step, day=None, leaf=None)
            c_after, c_after_labels = leaf_encoding.select_subset(centroids, centroid_labels, plant_nr = plant, timestep=time_step+time_gap, day=None, leaf=None)
            o_before, o_before_labels = leaf_encoding.select_subset(outline_data, outline_labels, plant_nr = plant, timestep=time_step, day=None, leaf=None)
            o_after, o_after_labels = leaf_encoding.select_subset(outline_data, outline_labels, plant_nr = plant, timestep=time_step+time_gap, day=None, leaf=None)
            if add_inf is not None:
                a_before, a_before_labels = leaf_encoding.select_subset(add_inf, outline_labels, plant_nr = plant, timestep=time_step, day=None, leaf=None)
                a_after, a_after_labels = leaf_encoding.select_subset(add_inf, outline_labels, plant_nr = plant, timestep=time_step+time_gap, day=None, leaf=None)

            # Remove leaves that exist only in the LATER scan. The method assumes that each leaf will be present in the next time step
            # New emerging leaves are permitted, but vanishing leaves are not expected
            if trim_missing:
                c_before = np.delete(c_before, np.where(np.isin(o_before_labels[:,-1], o_after_labels[:,-1]) == False), axis = 0)
                c_before_labels = np.delete(c_before_labels, np.where(np.isin(o_before_labels[:,-1], o_after_labels[:,-1]) == False), axis=0)
                o_before = np.delete(o_before, np.where(np.isin(o_before_labels[:,-1], o_after_labels[:,-1]) == False), axis = 0)
                o_before_labels = np.delete(o_before_labels, np.where(np.isin(o_before_labels[:,-1], o_after_labels[:,-1]) == False), axis=0)
                if add_inf is not None:
                    a_before = np.delete(a_before, np.where(np.isin(o_before_labels[:,-1], o_after_labels[:,-1]) == False), axis = 0)
                    a_before_labels = np.delete(a_before_labels, np.where(np.isin(o_before_labels[:,-1], o_after_labels[:,-1]) == False), axis=0)

            # Now perform the matching step using both methods
            centroid_dist = make_dist_matrix(c_before, c_after, centroids, mahalanobis_dist = False, draw=False)
            outline_dist = make_fs_dist_matrix(o_before, o_after, PCAH, draw=False, components=components)
            if add_inf is not None:
                add_inf_dist = make_dist_matrix(a_before, a_after, add_inf, mahalanobis_dist = True, draw=False)
                a_matches = compute_assignment(add_inf_dist, a_before_labels, a_after_labels)[1]
            else:
                a_matches = None

            c_matches = compute_assignment(centroid_dist, c_before_labels, c_after_labels)[1]
            o_matches = compute_assignment(outline_dist, o_before_labels, o_after_labels)[1]

            if plotting:
                plant_id = str(c_after_labels[0,:-1])
                plot_bonn_assignment(c_before, c_after, c_before_labels, c_after_labels, c_matches, title=plant_id)

            ''' SCORING'''
            # count total possible correct assignments (true positives)
            true_pairs = np.intersect1d(c_before_labels[:,-1],c_after_labels[:,-1])
            total_true_pairings += true_pairs.shape[0]

            only_before = np.setdiff1d(c_before_labels[:,-1],c_after_labels[:,-1])
            only_after = np.setdiff1d(c_after_labels[:,-1],c_before_labels[:,-1])

            for method, scores in zip([c_matches, o_matches, a_matches],[bonn_counts, outline_counts, add_inf_counts]):
                if method is not None:
                    for pair in method:
                        if pair[0,-1] == pair[1,-1]:
                            # A true match is found
                            scores[0] += 1
                        elif pair[0,-1] in true_pairs and pair[1,-1] in true_pairs:
                            # Two leaves from existing pairs have been mistakenly matched
                            scores[1] += 1
                        elif pair[0,-1] in true_pairs or pair[1,-1] in true_pairs:
                            # One leaf from an existing pair was mistakenly matched to an unpaired leaf
                            if pair[0,-1] in only_before:
                                scores[2] += 1
                            elif pair[1,-1] in only_after:
                                scores[3] += 1
                        else:
                            scores[4] += 1

    return bonn_counts, outline_counts, add_inf_counts, total_true_pairings

def testing_pipeline(location=False, rotation=False, scale=False, as_features=False, standardise = True, trim_missing=True, components=50, time_gap=1):
    # Load data needed for Bonn method
    directory = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D', '_processed', 'transform_log')
    centroids, centroid_labels = get_location_info(directory) # already sorted

    # Load data needed for my method
    directory = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D', '_processed', 'pca_input')
    PCAH, test_ds, test_labels = leaf_encoding.get_encoding(train_split=0, directory=directory, standardise=standardise, location=location, rotation=rotation, scale=scale, as_features=as_features)
    if as_features:
        coordinates, additional_info = leaf_encoding.reshape_coordinates_and_additional_features(PCAH.training_data, nr_coordinates=500)
    else:
        additional_info = None

    bonn_count, our_count, add_inf_count, total = get_score_across_dataset(centroids, centroid_labels, PCAH.training_data, PCAH.training_labels, PCAH, components=components, add_inf = additional_info, trim_missing=trim_missing, plotting=False)
    return bonn_count, our_count, add_inf_count, total

def tests():

    conditions = [] # each element is location, rotation, scale, as_features)
    descriptions = [] # Strings for the user to interpret the output by
    i=0

    # First with extra info as features in the input vector
    conditions.append((False, False, False, True))
    descriptions.append(f'Test {i}: Coordinates only, no additional information given')
    i+=1
    conditions.append((True, False, False, True))
    descriptions.append(f'Test {i}: Coordinates + location as features')
    i+=1
    conditions.append((False, True, False, True))
    descriptions.append(f'Test {i}: Coordinates + rotation as features')

    i+=1
    conditions.append((False, False, True, True))
    descriptions.append(f'Test {i}: Coordinates + scale as features')
    i+=1
    conditions.append((True, False, True, True))
    descriptions.append(f'Test {i}: Coordinates + scale + location as features')
    i+=1
    conditions.append((False, True, True, True))
    descriptions.append(f'Test {i}: Coordinates + scale + rotation as features')
    i+=1
    conditions.append((True, True, False, True))
    descriptions.append(f'Test {i}: Coordinates + location + rotation as features')
    i+=1
    conditions.append((True, True, True, True))
    descriptions.append(f'Test {i}: Coordinates + location + rotation + scale as features')
    i+=1

    # Then with extra info given as intrinsic to the outline coordinates
    conditions.append((False, False, False, False))
    descriptions.append(f'Test {i}: Coordinates only, no additional information given')
    i+=1
    conditions.append((True, False, False, False))
    descriptions.append(f'Test {i}: Coordinates + location given intrinsic to outline coordinates ')
    i+=1
    conditions.append((False, True, False, False))
    descriptions.append(f'Test {i}: Coordinates + rotation given intrinsic to outline coordinates ')
    i+=1
    conditions.append((False, False, True, False))
    descriptions.append(f'Test {i}: Coordinates + scale given intrinsic to outline coordinates ')
    i+=1
    conditions.append((True, False, True, False))
    descriptions.append(f'Test {i}: Coordinates + scale + location given intrinsic to outline coordinates ')
    i+=1
    conditions.append((False, True, True, False))
    descriptions.append(f'Test {i}: Coordinates + scale + rotation given intrinsic to outline coordinates ')
    i+=1
    conditions.append((True, True, False, False))
    descriptions.append(f'Test {i}: Coordinates + location + rotation given intrinsic to outline coordinates ')
    i+=1
    conditions.append((True, True, True, False))
    descriptions.append(f'Test {i}: Coordinates + location + rotation + scale given intrinsic to outline coordinates ')
    i+=1

    # Could run this routine with different datasets as well
    our_counts = []
    bonn_counts = []
    add_inf_counts = []
    totals = []
    for test, descr in zip(conditions, descriptions):
        print('Running ' + descr)
        bonn_count, our_count, add_inf_count, total = testing_pipeline(location=test[0], rotation=test[1], scale=test[2], as_features=test[3], standardise = False, trim_missing=False, components=22, time_gap=1)
        our_counts.append(our_count)
        bonn_counts.append(bonn_count)
        add_inf_counts.append(add_inf_count)
        totals.append(total)
    print('Counts represent [true matches, mismatch of two leaves with existing matches, open before mismatch, open after mismatch, open open mismatch]')
    print('Contour method')
    print(np.asarray(our_counts))
    print('Bonn method')
    print(np.asarray(bonn_counts))
    print('Only extra features method')
    print(np.asarray(add_inf_counts))
    print('Total possible true matches')
    print(np.asarray(totals))

    import pdb; pdb.set_trace()

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

    tests()
