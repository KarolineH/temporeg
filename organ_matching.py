import os
import numpy as np
import leaf_encoding
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import leaf_encoding
import visualise

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

def fs_distances_between_steps(data, labels, pca):
    # get the feature space representations for all data
    all_eigenleaves = pca.components_
    nr_components = 50
    query = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
    query_weight = leaf_encoding.compress(query, all_eigenleaves[:nr_components], pca)
    query_weight.T

    # Compare timesteps, pairwise (before and after)
    same_leaf_distances = np.array([])
    different_leaf_distances = np.array([])
    for plant in range(np.max(labels[:,0])+1):
        for time_step in range(np.max(labels[:,1])):
            before, before_labels = leaf_encoding.select_subset(query_weight.T, train_labels, plant_nr = plant, timestep=time_step, leaf=None)
            after, after_labels = leaf_encoding.select_subset(query_weight.T, train_labels, plant_nr = plant, timestep=(time_step+1), leaf=None)
            dist = distance_matrix(before, after)
            leaf_nr_before = before_labels[:,2]
            leaf_nr_after = after_labels[:,2]

            for i,leaf in enumerate(np.unique(leaf_nr_before)):
                m = np.argwhere(leaf_nr_after == leaf).flatten()
                k = np.argwhere(leaf_nr_after != leaf).flatten()
                same_leaf_distances = np.append(same_leaf_distances, dist[i,m], axis=0)
                different_leaf_distances = np.append(different_leaf_distances, dist[i,k], axis=0)

                # assignment optimisation
    return same_leaf_distances, different_leaf_distances

def plot_heatmap(data, show_values=False):
    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap='hot')
    # Loop over matrix to add numbers in the boxes
    if show_values:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                text = ax.text(j, i, str(round(data[i, j], 2)), ha="center", va="center", color="w")
    fig.tight_layout()
    plt.show()

def vis_compare_fs_distance(data, labels, pca):
    all_eigenleaves = pca.components_
    nr_components = 500

    sorted_data = data[np.lexsort((labels[:,2], labels[:,1],labels[:,0])),:]
    sorted_labels = labels[np.lexsort((labels[:,2], labels[:,1],labels[:,0])),:]

    for plant in np.unique(labels[:,0]):
        for leaf in np.unique(labels[:,2]):
            subset, sub_labels = leaf_encoding.select_subset(sorted_data, sorted_labels, plant_nr = plant, leaf=leaf)
            subset2, sub_labels2 = leaf_encoding.select_subset(sorted_data, sorted_labels, plant_nr = plant, leaf=leaf+1)
            # plots the pairwise feature space  distance matrix
            query = subset.reshape(subset.shape[0], subset.shape[1]*subset.shape[2])
            query2 = subset2.reshape(subset2.shape[0], subset2.shape[1]*subset2.shape[2])
            query_weight = leaf_encoding.compress(np.concatenate((query, query2), axis = 0), all_eigenleaves[:nr_components], pca)
            dist = distance_matrix(query_weight.T, query_weight.T)
            plot_heatmap(dist, show_values=True)

            #shows the sequence of the same leaf over time
            clouds = [visualise.array_to_o3d_pointcloud(outline) for outline in subset]
            clouds2 = [visualise.array_to_o3d_pointcloud(outline) for outline in subset2]
            visualise.draw2(clouds+clouds2, "leaf number {}".format(leaf), offset=True, labels=sub_labels[:,1])

def fs_distance_matrix(data, labels, pca, components=None):
    all_eigenleaves = pca.components_
    if components is None:
        components = 500
    query = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
    query_weight = leaf_encoding.compress(query, all_eigenleaves[:components], pca)
    dist = distance_matrix(query_weight.T, query_weight.T)
    plot_heatmap(dist)
    return dist

if __name__== "__main__":
    # Load data and fit pca
    directory = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D', '_processed', 'pca_input')
    train_ds, test_ds, train_labels, test_labels, pca, transformed = leaf_encoding.get_encoding(train_split=0, dir=directory)

    # sort
    labels = train_labels[np.lexsort((train_labels[:,2], train_labels[:,1],train_labels[:,0])),:]
    data = train_ds[np.lexsort((train_labels[:,2], train_labels[:,1],train_labels[:,0])),:]

    #vis_compare_fs_distance(data, labels, pca)
    dist = fs_distance_matrix(data, labels, pca)
    import pdb; pdb.set_trace()

    same_leaf_distances, different_leaf_distances = fs_distances_between_steps(data, labels, pca)
    plot_feature_space_distance_histogram(same_leaf_distances, different_leaf_distances)
    import pdb; pdb.set_trace()
