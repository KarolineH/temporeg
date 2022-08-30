import os
import open3d as o3d
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import visualise
from sklearn.manifold import TSNE
import copy
import matplotlib.path as path
import matplotlib.cm as cm
from sklearn.cluster import KMeans
import leaf_matching
import packages.pheno4d_util as util
from packages.LeafSurfaceReconstruction import helper_functions

'''
Data pre-processing and utilities
'''

def load_inputs(directory):
    leaves = os.listdir(directory)
    data = np.asarray([np.load(os.path.join(directory, leaf)) for leaf in leaves])
    return data, leaves

def get_labels(file_names):
    '''
    IDs are Plant, Timestep, Day number, and Leaf number
    Note that Leaf Numbers are only consistent within the same plant
    Day number refers to the actual day after recording started
    Timestep is the continuous count of available timesteps loaded (for example when excluding non-labelled steps)
    '''
    ids = np.asarray([[''.join([letter for letter in word if letter.isnumeric()]) for word in name.split('_')[:4]]for name in file_names], dtype='int')
    return ids

def add_scale_location_rotation_as_features(data, labels, loc=False, rot=False, sc=False):
    # Get the required information, but get the normalised outlines (no global location, orientation, or scale retained. Only shape.)
    normalised_data, normalised_labels, location_info, rotation_info, scale_info  = add_scale_location_rotation_full(data, labels, location=False, rotation=False, scale=False)
    feature_vector = normalised_data.reshape(normalised_data.shape[0], normalised_data.shape[1]*normalised_data.shape[2])

    if loc:
        feature_vector = np.append(feature_vector, location_info, axis = 1)
    if rot:
        unrolled_rotations = rotation_info.reshape(rotation_info.shape[0], rotation_info.shape[1]*rotation_info.shape[2])
        feature_vector = np.append(feature_vector, unrolled_rotations, axis = 1)
    if sc:
        stacked_scales = scale_info.reshape(scale_info.shape[0], 1)
        feature_vector = np.append(feature_vector, stacked_scales, axis = 1)

    return feature_vector, normalised_labels

def add_scale_location_rotation_full(data, labels, location=False, rotation=False, scale=False):
    # the loaded data from pca_inputs are in leaf centroid coordinate frame
    # which aligned axes (along the largest spread), and not normalised by outline length

    sorted_data, sorted_labels = util.sort_examples(data, labels)  # has scale, but no location or rotation info

    # To rotate back to original recording frame, load previously saved information about leaf axes in the global reference frame
    directory = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D', '_processed', 'transform_log')
    log_files = os.listdir(directory)
    logs = np.asarray([np.load(os.path.join(directory, leaf)) for leaf in log_files])
    axes = logs[:,1:,:]
    log_labels = get_labels(log_files)
    sorted_axes, sorted_axes_labels = util.sort_examples(axes, log_labels)
    # find the same leaves in both data sets
    indeces = np.asarray([np.argwhere((sorted_axes_labels == leaf).all(axis=1)) for leaf in sorted_labels]).flatten()
    original_target_axes = sorted_axes[indeces,:,:]

    if rotation:
        #rotate back
        rotated_outlines = []
        for i,leaf_rotation in enumerate(original_target_axes):
            # Get the zero axes (the coordinate frame at recording time) w.r.t the leaf's specific aligned coordinate system
            cloud_rotated = helper_functions.transform_axis_pointcloud(sorted_data[i], leaf_rotation.T[0], leaf_rotation.T[1], leaf_rotation.T[2])
            rotated_outlines.append(cloud_rotated)
            # Adapted from https://stackoverflow.com/questions/55082928/change-of-basis-in-numpy
            # vec_new = np.linalg.inv(original_axes[0].T).dot(vec_old)
        rotated_data = np.asarray(rotated_outlines)
    else:
        rotated_data = sorted_data

    # In order to normalise leaf scaling, we need to find the outline lengths
    # scaling happens in centroid reference frame, so around the centroid. Doing this in global frame distorts global locations!

    scales = []
    normalised_loops = []
    for loop in rotated_data:
        shifted_loop = np.append(loop, [loop[0]], axis=0)
        shifted_loop = np.delete(shifted_loop, (0), axis=0)

        edge_vectors = shifted_loop-loop # vectors from point to point along the loop
        vector_length = np.linalg.norm(edge_vectors, axis=1)
        cumulative_edges = np.cumsum(vector_length) #get the cumulative distance from start to each point
        total_length = cumulative_edges[-1]
        scales.append(total_length)
        # Normalise by total outline length
        if not scale:
            normalised_points = loop/total_length
            normalised_loops.append(normalised_points)
    scales = np.asarray(scales)
    if not scale:
        scaled_data = np.asarray(normalised_loops)
    else:
        scaled_data=rotated_data


    # In order to shift levaes back from leaf centroid to the plant emergence point as origin
    # we need to load the displacement from earlier saved files
    data_wrt_centroid = scaled_data # at this point the data is already expressed w.r.t. the leaf centroid

    # get the leaf centroid info with respect to the plant emergence point
    directory = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D', '_processed', 'transform_log')
    locations, location_labels = leaf_matching.get_location_info(directory)
    sorted_centroids, sorted_centroid_labels = util.sort_examples(locations, location_labels)
    # find the same leaves in both data sets
    indeces = np.asarray([np.argwhere((sorted_centroid_labels == leaf).all(axis=1)) for leaf in sorted_labels]).flatten()
    # shift the leaf point clouds back to their centroid in global frame
    shift_vectors = sorted_centroids[indeces,:]
    #shifted_labels = sorted_centroid_labels[indeces,:] #only for verification
    # translate back to plant emergence point as origin

    if location:
        located_data = np.asarray([scaled_data[i] + vector for i,vector in enumerate(shift_vectors)])
    else:
        located_data = scaled_data

    # Lastly, translate to outline starting point as origin. But only if no location is wanted
    if not location:
        offsets_from_centroid = located_data[:,0,:]
        out_data = np.asarray([loop - loop[0,:] for loop in located_data])
    else:
        out_data = located_data

    return out_data, sorted_labels, shift_vectors, original_target_axes, scales

def reshape_coordinates_and_additional_features(data, nr_coordinates=500):
    # data can be 1D feature vector or 2D matrix of examples x features
    if len(data.shape) > 1:
        trimmed = data[:,:nr_coordinates*3]
        stacked = trimmed.reshape(trimmed.shape[0],nr_coordinates,-1)
        additional_features = data[:,nr_coordinates*3:]
    else:
        trimmed = data[:nr_coordinates*3]
        stacked = trimmed.reshape(nr_coordinates,-1)
        additional_features = data[nr_coordinates*3:]
    return stacked,additional_features

def split_dataset(data, labels, split = 0.2, random=True):

    ''' splits of a specified % of the input leaves for testing only. The PCA will not be trained on this data.'''

    total_nr_examples = data.shape[0]
    nr_test_examples = int(np.floor(split * total_nr_examples))
    if random:
        indeces = np.random.choice(range(total_nr_examples), size=nr_test_examples, replace=False)
        test_data = data[indeces,:,:]
        test_labels = labels[indeces,:]
        train_data = np.delete(data, indeces, axis=0)
        train_labels = np.delete(labels, indeces, axis=0)
    else:
        indeces = np.arange(0, nr_test_examples)
        test_data = data[indeces,:,:]
        test_labels = labels[indeces,:]
        train_data = np.delete(data, indeces, axis=0)
        train_labels = np.delete(labels, indeces, axis=0)
    return train_data, test_data, train_labels, test_labels

def select_subset(data, labels, plant_nr=None, timestep=None, day=None, leaf=None):
    subset = copy.deepcopy(data)
    sublabels = copy.deepcopy(labels)

    if plant_nr is not None:
        subset = subset[np.argwhere(sublabels[:,0]==plant_nr).flatten(),:]
        sublabels = sublabels[np.argwhere(sublabels[:,0]==plant_nr).flatten(),:]
    if timestep is not None:
        subset = subset[np.argwhere(sublabels[:,1]==timestep).flatten(),:]
        sublabels = sublabels[np.argwhere(sublabels[:,1]==timestep).flatten(),:]
    if day is not None:
        subset = subset[np.argwhere(sublabels[:,2]==day).flatten(),:]
        sublabels = sublabels[np.argwhere(sublabels[:,2]==day).flatten(),:]
    if leaf is not None:
        subset = subset[np.argwhere(sublabels[:,3]==leaf).flatten(),:]
        sublabels = sublabels[np.argwhere(sublabels[:,3]==leaf).flatten(),:]
    return subset, sublabels

'''
PCA Encoding
'''

class PCA_Handler:

    def __init__(self, input_data, input_labels, standard=True):
        self.training_data = input_data # already sorted
        self.training_labels = input_labels # already sorted
        self.standard_flag = standard
        self.default_component_nr = 50
        self.fit_scaler()
        self.fit_pca()

    def fit_scaler(self):
        self.scaler = StandardScaler()
        self.scaler.fit(self.training_data)
        self.standard_training_data = self.scaler.transform(self.training_data, copy=True)
        return

    def fit_pca(self):
        self.pca = PCA()
        if self.standard_flag:
            self.pca.fit(self.standard_training_data)
            self.transformed_training_data = self.pca.transform(self.standard_training_data)
        else:
            self.pca.fit(self.training_data)
            self.transformed_training_data = self.pca.transform(self.training_data)
        self.max_components = self.pca.components_.shape[0]
        return

    def compress(self, data, nr_components=None):
        all_components = self.pca.components_
        if nr_components is None:
            nr_components = self.default_component_nr

        if self.standard_flag:
            # Standardise
            if len(data.shape) == 1:
                data = data.reshape(1, -1) # necessary if only one feature vectore is passed instead of a matrix of multiple examples
            data = self.scaler.transform(data)

        transformed = np.dot((data - self.pca.mean_), self.pca.components_[:nr_components].T)
        #weights = all_components @ (data - self.pca.mean_).T # this is the transpose of the above
        # Preivously I have gotten the transpose of the sklearn implementation
        return transformed

    def decompress(self, weights, nr_components=None):
        all_components = self.pca.components_
        if nr_components is None:
            nr_components = self.default_component_nr

        projection = np.dot(weights, self.pca.components_[:nr_components]) + self.pca.mean_
        # projection = weights.T @ all_components[:nr_components] + self.pca.mean_

        if self.standard_flag:
            if len(projection.shape) == 1:
                projection = projection.reshape(1, -1) # necessary if only one feature vectore is passed instead of a matrix of multiple examples
            output_data = self.scaler.inverse_transform(projection)
        else:
            output_data = projection
        return output_data

def get_encoding(train_split=0, random_split=True, directory=None, standardise=True, location=False, rotation=False, scale=False, as_features=False):
    if directory is None:
        directory = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D', '_processed', 'pca_input')
    data, names = load_inputs(directory) # at this stage they are in local frame, with centroid as origin, but the scale is still not normalised
    labels = get_labels(names)
    data, labels = util.sort_examples(data, labels) #sort

    train_ds, test_ds, train_labels, test_labels = split_dataset(data, labels, split=train_split, random=random_split)

    if as_features:
        input_data, train_labels = add_scale_location_rotation_as_features(train_ds, train_labels, loc=location, rot=rotation, sc=scale)
        if test_ds.size > 0:
            test_ds, test_labels =  add_scale_location_rotation_as_features(test_ds, test_labels, loc=location, rot=rotation, sc=scale)
    else:
        train_ds, train_labels, location_info, rotation_info, scale_info = add_scale_location_rotation_full(train_ds, train_labels, location, rotation, scale)
        if test_ds.size > 0:
            test_ds, test_labels, test_location_info, test_rotation_info, test_scale_info = add_scale_location_rotation_full(test_ds, test_labels, location, rotation, scale)
        # Flatten training set to use for PCA
        input_data = train_ds.reshape(train_ds.shape[0], train_ds.shape[1]*train_ds.shape[2])

    input_data, input_labels = util.sort_examples(input_data, train_labels) #sort
    PCAH = PCA_Handler(input_data, input_labels, standard=standardise)
    return PCAH, test_ds, test_labels

'''
TESTING and plotting
'''

def plot_explained_variance(PCAH):
    print(f"Using {PCAH.pca.components_.shape[0]} components")
    print(f"Explained variance ratio: {sum(PCAH.pca.explained_variance_ratio_)}")
    plt.plot(np.cumsum(PCAH.pca.explained_variance_ratio_[:100]))
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Number of principal components')
    plt.show()

def perform_single_reprojection(query, PCAH, components=None, draw = False):
    '''
    Takes one single leaf outline (point cloud) at a time
    Encodes and decode the outline, prints the reprojection error
    and draws the input and reprojection side-by-side
    '''

    weights = PCAH.compress(query, components)
    reprojection = PCAH.decompress(weights, components)

    single_error = query-reprojection # distance for each individual feature
    mean_error = np.mean(single_error)
    print(f"Mean reprojection error at {components} components : {mean_error}")

    if draw:
        stacked_query, query_additional_features = reshape_coordinates_and_additional_features(query, nr_coordinates=500)
        stacked_reprojection, reprojection_additional_features = reshape_coordinates_and_additional_features(reprojection, nr_coordinates=500)

        print(f'Aditional features given {query_additional_features} \n vs. reprojected {reprojection_additional_features}')
        cloud1 = visualise.array_to_o3d_pointcloud(np.squeeze(stacked_query))
        cloud2 = visualise.array_to_o3d_pointcloud(np.squeeze(stacked_reprojection))
        visualise.draw([cloud1, cloud2], 'test', offset = True )
    return

def test_reprojection_loss(query, PCAH):
    '''
    Performs encoding and reprojection of a dataset
    with 1 - n components
    plots the reprojection loss by number of used components
    '''
    max_components = PCAH.pca.components_.shape[0]
    mean_dist_errors = []
    all_eigenleaves = PCAH.pca.components_
    RMSEs = []
    for i in range(max_components, 0, -1):

        weights = PCAH.compress(query, i)
        reprojection = PCAH.decompress(weights, i)

        single_error = query-reprojection # (nr_examples x nr_features) distance for each individual feature, coordinates are unrolled
        single_RMSEs=np.sqrt((single_error**2).mean(axis=-1)) # One RMSE for each example
        mRMSE = np.mean(single_RMSEs) # Mean RMSE across all examples
        mean_error = abs(single_error).mean(axis=-1).mean() # mean error across all examples and all features
        mean_dist_errors.append(mean_error)
        RMSEs.append(mRMSE)

    plt.plot(range(max_components, 0, -1), mean_dist_errors)
    plt.ylabel('Mean absolute error across all examples and all features')
    plt.xlabel('Number of principal components')
    plt.show()

    plt.plot(range(max_components, 0, -1), RMSEs)
    plt.ylabel('Mean RMSE across all examples')
    plt.xlabel('Number of principal components')
    plt.show()

def plot_3_components(data, PCAH, labels=None):
    '''
    Transforms a data set to feature space and plots
    the data's first three components
    '''
    coordinates, additional_features = reshape_coordinates_and_additional_features(data, nr_coordinates=500)
    markers = [path.Path(leaf[:,:2]) for leaf in coordinates]
    weights = PCAH.compress(data, PCAH.max_components)

    # Plot with regular markers using labels as colours
    if labels is not None:
        ax = plt.figure(figsize=(16,10)).gca(projection='3d')
        scatterplot = ax.scatter(xs=weights[:,0], ys=weights[:,1], zs=weights[:,2],c=labels[:data.shape[0],3], cmap='rainbow')
        ax.set_xlabel('first component')
        ax.set_ylabel('second component')
        ax.set_zlabel('third component')
        legend1 = ax.legend(*scatterplot.legend_elements(), title="Classes")
        ax.add_artist(legend1)
        plt.show()

    # Plot using 2D projected leaf outline as markers, but no colours
    ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    for _m, _x, _y, _z in zip(markers, weights[:,0], weights[:,1], weights[:,2]):
        scatterplot = ax.scatter(xs=_x, ys=_y, zs=_z,  marker=_m, s=100, c='#1f77b4')
    plt.tight_layout()
    ax.set_xlabel('first component')
    ax.set_ylabel('second component')
    ax.set_zlabel('third component')
    legend1 = ax.legend(*scatterplot.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    plt.show()

def plot_2_components(data, PCAH, labels=None, components=[0,1]):
    '''
    Transforms a data set to feature space and plots
    the data's first two components
    '''
    coordinates, additional_features = reshape_coordinates_and_additional_features(data, nr_coordinates=500)
    markers = [path.Path(leaf[:,:2]) for leaf in coordinates]
    weights = PCAH.compress(data, PCAH.max_components)

    # Plot with regular markers using labels as colours
    fig, ax = plt.subplots()
    if labels is not None:
        scatterplot = ax.scatter(weights[:,components[0]], weights[:,components[1]],c=labels[:data.shape[0],3], cmap='rainbow')
        ax.set_xlabel('component {}'.format(components[0]))
        ax.set_ylabel('component {}'.format(components[1]))
        legend1 = ax.legend(*scatterplot.legend_elements(), title="Classes")
        ax.add_artist(legend1)
        plt.show()

    # Plot using 2D projected leaf outline as markers, but no colours
    fig, ax = plt.subplots()
    for _m, _x, _y in zip(markers, weights[:,components[0]], weights[:,components[1]]):
        scatterplot = ax.scatter(_x, _y, marker=_m, s=500, c='#FFFFFF', edgecolors='#1f77b4')
    plt.tight_layout()
    ax.set_xlabel('component {}'.format(components[0]))
    ax.set_ylabel('component {}'.format(components[1]))
    legend1 = ax.legend(*scatterplot.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    plt.show()

def new_random_leaf_from_distribution(query, labels, PCAH, components=50, draw=True):
    '''
    Gets the distributions for all principal components
    Randomly samples components from their distributions to create a new plausible random leaf
    '''
    weights = PCAH.compress(query, components)
    # components = 50
    # all_eigenleaves = pca.components_
    # query_weight = compress(query, all_eigenleaves[:components], pca).T #(examples x components)

    ranges = np.asarray([np.min(weights, axis=0),np.max(weights, axis=0)])
    if draw:
        plt.boxplot(weights)
        plt.title("Boxplots of Value distributions for the first 50 components")
        plt.show()
        plt.hist(weights[:,0], 20)
        plt.title("Histogram of feature values on the first principal component")
        plt.show()
        plt.hist(weights[:,1], 20)
        plt.title("Histogram of feature values on the second principal component")
        plt.show()

    # random new leaf
    # sample each feature randomly from the existing feature vectors
    new_leaf = np.asarray([weights[np.random.choice(weights.shape[0]),i] for i in range(weights.shape[1])])
    reprojection = PCAH.decompress(new_leaf, components)

    random_leaf, random_additional_features = reshape_coordinates_and_additional_features(reprojection, nr_coordinates=500)
    mean_leaf, mean_additional_features = reshape_coordinates_and_additional_features(PCAH.pca.mean_, nr_coordinates=500)
    cloud = visualise.array_to_o3d_pointcloud(np.squeeze(random_leaf))
    cloud2 = visualise.array_to_o3d_pointcloud(np.squeeze(mean_leaf))
    if draw:
        visualise.draw2([cloud, cloud2], 'test', offset = True )
    return ranges

def t_sne(data, labels=None, label_meaning=None):
    '''
    Performs tSNE on the given feature vectors, including any additional information given at the end of the vector.
    Intended to use only with coordinates normalised in position, rotation, and scale.
    The markers in the plot do not show location, rotation, or scale information. Only leaf shape.
    '''
    tsne = TSNE()
    X_embedded = tsne.fit_transform(data)
    coordinates, additional_features = reshape_coordinates_and_additional_features(data, nr_coordinates=500)
    markers = [path.Path(leaf[:,:2]) for leaf in coordinates]

    if labels is not None:
        max_color = max(labels)
        cmap = cm.get_cmap('rainbow')
        colours = cmap((labels/max_color))

    fig, ax = plt.subplots()
    ax.set_title('t-SNE embedding of leaf input vectors (coordinates and potential additional information')
    if labels is not None:
        for i, _m, _x, _y in zip(range(len(markers)), markers, X_embedded[:,0], X_embedded[:,1]):
            scatterplot = ax.scatter(_x, _y, marker=_m, s=500, c='#FFFFFF', edgecolors=colours[i].reshape(1,-1))
        #legend1 = ax.legend(*ax.legend_elements(), title="Classes")
        legend1 = ax.legend(np.unique(labels, axis = 0), labelcolor=cmap((np.unique(labels, axis = 0)/max_color)), ncol=2, markerscale=0)
        ax.add_artist(legend1)
        if label_meaning is not None:
            ax.set_title('t-SNE embedding of leaf input vectors (coordinates and potential additional information. Colours represent {}'.format(label_meaning))
    else:
        for _m, _x, _y in zip(markers, X_embedded[:,0], X_embedded[:,1]):
            scatterplot = ax.scatter(_x, _y, marker=_m, s=500, c='#FFFFFF', edgecolors='#1f77b4')
    plt.show()

def pca_then_t_sne(data, PCAH, labels=None, label_meaning=None, nr_components=50):
    '''
    Performs tSNE on the extracted feature components by PCA.
    Intended to use only with coordinates normalised in position, rotation, and scale.
    The markers in the plot do not show location, rotation, or scale information. Only leaf shape.
    '''
    #tsne on the pca extracted feature components
    tsne = TSNE()
    weights = PCAH.compress(data, nr_components)
    X_embedded = tsne.fit_transform(weights.T)
    coordinates, additional_features = reshape_coordinates_and_additional_features(data, nr_coordinates=500)
    markers = [path.Path(leaf[:,:2]) for leaf in coordinates]

    if labels is not None:
        max_color = max(labels)
        cmap = cm.get_cmap('rainbow')
        colours = cmap((labels/max_color))

    fig, ax = plt.subplots()
    ax.set_title('t-SNE embedding of leaf shape feature vectors after PCA')
    if labels is not None:
        for i, _m, _x, _y in zip(range(len(markers)), markers, X_embedded[:,0], X_embedded[:,1]):
            scatterplot = ax.scatter(_x, _y, marker=_m, s=500, c='#FFFFFF', edgecolors=colours[i].reshape(1,-1))
        #legend1 = ax.legend(*ax.legend_elements(), title="Classes")
        legend1 = ax.legend(np.unique(labels, axis = 0), labelcolor=cmap((np.unique(labels, axis = 0)/max_color)), ncol=2, markerscale=0)
        ax.add_artist(legend1)
        if label_meaning is not None:
            ax.set_title('t-SNE embedding of leaf shape feature vectors after PCA. Colours represent {}'.format(label_meaning))
    else:
        for _m, _x, _y in zip(markers, X_embedded[:,0], X_embedded[:,1]):
            scatterplot = ax.scatter(_x, _y, marker=_m, s=500, c='#FFFFFF', edgecolors='#1f77b4')
    plt.show()
    return

def analyse_feature_space_clusters(data, labels, PCAH, nr_clusters=4):
    weights = PCAH.compress(data, PCAH.max_components)
    kmeans = KMeans(n_clusters=nr_clusters, random_state=0).fit(weights)
    plt.scatter(weights[:,0], weights[:,1],c=kmeans.labels_)
    plt.show()

    fig, axs = plt.subplots(3,3)
    stacked_coordinates, add_feat = reshape_coordinates_and_additional_features(data, nr_coordinates=500)
    clusters = [stacked_coordinates[np.argwhere(kmeans.labels_== cluster).flatten(),:] for cluster in range(nr_clusters)]
    for cluster in clusters:
        mins = np.min(cluster, axis=1)
        maxes = np.max(cluster, axis=1)
        ranges = maxes - mins
        axs[0][0].hist(mins[:,0], 50, alpha=0.5)
        axs[0][1].hist(maxes[:,0], 50, alpha=0.5)
        axs[0][2].hist(ranges[:,0], 50, alpha=0.5)
        axs[1][0].hist(mins[:,1], 50, alpha=0.5)
        axs[1][1].hist(maxes[:,1], 50, alpha=0.5)
        axs[1][2].hist(ranges[:,1], 50, alpha=0.5)
        axs[2][0].hist(mins[:,2], 50, alpha=0.5)
        axs[2][1].hist(maxes[:,2], 50, alpha=0.5)
        axs[2][2].hist(ranges[:,2], 50, alpha=0.5)
    axs[0][0].set_title('Min value in x direction')
    axs[0][1].set_title('Max value in x direction')
    axs[0][2].set_title('Leaf width in x direction')
    axs[1][0].set_title('Min value in y direction')
    axs[1][1].set_title('Max value in y direction')
    axs[1][2].set_title('Leaf height in y direction')
    axs[2][0].set_title('Min value in z direction')
    axs[2][1].set_title('Max value in z direction')
    axs[2][2].set_title('Leaf depth in z direction')
    plt.show()

def recreate_artefact(data, PCAH):
    '''
    Recreates a sampling artefact, which makes the same shapes look dissimilar in PCA feautre space,
    if their extracted boundary chains are sampled in opposite direction (clockwise and counter-clockwise)
    '''
    sample = data[1]
    sample_flipped = np.flip(sample, 0)
    weight = PCAH.compress(sample, 50)
    flipped_weight = PCAH.compress(sample_flipped, 50)

    dist_same = leaf_matching.make_fs_dist_matrix(sample, sample, PCAH, mahalanobis_dist = True, draw=False, components=50)
    dist_flipped = leaf_matching.make_fs_dist_matrix(sample, sample_flipped, PCAH, mahalanobis_dist = True, draw=False, components=50)
    print(f'Feature space distance: {dist_flipped}')

if __name__== "__main__":
    directory = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D', '_processed', 'pca_input')

    PCAH, test_ds, test_labels = get_encoding(train_split=0.2, random_split=True, directory=directory, standardise=True, location=False, rotation=False, scale=False, as_features=False)

    # Verify that scaled and re-scaled data is the same as the original
    np.testing.assert_array_almost_equal(PCAH.scaler.inverse_transform(PCAH.scaler.transform(PCAH.training_data[:2])), PCAH.training_data[:2])
    # Verify that the compressed and decompressed data is the same as the original when using ALL components
    np.testing.assert_array_almost_equal(PCAH.training_data[:2], PCAH.decompress(PCAH.compress(PCAH.training_data[:2], PCAH.max_components),PCAH.max_components))
    import pdb; pdb.set_trace()
    #plot_explained_variance(PCAH)
    #perform_single_reprojection(PCAH.training_data[0], PCAH, components=5, draw=True)
    #test_reprojection_loss(PCAH.training_data, PCAH)

    #plot_3_components(PCAH.training_data, PCAH)
    #plot_3_components(PCAH.training_data, PCAH, labels = PCAH.training_labels)
    #plot_2_components(PCAH.training_data, PCAH, labels = PCAH.training_labels, components = [0,1])

    #ranges = new_random_leaf_from_distribution(PCAH.training_data, PCAH.training_labels, PCAH)

    #t_sne(PCAH.training_data, PCAH.training_labels[:,3], label_meaning='leaf number')
    #pca_then_t_sne(PCAH.training_data, PCAH, labels=PCAH.training_labels[:,3], label_meaning='leaf number')
    #t_sne(PCAH.training_data, label_meaning='leaf number')
    #pca_then_t_sne(PCAH.training_data, PCAH, label_meaning='leaf number')

    #single_plant_leaves, single_plant_labels= select_subset(PCAH.training_data, PCAH.training_labels, plant_nr = 6)
    #t_sne(single_plant_leaves, single_plant_labels[:,3], label_meaning='leaf number')
    #pca_then_t_sne(single_plant_leaves, PCAH, single_plant_labels[:,3], label_meaning='leaf number')

    #analyse_feature_space_clusters(PCAH.training_data, PCAH.training_labels, PCAH)
    #recreate_artefact(PCAH.training_data, PCAH)
