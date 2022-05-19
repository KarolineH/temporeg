import os
import open3d as o3d
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import visualise
from sklearn.manifold import TSNE
import seaborn as sns
import copy
import matplotlib.path as path

def load_inputs(dir):
    leaves = os.listdir(dir)
    data = np.asarray([np.load(os.path.join(dir, leaf)) for leaf in leaves])
    return data, leaves

def get_labels(file_names):
    '''
    IDs are Plant, Timestep, and Leaf
    Note that Leaf Numbers are only consistent within the same plant
    '''
    ids = np.asarray([[''.join([letter for letter in word if letter.isnumeric()]) for word in name.split('_')[:3]]for name in file_names], dtype='int')
    return ids

def standardise_pc_scale(data):
    '''
    Standardises all point clouds to values between 0 and 1
    and keeps record of the single value scaling factor for each point cloud
    '''
    original_shape = data.shape
    mins = np.asarray([np.min(leaf, axis=0) for leaf in data])
    maxes = np.asarray([np.max(leaf, axis=0) for leaf in data])
    ranges = np.asarray([np.ptp(leaf, axis=0) for leaf in data])
    flat_data = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
    scale_factor = np.max(abs(np.concatenate((mins, maxes), axis=1)), axis=1)
    standardised = flat_data / scale_factor[:,None]
    out = standardised.reshape(original_shape)
    return out, scale_factor

def fit_pca(data):
    flat_data = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
    pca = PCA()
    pca.fit(flat_data)
    transformed = pca.fit_transform(flat_data)
    import pdb; pdb.set_trace()
    return pca, transformed

def plot_explained_variance(pca):
    print(f"Using {pca.components_.shape[0]} components")
    print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_)}")
    plt.plot(np.cumsum(pca.explained_variance_ratio_[:100]))
    plt.show()

def split_dataset(data, labels, split = 0.2):
    total_nr_examples = data.shape[0]
    nr_test_examples = int(np.floor(split * total_nr_examples))
    indeces = np.random.choice(range(total_nr_examples), size=nr_test_examples, replace=False)
    test_data = data[indeces,:,:]
    test_labels = labels[indeces,:]
    train_data = np.delete(data, indeces, axis=0)
    train_labels = np.delete(labels, indeces, axis=0)
    return train_data, test_data, train_labels, test_labels

def perform_single_reprojection(sample, pca, components=None, draw = False):
    '''
    Takes one single leaf outline (point cloud) at a time
    Encodes and decode the outline, prints the reprojection error
    and draws the input and reprojection side-by-side
    '''
    if components is None:
        components = pca.components_.shape[0]

    original_shape = sample.shape
    query = sample.reshape(sample.shape[0]*sample.shape[1])
    all_eigenleaves = pca.components_
    query_weight = compress(query, all_eigenleaves[:components], pca)
    reprojection = decompress(query_weight, all_eigenleaves[:components], pca)
    unrolled = reprojection.reshape(original_shape) # restack
    euclidean_dist_error = np.linalg.norm(sample-unrolled, axis = 1) # euclidean distance between each point pair
    mean_error = euclidean_dist_error.mean()
    print(f"Mean reprojection error at {components} components : {mean_error}")

    if draw:
        cloud1 = visualise.array_to_o3d_pointcloud(sample)
        cloud2 = visualise.array_to_o3d_pointcloud(unrolled)
        visualise.draw([cloud1, cloud2], 'test', offset = True )
    return

def test_reprojection_loss(sample, pca):
    '''
    Performs encoding and reprojection of a dataset
    with 1 - n components
    plots the reprojection loss by number of used components
    '''
    original_shape = sample.shape
    max_components = pca.components_.shape[0]
    mean_dist_errors = []
    all_eigenleaves = pca.components_
    query = sample.reshape(sample.shape[0], sample.shape[1]*sample.shape[2])
    RMSEs = []
    for i in range(max_components, 0, -1):
        query_weight = compress(query, all_eigenleaves[:i], pca)
        reprojection = decompress(query_weight, all_eigenleaves[:i], pca)
        unrolled = reprojection.reshape(original_shape) # restack
        euclidean_dist_error = np.linalg.norm(sample-unrolled, axis = -1) # euclidean distance between each point pair
        RMSE=np.mean(np.sqrt((euclidean_dist_error**2).mean(axis=-1)))
        mean_error = euclidean_dist_error.mean(axis=-1).mean()
        mean_dist_errors.append(mean_error)
        RMSEs.append(RMSE)

    plt.plot(range(max_components, 0, -1), mean_dist_errors)
    plt.ylabel('Mean euclidean distance error')
    plt.xlabel('Number of principal components')
    plt.show()

    plt.plot(range(max_components, 0, -1), RMSEs)
    plt.ylabel('Mean RMSE')
    plt.xlabel('Number of principal components')
    plt.show()

def plot_3_components(data, pca, labels=None):
    '''
    Transforms a data set to feature space and plots
    the data's first three components
    '''
    flat_data = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
    transformed = pca.transform(flat_data)
    markers = [path.Path(leaf[:,:2]) for leaf in data]

    # Plot with regular markers using labels as colours
    ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    if labels is not None:
        scatterplot = ax.scatter(xs=transformed[:,0], ys=transformed[:,1], zs=transformed[:,2],c=labels[:flat_data.shape[0],2], cmap='rainbow')
        ax.set_xlabel('first component')
        ax.set_ylabel('second component')
        ax.set_zlabel('third component')
        legend1 = ax.legend(*scatterplot.legend_elements(), title="Classes")
        ax.add_artist(legend1)
        plt.show()

    # Plot using 2D projected leaf outline as markers, but no colours
    ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    for _m, _x, _y, _z in zip(markers, transformed[:,0], transformed[:,1], transformed[:,2]):
        scatterplot = ax.scatter(xs=_x, ys=_y, zs=_z,  marker=_m, s=100, c='#1f77b4')
    plt.tight_layout()
    ax.set_xlabel('first component')
    ax.set_ylabel('second component')
    ax.set_zlabel('third component')
    legend1 = ax.legend(*scatterplot.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    plt.show()

def plot_2_components(data, pca, labels=None):
    '''
    Transforms a data set to feature space and plots
    the data's first two components
    '''
    flat_data = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
    transformed = pca.transform(flat_data)
    markers = [path.Path(leaf[:,:2]) for leaf in data]

    # Plot with regular markers using labels as colours
    fig, ax = plt.subplots()
    if labels is not None:
        scatterplot = ax.scatter(transformed[:,0], transformed[:,1],c=labels[:flat_data.shape[0],2], cmap='rainbow')
        ax.set_xlabel('first component')
        ax.set_ylabel('second component')
        legend1 = ax.legend(*scatterplot.legend_elements(), title="Classes")
        ax.add_artist(legend1)
        plt.show()

    # Plot using 2D projected leaf outline as markers, but no colours
    fig, ax = plt.subplots()
    for _m, _x, _y in zip(markers, transformed[:,0], transformed[:,1]):
        scatterplot = ax.scatter(_x, _y, marker=_m, s=500, c='#FFFFFF', edgecolors='#1f77b4')
    plt.tight_layout()
    ax.set_xlabel('first component')
    ax.set_ylabel('second component')
    legend1 = ax.legend(*scatterplot.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    plt.show()

def new_random_leaf(data, labels, pca):
    '''
    Gets the distributions for all principal components
    Randomly samples components from their distributions to create a new plausible random leaf
    '''
    original_shape = data.shape
    query = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
    all_eigenleaves = pca.components_
    query_weight = compress(query, all_eigenleaves[:50], pca).T #(examples x components)

    plt.boxplot(query_weight)
    plt.show()

    # random new leaf
    # sample each feature randomly from the existing feature vectors
    new_leaf = np.asarray([query_weight[np.random.choice(query_weight.shape[0]),i] for i in range(query_weight.shape[1])])
    averages = np.mean(query_weight, axis = 0)
    reprojection = decompress(new_leaf, all_eigenleaves[:50], pca)
    average_reprojection = decompress(averages, all_eigenleaves[:50], pca)
    random_leaf = reprojection.reshape(1,original_shape[1],original_shape[2])
    average_leaf = average_reprojection.reshape(1,original_shape[1],original_shape[2])
    cloud = visualise.array_to_o3d_pointcloud(random_leaf[0])
    cloud2 = visualise.array_to_o3d_pointcloud(average_leaf[0])
    cloud3 = visualise.array_to_o3d_pointcloud(data[0])
    visualise.draw2([cloud3, cloud2], 'test', offset = True )

    import pdb; pdb.set_trace()

def t_sne(data, labels):
    # tsne on just coordinates
    tsne = TSNE()
    flat_data = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
    X_embedded = tsne.fit_transform(flat_data)
    markers = [path.Path(leaf[:,:2]) for leaf in data]

    fig, ax = plt.subplots()
    for _m, _x, _y in zip(markers, X_embedded[:,0], X_embedded[:,1]):
        scatterplot = ax.scatter(_x, _y, marker=_m, s=500, c='#FFFFFF', edgecolors='#1f77b4')
    import pdb; pdb.set_trace()

    #tsne on the pca extracted feature components
    all_eigenleaves = pca.components_
    nr_components = 50
    query_weight = compress(flat_data, all_eigenleaves[:nr_components], pca)
    query_weight.T
    tsne = TSNE()
    X_embedded = tsne.fit_transform(query_weight.T)
    sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=labels[:,2], legend='auto', palette=palette)
    plt.show()
    import pdb; pdb.set_trace()

def select_subset(data, labels, plant_nr=None, timestep=None, leaf=None):
    subset = copy.deepcopy(data)
    sublabels = copy.deepcopy(labels)

    if plant_nr is not None:
        subset = subset[np.argwhere(sublabels[:,0]==plant_nr).flatten(),:]
        sublabels = sublabels[np.argwhere(sublabels[:,0]==plant_nr).flatten(),:]
    if timestep is not None:
        subset = subset[np.argwhere(sublabels[:,1]==timestep).flatten(),:]
        sublabels = sublabels[np.argwhere(sublabels[:,1]==timestep).flatten(),:]
    if leaf is not None:
        subset = subset[np.argwhere(sublabels[:,2]==leaf).flatten(),:]
        sublabels = sublabels[np.argwhere(sublabels[:,2]==leaf).flatten(),:]
    return subset, sublabels

def compress(query, components, pca):
    query_weight = components @ (query - pca.mean_).T # compress
    return query_weight

def decompress(weights, components, pca):
    projection = weights.T @ components + pca.mean_ # decompress
    return projection

def get_encoding(train_split=0, dir=None):
    if dir is None:
        dir = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D', '_processed', 'pca_input')
    data, names = load_inputs(dir)
    labels = get_labels(names)
    standardised, scalar = standardise_pc_scale(data)
    train_ds, test_ds, train_labels, test_labels = split_dataset(standardised, labels, split=train_split)
    pca, transformed = fit_pca(standardised)
    return train_ds, test_ds, train_labels, test_labels, pca

if __name__== "__main__":
    dir = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D', '_processed', 'pca_input')
    train_ds, test_ds, train_labels, test_labels, pca = get_encoding(0, dir)

    #plot_explained_variance(pca)
    #test_reprojection_loss(train_ds, pca)
    new_random_leaf(train_ds, train_labels, pca)
    #perform_single_reprojection(train_ds[0], pca, components = 50, draw=True)
    single_plant_leaves, single_plant_labels= select_subset(train_ds, train_labels, plant_nr = 6)
    #plot_3_components(train_ds, pca)
    #plot_3_components(train_ds, pca, labels = train_labels)
    #plot_2_components(train_ds, pca, labels = train_labels)
    t_sne(train_ds, train_labels)
