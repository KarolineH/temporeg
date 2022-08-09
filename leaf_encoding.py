import os
import open3d as o3d
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import visualise
from sklearn.manifold import TSNE
import copy
import matplotlib.path as path
import matplotlib.cm as cm
from sklearn.cluster import KMeans

'''
ENCODING and utilities
'''

def load_inputs(dir):
    leaves = os.listdir(dir)
    data = np.asarray([np.load(os.path.join(dir, leaf)) for leaf in leaves])
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
    return pca, transformed

def plot_explained_variance(pca):
    print(f"Using {pca.components_.shape[0]} components")
    print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_)}")
    plt.plot(np.cumsum(pca.explained_variance_ratio_[:100]))
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Number of principal components')
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
    #standardised, scalar = standardise_pc_scale(data)
    train_ds, test_ds, train_labels, test_labels = split_dataset(data, labels, split=train_split)
    pca, transformed = fit_pca(data)
    return train_ds, test_ds, train_labels, test_labels, pca, transformed


'''
TESTING and plotting
'''

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
        scatterplot = ax.scatter(xs=transformed[:,0], ys=transformed[:,1], zs=transformed[:,2],c=labels[:flat_data.shape[0],3], cmap='rainbow')
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

def plot_2_components(data, pca, labels=None, components=[0,1]):
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
        scatterplot = ax.scatter(transformed[:,components[0]], transformed[:,components[1]],c=labels[:flat_data.shape[0],3], cmap='rainbow')
        ax.set_xlabel('component {}'.format(components[0]))
        ax.set_ylabel('component {}'.format(components[1]))
        legend1 = ax.legend(*scatterplot.legend_elements(), title="Classes")
        ax.add_artist(legend1)
        plt.show()

    # Plot using 2D projected leaf outline as markers, but no colours
    fig, ax = plt.subplots()
    for _m, _x, _y in zip(markers, transformed[:,components[0]], transformed[:,components[1]]):
        scatterplot = ax.scatter(_x, _y, marker=_m, s=500, c='#FFFFFF', edgecolors='#1f77b4')
    plt.tight_layout()
    ax.set_xlabel('component {}'.format(components[0]))
    ax.set_ylabel('component {}'.format(components[1]))
    legend1 = ax.legend(*scatterplot.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    plt.show()

def new_random_leaf_from_distribution(data, labels, pca, draw=True):
    '''
    Gets the distributions for all principal components
    Randomly samples components from their distributions to create a new plausible random leaf
    '''
    original_shape = data.shape
    query = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
    all_eigenleaves = pca.components_
    query_weight = compress(query, all_eigenleaves[:50], pca).T #(examples x components)
    ranges = np.asarray([np.min(query_weight, axis=0),np.max(query_weight, axis=0)])
    if draw:
        plt.boxplot(query_weight)
        plt.title("Boxplots of Value distributions for the first 50 components")
        plt.show()
        plt.hist(query_weight[:,0], 20)
        plt.title("Histogram of feature values on the first principal component")
        plt.show()
        plt.hist(query_weight[:,1], 20)
        plt.title("Histogram of feature values on the second principal component")
        plt.show()

    # random new leaf
    # sample each feature randomly from the existing feature vectors
    new_leaf = np.asarray([query_weight[np.random.choice(query_weight.shape[0]),i] for i in range(query_weight.shape[1])])
    reprojection = decompress(new_leaf, all_eigenleaves[:50], pca)
    random_leaf = reprojection.reshape(1,original_shape[1],original_shape[2])
    mean_leaf = pca.mean_.reshape(1,original_shape[1],original_shape[2])

    cloud = visualise.array_to_o3d_pointcloud(random_leaf[0])
    cloud2 = visualise.array_to_o3d_pointcloud(mean_leaf[0])
    if draw:
        visualise.draw2([cloud, cloud2], 'test', offset = True )
    return ranges

def t_sne(data, labels=None, label_meaning=None):
    # tsne on just coordinates
    tsne = TSNE()
    flat_data = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
    X_embedded = tsne.fit_transform(flat_data)
    markers = [path.Path(leaf[:,:2]) for leaf in data]
    if labels is not None:
        max_color = max(labels)
        cmap = cm.get_cmap('rainbow')
        colours = cmap((labels/max_color))

    fig, ax = plt.subplots()
    ax.set_title('t-SNE embedding of leaf outline coordinates')
    if labels is not None:
        for i, _m, _x, _y in zip(range(len(markers)), markers, X_embedded[:,0], X_embedded[:,1]):
            scatterplot = ax.scatter(_x, _y, marker=_m, s=500, c='#FFFFFF', edgecolors=colours[i].reshape(1,-1))
        #legend1 = ax.legend(*ax.legend_elements(), title="Classes")
        legend1 = ax.legend(np.unique(labels, axis = 0), labelcolor=cmap((np.unique(labels, axis = 0)/max_color)), ncol=2, markerscale=0)
        ax.add_artist(legend1)
        if label_meaning is not None:
            ax.set_title('t-SNE embedding of outline coordinates. Colours represent {}'.format(label_meaning))
    else:
        for _m, _x, _y in zip(markers, X_embedded[:,0], X_embedded[:,1]):
            scatterplot = ax.scatter(_x, _y, marker=_m, s=500, c='#FFFFFF', edgecolors='#1f77b4')
    plt.show()

def pca_then_t_sne(data, labels=None, label_meaning=None, nr_components=50):
    #tsne on the pca extracted feature components
    tsne = TSNE()
    flat_data = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
    all_eigenleaves = pca.components_
    query_weight = compress(flat_data, all_eigenleaves[:nr_components], pca)
    query_weight.T
    X_embedded = tsne.fit_transform(query_weight.T)
    markers = [path.Path(leaf[:,:2]) for leaf in data]
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

def analyse_feature_space_clusters(data, labels, pca, nr_clusters=4):
    flat_data = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
    transformed = pca.transform(flat_data)
    kmeans = KMeans(n_clusters=nr_clusters, random_state=0).fit(transformed)
    plt.scatter(transformed[:,0], transformed[:,1],c=kmeans.labels_)
    plt.show()

    fig, axs = plt.subplots(3,3)

    clusters = [data[np.argwhere(kmeans.labels_== cluster).flatten(),:] for cluster in range(nr_clusters)]
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

def recreate_artefact(data, pca):
    sample = data[1]
    original_shape = sample.shape
    all_eigenleaves = pca.components_

    sample_flipped = np.flip(sample, 0)
    query_flipped = sample_flipped.reshape(sample.shape[0]*sample.shape[1])
    flipped_weight = compress(query_flipped, all_eigenleaves[:50], pca)
    query = sample.reshape(sample.shape[0]*sample.shape[1])
    query_weight = compress(query, all_eigenleaves[:50], pca)

    np.linalg.norm(query_weight-flipped_weight)
    import pdb; pdb.set_trace()

if __name__== "__main__":
    dir = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D', '_processed', 'pca_input')
    train_ds, test_ds, train_labels, test_labels, pca, transformed = get_encoding(0, dir)
    single_plant_leaves, single_plant_labels= select_subset(train_ds, train_labels, plant_nr = 6)

    #plot_explained_variance(pca)
    #test_reprojection_loss(train_ds, pca)
    perform_single_reprojection(train_ds[0], pca, components = 50, draw=True)
    #ranges = new_random_leaf_from_distribution(train_ds, train_labels, pca)
    #recreate_artefact(train_ds, pca)

    #plot_3_components(train_ds, pca)
    #plot_3_components(train_ds, pca, labels = train_labels)
    #plot_2_components(train_ds, pca, labels = train_labels, components = [0,1])

    # t_sne(train_ds, train_labels[:,3], label_meaning='leaf number')
    # pca_then_t_sne(train_ds, train_labels[:,3], label_meaning='leaf number')
    #t_sne(train_ds, label_meaning='leaf number')
    #pca_then_t_sne(train_ds, label_meaning='leaf number')
    t_sne(single_plant_leaves, single_plant_labels[:,3], label_meaning='leaf number')
    pca_then_t_sne(single_plant_leaves, single_plant_labels[:,3], label_meaning='leaf number')
    #analyse_feature_space_clusters(train_ds, train_labels, pca)
