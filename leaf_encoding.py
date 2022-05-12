import os
import open3d as o3d
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import visualise
from sklearn.manifold import TSNE
import seaborn as sns
import copy

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
    original_shape = data.shape
    mins = np.asarray([np.min(leaf, axis=0) for leaf in data])
    maxes = np.asarray([np.max(leaf, axis=0) for leaf in data])
    ranges = np.asarray([np.ptp(leaf, axis=0) for leaf in data])
    flat_data = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
    scale_factor = np.max(abs(np.concatenate((mins, maxes), axis=1)), axis=1)
    standardised = flat_data / scale_factor[:,None]
    out = standardised.reshape(original_shape)
    return out

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
    '''One single leaf (point cloud) at a time'''
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
    original_shape = sample.shape
    max_components = pca.components_.shape[0]
    mean_errors = []
    all_eigenleaves = pca.components_
    query = sample.reshape(sample.shape[0], sample.shape[1]*sample.shape[2])
    for i in range(max_components, 0, -1):
        query_weight = compress(query, all_eigenleaves[:i], pca)
        reprojection = decompress(query_weight, all_eigenleaves[:i], pca)
        unrolled = reprojection.reshape(original_shape) # restack
        euclidean_dist_error = np.linalg.norm(sample-unrolled, axis = -1) # euclidean distance between each point pair
        mean_error = euclidean_dist_error.mean(axis=-1).mean()
        mean_errors.append(mean_error)

    plt.plot(range(max_components, 0, -1), mean_errors)
    plt.ylabel('Mean RMSE')
    plt.xlabel('Number of principal components')
    plt.show()

def plot_3pcs(data, labels, pca):
    flat_data = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
    transformed = pca.transform(flat_data)

    ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    scatterplot = ax.scatter(xs=transformed[:,0], ys=transformed[:,1], zs=transformed[:,2],c=labels[:flat_data.shape[0],2], cmap='rainbow')
    ax.set_xlabel('first component')
    ax.set_ylabel('second component')
    ax.set_zlabel('third component')
    legend1 = ax.legend(*scatterplot.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    plt.show()
    import pdb; pdb.set_trace()

def inspect_feature_vectors(data, labels, pca):
    original_shape = data.shape
    query = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
    all_eigenleaves = pca.components_
    query_weight = compress(query, all_eigenleaves[:50], pca).T #(examples x components)

    mins = np.asarray([np.min(leaf, axis=0) for leaf in data])
    maxes = np.asarray([np.max(leaf, axis=0) for leaf in data])
    plt.boxplot(query_weight)
    plt.show()

    # random new leaf
    # sample each feature randomly from the existing feature vectors
    new_leaf = np.asarray([query_weight[np.random.choice(query_weight.shape[0]),i] for i in range(query_weight.shape[1])])
    reprojection = decompress(new_leaf, all_eigenleaves[:50], pca)
    random_leaf = reprojection.reshape(1,original_shape[1],original_shape[2])
    cloud = visualise.array_to_o3d_pointcloud(random_leaf[0])
    visualise.draw2([cloud], 'test', offset = True )

    import pdb; pdb.set_trace()

def t_sne(data, labels):
    # tsne on just coordinates
    tsne = TSNE()
    flat_data = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
    X_embedded = tsne.fit_transform(flat_data)
    palette = sns.color_palette("hls", np.unique(labels[:,2]).shape[0])
    sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=labels[:,2], legend='auto', palette=palette)
    plt.show()

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
    standardised = standardise_pc_scale(data)
    train_ds, test_ds, train_labels, test_labels = split_dataset(standardised, labels, split=train_split)
    pca, transformed = fit_pca(standardised)
    return train_ds, test_ds, train_labels, test_labels, pca

if __name__== "__main__":
    dir = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D', '_processed', 'pca_input')
    train_ds, test_ds, train_labels, test_labels, pca = get_encoding(0, dir)

    #plot_explained_variance(pca)
    #test_reprojection_loss(train_ds, pca)

    inspect_feature_vectors(train_ds, train_labels, pca)

    single_plant_leaves, single_plant_labels= select_subset(train_ds, train_labels, plant_nr = 6)
    t_sne(single_plant_leaves, single_plant_labels)


    #plot_3pcs(single_plant_leaves, single_plant_labels, pca)
    for leaf in train_ds:
        perform_single_reprojection(leaf, pca, components = 50, draw=True)
