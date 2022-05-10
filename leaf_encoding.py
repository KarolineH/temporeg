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

def fit_pca(data):
    flat_data = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
    pca = PCA()
    pca.fit(flat_data)
    transformed = pca.fit_transform(flat_data)

    print(f"Using {pca.components_.shape[0]} components")
    print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_)}")
    plt.plot(np.cumsum(pca.explained_variance_ratio_[:100]))
    plt.show()
    return pca, transformed

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

def select_subset(data, labels, plant_nr=None, timestep=None, leaf=None):
    subset = copy.deepcopy(data)
    sublabels = copy.deepcopy(labels)

    if plant_nr is not None:
        subset = subset[np.argwhere(sublabels[:,0]==plant_nr).flatten(),:]
        sublabels = sublabels[np.argwhere(sublabels[:,0]==plant_nr).flatten(),:]
    if timestep is not None:
        subset = subset[np.argwhere(sublabels[:,1]==timestep).flatten(),:]
        sublabels = sublabels[np.argwhere(sublabels[:,1]==plant_nr).flatten(),:]
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

if __name__== "__main__":
    dir = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D', '_processed', 'pca_input')
    data, names = load_inputs(dir)
    labels = get_labels(names)
    train_ds, test_ds, train_labels, test_labels = split_dataset(data, labels)
    pca, transformed = fit_pca(train_ds)
    #test_reprojection_loss(test_ds, pca)

    single_plant_leaves, single_plant_labels= select_subset(data, labels, plant_nr = 2)
    plot_3pcs(single_plant_leaves, single_plant_labels, pca)

    for leaf in data:
        perform_single_reprojection(leaf, pca, components = 50, draw=True)
