import os
import open3d as o3d
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import visualise

def load_inputs(dir):
    leaves = os.listdir(dir)
    data = np.asarray([np.load(os.path.join(dir, leaf)) for leaf in leaves])
    return data

def fit_pca(sample):
    x_data = sample[:,:,0]
    y_data = sample[:,:,1]
    z_data = sample[:,:,2]

    pca_x = PCA()
    pca_y = PCA()
    pca_z = PCA()

    pca_x.fit(x_data)
    pca_y.fit(y_data)
    pca_z.fit(z_data)

    transformed_x = pca_x.fit_transform(x_data)
    transformed_y = pca_y.fit_transform(y_data)
    transformed_z = pca_z.fit_transform(z_data)

    print(f"Using {pca_x.components_.shape[0]} components")
    print(f"Explained variance ration in the x dimension : {sum(pca_x.explained_variance_ratio_)}")
    print(f"Explained variance ration in the y dimension : {sum(pca_y.explained_variance_ratio_)}")
    print(f"Explained variance ration in the z dimension : {sum(pca_z.explained_variance_ratio_)}")

    return [pca_x, pca_y, pca_z]

def split_dataset(data, split = 0.2):
    total_nr_examples = data.shape[0]
    nr_test_examples = int(np.floor(split * total_nr_examples))
    indeces = np.random.choice(range(total_nr_examples), size=nr_test_examples, replace=False)
    test_data = data[indeces,:,:]
    train_data = np.delete(data, indeces, axis=0)
    return train_data, test_data

def perform_single_reprojection(sample, pcas, components=None, draw = False):
    '''One single leaf (point cloud) at a time'''
    if components is None:
        components = pcas[0].components_.shape[0]
    reprojection_parts = []
    for dimension in range(len(pcas)):
        query = sample[:,dimension]
        all_eigenleaves = pcas[dimension].components_
        query_weight = compress(query, all_eigenleaves[:components], pcas[dimension])
        reprojection_part = decompress(query_weight, all_eigenleaves[:components], pcas[dimension])
        reprojection_parts.append(reprojection_part)
    full_reprojection = np.stack((reprojection_parts), axis = -1)

    euclidean_dist_error = np.linalg.norm(sample-full_reprojection, axis = 1) # euclidean distance between each point pair
    mean_error = euclidean_dist_error.mean()
    print(f"Mean reprojection error at {components} components : {mean_error}")

    if draw:
        cloud1 = visualise.array_to_o3d_pointcloud(sample)
        cloud2 = visualise.array_to_o3d_pointcloud(full_reprojection)
        visualise.draw([cloud1, cloud2], 'test', offset = True )
    return

def test_reprojection_loss(sample, pcas):
    max_components = pcas[0].components_.shape[0]
    mean_errors = []
    for i in range(max_components, 0, -1):
        reprojection_parts = []
        for dimension in range(len(pcas)):
            query = sample[:,:,dimension]
            all_eigenleaves = pcas[dimension].components_
            query_weight = compress(query, all_eigenleaves[:i], pcas[dimension])
            reprojection = decompress(query_weight, all_eigenleaves[:i], pcas[dimension])
            reprojection_parts.append(reprojection)
        full_reprojection = np.stack((reprojection_parts), axis = -1)

        euclidean_dist_error = np.linalg.norm(sample-full_reprojection, axis = -1) # euclidean distance between each point pair
        mean_error = euclidean_dist_error.mean(axis=-1).mean()
        mean_errors.append(mean_error)

    plt.plot(range(max_components, 0, -1), mean_errors)
    plt.ylabel('Mean RMSE')
    plt.xlabel('Number of principal components')
    plt.show()

def compress(query, components, pca):
    query_weight = components @ (query - pca.mean_).T # compress
    return query_weight

def decompress(weights, components, pca):
    projection = weights.T @ components + pca.mean_ # decompress
    return projection

if __name__== "__main__":
    dir = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D', '_processed', 'pca_input')
    data = load_inputs(dir)
    train, test = split_dataset(data)
    pcas = fit_pca(train)
    test_reprojection_loss(test, pcas)
    for leaf in data:
        perform_single_reprojection(leaf, pcas, components = 100, draw=True)
