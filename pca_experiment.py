import os
import pheno4d_util as util
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import open3d as o3d

def pca(points, num_comp):
    pca = PCA(n_components=num_comp)
    pca.fit(points)
    transformed_points = pca.fit_transform(points)
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)
    return transformed_points, pca.components_

def experiment_simple_whole_plant(points, labels):
    '''
    PCA with 1,2, and 3 components on a simple 3D point cloud,
    includes plots of 1D histogram, 2D plane, and 2 3D visualisations:
    One of the original data with the principal component vectors,
    and one with the transformation applied
    '''
    #center the point cloud around the origin
    for dimension in range(points.shape[1]):
        points[:,dimension] = points[:,dimension] - (min(points[:,dimension] + (max(points[:,dimension])-min(points[:,dimension]))/2))

    result_3comp, vectors = pca(points, 3)
    result_2comp, _ = pca(points, 2)
    result_1comp, _ = pca(points, 1)

    #plt.hist(result_1comp, 100) #histogram of point distribution along single principal component
    #plt.show()

    #plt.scatter(result_2comp[:,0], result_2comp[:,1]); #2D downsampled visualisation
    #plt.show()

    nodes = [[0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 100]]
    lines = [[0, 1], [0, 2], [0, 3]]
    coordinate_frame = o3d.geometry.LineSet()
    coordinate_frame.points = o3d.utility.Vector3dVector(nodes)
    coordinate_frame.lines = o3d.utility.Vector2iVector(lines)

    nodes = [[0,0,0]] + (vectors *200).tolist()
    principal_vectors = o3d.geometry.LineSet()
    principal_vectors.points = o3d.utility.Vector3dVector(nodes)
    principal_vectors.lines = o3d.utility.Vector2iVector(lines)
    colors = [[1, 0, 0] for i in range(len(lines))]
    principal_vectors.colors = o3d.utility.Vector3dVector(colors)

    vis = util.draw_cloud(result_3comp, labels, draw=False)
    vis.add_geometry(coordinate_frame)
    vis.add_geometry(principal_vectors)
    vis.run()

def experiment_labels_inlcuded(points, labels):
    features = np.concatenate((points,labels), axis=1)
    #center the point cloud around the origin
    for dimension in range(features.shape[1]):
        features[:,dimension] = features[:,dimension] - (min(features[:,dimension] + (max(features[:,dimension])-min(features[:,dimension]))/2))

    result_3comp, vectors = pca(points, 3)
    vis = util.draw_cloud(result_3comp, labels, draw=False)
    vis.run()

if __name__== "__main__":
    data_directory = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D')
    all_files, annotated_files = util.get_file_locations(data_directory)
    points,labels = util.open_file(annotated_files[0][0])

    #experiment_simple_whole_plant(points, labels)
    experiment_labels_inlcuded(points, labels)
