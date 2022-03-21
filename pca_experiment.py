import os
import pheno4d_util as util
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import open3d as o3d

def pca(points, num_comp):
    pca = PCA(n_components=num_comp)
    pca.fit(points)
    transformed_points = pca.fit_transform(points)
    #print(pca.explained_variance_ratio_)
    #print(pca.singular_values_)
    return transformed_points, pca.components_

def show_all_images(data):
    im_shape = (256,256)
    counter = 0
    for j in range(int(np.floor(data.shape[0]/16))):
        fig, axes = plt.subplots(4,4,sharex=True,sharey=True,figsize=(8,10))
        for i in range(16):
            axes[i%4][i//4].imshow(data[counter].reshape(im_shape), cmap="gray")
            counter += 1
        plt.show()

# Left over code to test discretised pca like a recognition task
# def test_discretised_pca(query, real_label, pca, eigenleaves, weights, train_set, train_labels, im_shape):
#     #print('real label '+ str(real_label))
#     query_weight = eigenleaves @ (query - pca.mean_).T
#     euclidean_distance = np.linalg.norm((weights.T - query_weight).T, axis=0)
#     best_match = np.argmin(euclidean_distance)
#     #print("Best match %s with Euclidean distance %f" % (train_labels[best_match], euclidean_distance[best_match]))
#     outcome = (train_labels[best_match] == real_label)
#
#     #Visualize
#     fig, axes = plt.subplots(1,2,sharex=True,sharey=True,figsize=(8,6))
#     axes[0].imshow(query.reshape(im_shape), cmap="gray")
#     axes[0].set_title("Query")
#     axes[1].imshow(train_set[best_match].reshape(im_shape), cmap="gray")
#     axes[1].set_title("Best match")
#     plt.show()
#
#     return outcome, train_labels[best_match]

def experiment_split_by_organ(points, labels):
    '''
    takes one plant point cloud, split into components (organs)
    selects the first organ and rotates it along the principal component vectors
    '''
    components = util.split_into_organs(points, labels)
    transformed_component3, vectors = pca(components[0][0], 3)
    transformed_component2, _ = pca(components[0][0], 2)
    transformed_component1, _ = pca(components[0][0], 1)

    # plt.ion()
    # plt.hist(transformed_component1, 100) #histogram of point distribution along single principal component
    # import pdb; pdb.set_trace()
    # plt.clf()
    # plt.scatter(transformed_component2[:,0], transformed_component2[:,1]); #2D downsampled visualisation
    # import pdb; pdb.set_trace()
    vis = util.draw_cloud(transformed_component3, components[0][1], draw=True)
    import pdb; pdb.set_trace()

def multi_leaf_pca(points, labels):
    '''
    Takes multiple leaf point clouds, randomly drops points to adjust them to the same size
    pca is broken currently
    '''
    organs = util.split_into_organs(points, labels)

    # from all leaf point clouds of this plant,
    # find what size the smallest point cloud is
    min_p = None
    for organ in organs[2:]:
        if min_p == None or min_p > organ[0].shape[0]:
            min_p = organ[0].shape[0]

    # randomly drop points from the larger clouds to make them uniform in size
    leaves = np.empty([(len(organs)-2),min_p,3])
    for i, organ in enumerate(organs[2:]):
        indeces = np.arange(organ[0].shape[0])
        np.random.shuffle(indeces)
        random_point_selection = organ[0][indeces[:min_p],:]
        leaves[i,:,:] = random_point_selection
    leaves = np.reshape(leaves, (7,-1))
    np.reshape(leaves, (7,-1))

    import pdb; pdb.set_trace()
    pca = PCA(0.95)
    pca.fit(leaves)
    low_dim_leaf = pca.transform(leaves[0,:,:])
    high_dim_leaf = leaves[0,:,:]
    import pdb; pdb.set_trace()

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

    import pdb; pdb.set_trace()
    result_3comp, vectors = pca(points, 3)
    result_2comp, _ = pca(points, 2)
    result_1comp, _ = pca(points, 1)

    plt.ion()

    #plt.hist(result_1comp, 100) #histogram of point distribution along single principal component
    #plt.show()

    plt.scatter(result_2comp[:,0], result_2comp[:,1]); #2D downsampled visualisation
    import pdb; pdb.set_trace()

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

    #show_all_images(data)
    #points,labels,id = util.open_file(annotated_tomatoes[0][3])
    #multi_leaf_pca(points, labels)
    #experiment_simple_whole_plant(points, labels)
    #experiment_split_by_organ(points, labels)
