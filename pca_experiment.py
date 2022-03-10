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
    #print(pca.explained_variance_ratio_)
    #print(pca.singular_values_)
    return transformed_points, pca.components_

def split_into_organs(points, labels):
    #center the point cloud around the origin BEFORE splitting, retain relative global pose for the components
    for dimension in range(points.shape[1]):
        points[:,dimension] = points[:,dimension] - (min(points[:,dimension] + (max(points[:,dimension])-min(points[:,dimension]))/2))

    # Splitting the point cloud into sub-clouds for each unique label
    organs = []
    for label in np.unique(labels).astype(int):
        organs.append((points[(labels.astype(int).flatten()==label),:], labels[(labels.astype(int).flatten()==label),:]))

    # for organ in organs:
    #     vis = util.draw_cloud(organ[0], organ[1], draw=True)
    return organs

def discretise_and_flatten(points, labels):
    '''
    takes one plant point cloud, split into components (organs)
    rotates each leaf along the broadest spread with the principal components, maintaining 3 dimensions
    then centers and scales each leaf individually, discretises into a 256x256 depth map
    returns an array of unraveled depth maps, one for each leaf of the plant
    '''
    organs = split_into_organs(points, labels)

    leaves = []
    instance_labels_lst = []
    for organ in organs[2:]: # this takes every organ excluding the stem, which is the first
        rotated_organ, principal_components = pca(organ[0], 3)

        mins = np.min(rotated_organ, axis=0)
        maxes = np.max(rotated_organ, axis=0)
        ranges = np.ptp(rotated_organ, axis=0)
        centered = rotated_organ - (mins+ranges/2) #all three axes centered on 0
        rescaled = ((255/2) * centered / ((max([ranges[0],ranges[1]]))/2)) + 127.5 #rescale all 3 dimensions by the same scalar so that all values lie between 0 and 255
        pixel_coordinates = np.rint(rescaled[:,:2]).astype(int)
        ind = np.lexsort((pixel_coordinates[:,1],pixel_coordinates[:,0]))
        sorted_by_pixels = pixel_coordinates[ind]

        img = np.ones((256,256))*-1
        relevant_pixels = np.unique(sorted_by_pixels[:,:2], axis = 0) #loop over each unique pixel that has points in it
        for pixel in relevant_pixels:
            height_indeces = np.argwhere((sorted_by_pixels[:,0]==pixel[0]) & (sorted_by_pixels[:,1]==pixel[1]))
            img[pixel[0],pixel[1]] = np.rint(np.mean(rescaled[height_indeces,2])).astype(int)

        leaves.append(img)
        instance_labels_lst.append(int(organ[1][0]))

    instance_labels = np.asarray(instance_labels_lst)
    data = np.asarray(leaves) # stack all images
    data = np.reshape(data, (len(leaves),-1)) #unroll the single images to obtain 2D vector
    return data, instance_labels

def stack_plant_leaves(files):
    '''
    Takes a list of files, opens and stacks the data in a list of lists by plant and timestep
    '''
    label_counter = 0
    label_ids = {}

    for plant_series in files:
        max_nr_leaves = 0
        for time_step in plant_series:
            points,labels,plant_id = util.open_file(time_step)
            leaves, instance_labels = discretise_and_flatten(points, labels)
            numeric_labels = instance_labels - 1 + label_counter

            try:
                data = np.concatenate((data, leaves), axis=0)
            except:
                data = leaves

            try:
                full_labels = np.concatenate((full_labels, numeric_labels), axis = 0)
            except:
                full_labels = numeric_labels

            if max(numeric_labels) > max_nr_leaves:
                max_nr_leaves = max(numeric_labels)

        for i in range(max_nr_leaves):
            label_ids[label_counter + i + 1] = plant_id + '_leaf_' + str(i + 1)

        label_counter += max_nr_leaves

    np.save('flattened_leaves.npy', data)
    np.save('labels.npy', full_labels)
    print('Flattened and discretised leaf data set saved to file')
    import pdb; pdb.set_trace()

def pca_across_discretised_leaves(data):
    '''
    takes an array of unravelled depth maps
    of size (examples x 65536)
    '''

    return


def experiment_split_by_organ(points, labels):
    '''
    takes one plant point cloud, split into components (organs)
    selects the first organ and rotates it along the principal component vectors
    '''
    components = split_into_organs(points, labels)
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
    organs = split_into_organs(points, labels)

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
    all_files, annotated_files = util.get_file_locations(data_directory)
    annotated_tomatoes = [entry for entry in annotated_files if "Tomato" in entry[0]]

    stack_plant_leaves(annotated_tomatoes)
    #discretise_and_flatten(points, labels)
    #points,labels,id = util.open_file(annotated_tomatoes[0][3])
    #multi_leaf_pca(points, labels)
    #experiment_simple_whole_plant(points, labels)
    #experiment_split_by_organ(points, labels)
