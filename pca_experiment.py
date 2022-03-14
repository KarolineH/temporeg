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
    takes one plant point cloud, split into components (organs),
    rotates each leaf along the broadest spread with the principal components, maintaining 3 dimensions
    then centers and scales each leaf individually, discretises into a 256x256 depth map with a depth average at each pixel
    returns an array of unraveled depth maps, one for each leaf of the plant
    '''
    organs = split_into_organs(points, labels)

    leaves = []
    instance_labels_lst = []
    for organ in organs[2:]: # this takes every organ excluding the stem, which is the first
        rotated_organ, principal_components = pca(organ[0], 3) #rotated to align with the 3 eigenvectors

        mins = np.min(rotated_organ, axis=0)
        maxes = np.max(rotated_organ, axis=0)
        ranges = np.ptp(rotated_organ, axis=0)
        centered = rotated_organ - (mins+ranges/2) #all three axes centered on 0

        # Check if the orientation is right, otherwise flip the eigenvectors
        # from the min and max point in each dimension, find their distance to every other point, along that dimension
        # compare sum of those distances from both ends and align towards the larger one
        # see https://dro.dur.ac.uk/18562/1/18562.pdf?DDD10+dcs0ii+dul4eg
        distsum_low = np.sum((rotated_organ - mins), axis=0)
        distsum_high = np.sum((maxes - rotated_organ), axis=0)
        for i in range(distsum_low.shape[0]):
            # Flip the axis if the sum of distances from the low point is higher
            if [distsum_low[i] > distsum_high[i]]:
                centered[:,i] = -centered[:,i]

        rescaled = ((255/2) * centered / ((max([ranges[0],ranges[1]]))/2)) + 127.5 #rescale all 3 dimensions by the same scalar so that all values lie between 0 and 255

        # now discretise the point set into a depth map
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

def stack_Tomato_leaves(file_directory):
    '''
    Takes a directory, opens all the annotated tomato files, flattens and discretises the leaves,
    stacks them as depth maps with their individual leaf label,
    also saves a dictionary of IDs to reference which plant and timestep they came from.
    If the files are already in the data directory, it skips over the procedure and loads from file.
    '''

    if not os.path.isfile(os.path.join(file_directory,'flattened_leaves.npy')):
        all_files, annotated_files = util.get_file_locations(file_directory)
        annotated_tomatoes = [entry for entry in annotated_files if "Tomato" in entry[0]] #consider all annotated tomato files

        label_counter = 0
        label_ids = {}

        for plant_series in annotated_tomatoes:
            max_nr_leaves = 0
            for time_step in plant_series: # for inividual scans
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

                if max(instance_labels) > max_nr_leaves:
                    max_nr_leaves = max(instance_labels) - 1

            for i in range(max_nr_leaves):
                label_ids[label_counter + i + 1] = plant_id + '_leaf_' + str(i + 1)

            label_counter += max_nr_leaves

        np.save(os.path.join(file_directory, 'flattened_leaves.npy'), data)
        np.save(os.path.join(file_directory, 'labels.npy'), full_labels)
        np.save(os.path.join(file_directory, 'label_IDs.npy'), label_ids)
        print('Flattened and discretised leaf data set saved to file')
    else:
        'Data has already been extracted and stacked. Loading data instead.'
        data = np.load(os.path.join(file_directory, 'flattened_leaves.npy'))
        full_labels = np.load(os.path.join(file_directory, 'labels.npy'))
        label_ids = np.load(os.path.join(file_directory, 'label_IDs.npy'), allow_pickle=True)

    return data, full_labels, label_ids

def pca_across_discretised_leaves(data, labels):
    '''
    takes an array of unravelled depth maps
    of size (examples x 65536)
    performs pca to indetify eigenleaves and the individual linear combination of them for each input
    Adapted from https://machinelearningmastery.com/face-recognition-using-principal-component-analysis/
    '''
    visualisations = True
    out_data_leaf = 3 # number of which leaf to remove completely from the set for testing
    im_shape = (256,256)

    leaf_sort_order = labels.argsort()
    sorted_labels = labels[leaf_sort_order]
    sorted_data = data[leaf_sort_order] # sort the data by leaf number

    # First remove one leaf completely from the set for an out-data test
    out_data_test_set = sorted_data[np.where(sorted_labels == 3)]#  All instances of nr 3, which will be removed from the training data too
    sorted_data = np.delete(sorted_data, np.where(sorted_labels == out_data_leaf), axis=0)
    sorted_labels = np.delete(sorted_labels, np.where(sorted_labels == out_data_leaf), axis=0)
    # Then extract the in-data test set, selecting one instance of each leaf that is included in the data set more than once
    (unique, counts) = np.unique(sorted_labels, return_counts=True)
    in_data_test_leaves = unique[np.where(counts > 1)]
    in_data_test_examples = [np.searchsorted(sorted_labels, leaf) for leaf in in_data_test_leaves]
    in_data_test_set = np.take(sorted_data, in_data_test_examples, axis=0)
    in_data_test_labels = np.take(sorted_labels, in_data_test_examples, axis=0)
    train_set = np.delete(sorted_data, in_data_test_examples, axis=0)
    train_labels = np.delete(sorted_labels, in_data_test_examples, axis=0)

    pca = PCA().fit(train_set)
    n_components = 100
    eigenleaves = pca.components_[:n_components]

    if visualisations==True:
        # Show some examples of input leaves
        fig, axes = plt.subplots(4,4,sharex=True,sharey=True,figsize=(8,10))
        for i in range(16):
            axes[i%4][i//4].imshow(train_set[i].reshape(im_shape), cmap="gray")
        print("Showing the input data")
        plt.show()

        # Show the first 16 eigenleaves
        fig, axes = plt.subplots(4,4,sharex=True,sharey=True,figsize=(8,10))
        for i in range(16):
            axes[i%4][i//4].imshow(eigenleaves[i].reshape(im_shape), cmap="gray")
        print("Showing the eigenleaves")
        plt.show()

        # Show how a single leaf changes over time
        leaf_1 = np.take(sorted_data, np.where(sorted_labels == 8), axis=0).squeeze()
        fig, axes = plt.subplots(4,4,sharex=True,sharey=True,figsize=(8,10))
        for i in range(leaf_1.shape[0]):
            axes[i%4][i//4].imshow(leaf_1[i].reshape(im_shape), cmap="gray")
        print("Showing the same leaf over time")
        plt.show()

    weights = eigenleaves @ (train_set - pca.mean_).T

    ### TESTING
    outcomes = []
    correct = 0
    for i,leaf in enumerate(in_data_test_set):
        query = leaf
        query_label = in_data_test_labels[i]
        outcome, prediction = test_discretised_pca(query, query_label, pca, eigenleaves, weights, train_set, train_labels, im_shape)
        outcomes.append(prediction)
        if outcome == True:
            correct += 1
    import pdb; pdb.set_trace()

def show_all_images(data):
    im_shape = (256,256)
    counter = 0
    for j in range(int(np.floor(data.shape[0]/16))):
        fig, axes = plt.subplots(4,4,sharex=True,sharey=True,figsize=(8,10))
        for i in range(16):
            axes[i%4][i//4].imshow(data[counter].reshape(im_shape), cmap="gray")
            counter += 1
        plt.show()

def test_discretised_pca(query, real_label, pca, eigenleaves, weights, train_set, train_labels, im_shape):
    #print('real label '+ str(real_label))
    query_weight = eigenleaves @ (query - pca.mean_).T
    euclidean_distance = np.linalg.norm((weights.T - query_weight).T, axis=0)
    best_match = np.argmin(euclidean_distance)
    #print("Best match %s with Euclidean distance %f" % (train_labels[best_match], euclidean_distance[best_match]))
    outcome = (train_labels[best_match] == real_label)

    #Visualize
    fig, axes = plt.subplots(1,2,sharex=True,sharey=True,figsize=(8,6))
    axes[0].imshow(query.reshape(im_shape), cmap="gray")
    axes[0].set_title("Query")
    axes[1].imshow(train_set[best_match].reshape(im_shape), cmap="gray")
    axes[1].set_title("Best match")
    plt.show()

    return outcome, train_labels[best_match]

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

    data, labels, __ = stack_Tomato_leaves(data_directory)
    pca_across_discretised_leaves(data, labels)

    #show_all_images(data)
    #discretise_and_flatten(points, labels)
    #points,labels,id = util.open_file(annotated_tomatoes[0][3])
    #multi_leaf_pca(points, labels)
    #experiment_simple_whole_plant(points, labels)
    #experiment_split_by_organ(points, labels)
