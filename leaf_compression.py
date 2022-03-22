import os
import pheno4d_util as util
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

def get_pca(points, stempoint=None, num_comp):
    pca = PCA(n_components=num_comp)
    pca.fit(points)
    transformed_points = pca.fit_transform(points)
    if stem is not None:
        transformed_stempoint = pca.transform(stem)
    else:
        transformed_stempoint = None
    #print(pca.explained_variance_ratio_)
    #print(pca.singular_values_)
    return transformed_points, transformed_stempoint, pca.components_

def discretise_and_flatten(points, labels):
    '''
    takes one plant point cloud, split into components (organs),
    rotates each leaf along the broadest spread with the principal components, maintaining 3 dimensions
    then centers and scales each leaf individually, discretises into a 256x256 depth map with a depth average at each pixel
    returns an array of unraveled depth maps, one for each leaf of the plant
    '''
    organs = util.split_into_organs(points, labels)
    stem = organs[1][0]

    leaves = []
    instance_labels_lst = []
    for organ in organs[2:]: # this takes every organ excluding the stem, which is the first

    ## TODO: Find nearest stem point to this leaf
        stem_point = None
        rotated_organ, rotated_stem_point, principal_components = get_pca(organ[0], stem_point, 3) #rotated to align with the 3 eigenvectors

        import pdb; pdb.set_trace()
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

def plot_data_examples(train_set, eigenleaves, sorted_data, sorted_labels, im_shape):
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

def split_train_and_test(data, labels):
    '''
    takes an array of unravelled depth maps of size (examples x 65536)
    Creates a training set, and an out-of-sample test set with several instances of one leaf that is not in the training data,
    and one in-sample test set featuring new instances of leaves that are already featured in the training data at a different time step
    '''
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

    return train_set, train_labels, in_data_test_set, in_data_test_labels, out_data_test_set


def pca_across_discretised_leaves(data, labels):
    '''
    performs pca to indetify eigenleaves and the individual linear combination of them for each input
    Adapted from https://machinelearningmastery.com/face-recognition-using-principal-component-analysis/
    '''

    train_set, train_labels, in_data_test_set ,__, out_data_test_set = split_train_and_test(data, labels)
    pca_trained = PCA().fit(train_set)

    return pca_trained, train_set, in_data_test_set, out_data_test_set

def test_compression_loss(pca, train_set, in_set, out_set):
    all_eigenleaves = pca.components_
    query = np.concatenate((in_set, out_set)) # all test cases in one test

    results = []
    for i in range(all_eigenleaves.shape[0], 0, -1):
        query_weight = all_eigenleaves[:i] @ (query - pca.mean_).T # compress
        reprojection = query_weight.T @ all_eigenleaves[:i] + pca.mean_ # decompress

        # compare query to reprojection (compressed and decompressed version)
        rmse = np.sqrt(((query - reprojection)**2).mean(axis=1)) # Root Mean Squared Error (pixelwise) for each test example
        results.append(rmse.mean())

        #plot_reprojection(query, reprojection)
    import pdb; pdb.set_trace()
    plt.plot(range(all_eigenleaves.shape[0], 0, -1), results)
    plt.ylabel('Mean RMSE')
    plt.xlabel('Number of principal components')
    plt.show()



def plot_reprojection(target, projcetion):
    fig, axes = plt.subplots(6,6,sharex=True,sharey=True,figsize=(8,10))
    im_shape = (256,256)
    for i in range(18):
        axes[i//3][(i%3)*2].imshow(target[i].reshape(im_shape), cmap="gray")
        axes[i//3][(i%3)*2 + 1].imshow(projcetion[i].reshape(im_shape), cmap="gray")
    plt.show()

if __name__== "__main__":
    data_directory = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D')

    data, labels, ids = stack_Tomato_leaves(data_directory)
    pca_trained, train_set, in_set, out_set = pca_across_discretised_leaves(data, labels)
    test_compression_loss(pca_trained, train_set, in_set, out_set)
