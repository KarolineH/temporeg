import leaf_encoding
import os
import numpy as np

def get_organ_means(data):
    # Takes a list of 2 datasets, which are lists of coordinates and labels
    # [[[coords1],[lables1]], [[coords2],[lables2]]]
    # location w.r.t. the plant stem emergence point
    import pdb; pdb.set_trace()
    pass

def get_cost_matrix(mean_points):
    pass


def compute_assignment(cost_mat, label_set_1, label_set_2):
    assignment = linear_sum_assignment(cost_mat)
    match = (label_set_1[assignment[0]], label_set_2[assignment[1]])
    return match, np.array(list(zip(match[0],match[1])))

if __name__== "__main__":
    # Load data and fit pca
    directory = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D', '_processed', 'pca_input')
    train_ds, test_ds, train_labels, test_labels, pca, transformed = leaf_encoding.get_encoding(train_split=0, dir=directory)

    # sort
    labels = train_labels[np.lexsort((train_labels[:,2], train_labels[:,1],train_labels[:,0])),:]
    data = train_ds[np.lexsort((train_labels[:,2], train_labels[:,1],train_labels[:,0])),:]

    datasets = []
    IDs = [(0,2,None),(0,3,None)]
    for ID in IDs:
        subset, sub_labels = leaf_encoding.select_subset(data, labels, plant_nr = ID[0], timestep=ID[1], leaf=ID[2])
        datasets.append((subset,sub_labels))

    get_organ_means(datasets)
