import os
import leaf_encoding
import visualise
import leaf_matching
import packages.pheno4d_util as putil
import numpy as np
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

directory = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D', '_processed', 'pca_input')
PCAH, test_ds, test_labels = leaf_encoding.get_encoding(train_split=0, random_split=False, directory=directory, standardise=True,  \
                            location=True, rotation=True, scale=False, as_features=False)
data = PCAH.training_data
labels = PCAH.training_labels

IDs = [[2,5,None,None],[2,6,None,None]] # Plant, timestep, day, leafnr
annotations = 3 # 0 for plant number, 1 for timestep, 2 for day, 3 for leaf number

fig = plt.figure()
fig2 = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax2 = fig2.add_subplot(1, 1, 1, projection='3d')
fig3, ax3 = plt.subplots(1,1)
ax_3d = [ax,ax2]

components = 23
datasets = []
for ID in IDs:
    subset, sub_labels = leaf_encoding.select_subset(data, labels, plant_nr = ID[0], timestep=ID[1], day=ID[2], leaf=ID[3])
    datasets.append((subset,sub_labels))
    if subset.size == 0:
        print('WARNING: One of your selected subsets is empty!')
        break

max_color = np.max((np.max(datasets[0][1][:,3]), np.max(datasets[1][1][:,3])))
cmap = cm.get_cmap('rainbow')

data_copy = copy.deepcopy(datasets)

offset = True
show_decompressed = False
for subset, ax in zip(data_copy, ax_3d):
    ax.clear()
    ax.grid(False)
    ax.axis('off')
    colours = cmap((subset[1][:,3]/max_color))
    cum_offset = 0
    width = 0
    for leaf,label,col in zip(subset[0],subset[1],colours):

        if show_decompressed:
            ax.title.set_text('This plot shows not the original 3D outline shapes, but projected 3D outlines after PCA and subsequent decrompression')
            weights = PCAH.compress(leaf, components)
            reprojection = PCAH.decompress(weights, components) # shape (1xfeatures)
            decompressed_outline, add_f = leaf_encoding.reshape_coordinates_and_additional_features(reprojection)
            outline = np.squeeze(decompressed_outline) # Remove extra dimension 1x500x3 to 500x3
        else:
            outline = leaf_encoding.reshape_coordinates_and_additional_features(leaf)[0] # 500x3

        if offset:
            cum_offset += 1 * width
            width = (np.max(outline[:,0]) - np.min(outline[:,0]))
            cum_offset += 1 * width
            outline[:,0] += cum_offset
        scatterplot = ax.scatter(xs=outline[:,0], ys=outline[:,1], zs=outline[:,2], s=1, color=col)
        start_point = ax.scatter(xs=outline[0][0], ys=outline[0][1], zs=outline[0][2], s=10, color='black')
        out = ax.text(outline[0,0], outline[0,1], outline[0,2], s=str(label[annotations]), color='red')
    visualise.set_axes_equal(ax)

ax3.clear()
dist = leaf_matching.make_fs_dist_matrix(datasets[0][0], datasets[1][0], PCAH, mahalanobis_dist = True, draw=False, components=components)
heatplot = leaf_matching.plot_heatmap(dist, ax=ax3, show_values=False)

out = heatplot.set_xticks(range(datasets[1][1][:].shape[0]))
out = heatplot.set_yticks(range(datasets[0][1][:].shape[0]))
out = heatplot.set_xticklabels(datasets[1][1][:], rotation = 60)
out = heatplot.set_yticklabels(datasets[0][1][:])

# get assignment via  Hungarian Algorithm
match, match_array,legible_matches = leaf_matching.compute_assignment(dist, datasets[0][1], datasets[1][1])
print(legible_matches)

plt.show()
