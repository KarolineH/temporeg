import visualise
import os
import open3d as o3d
import packages.pheno4d_util as util
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
directory = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D', 'Tomato01')

def get_fullplant_snapshots():
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    files = sorted(os.listdir(directory))
    for i,file in enumerate(files):
        points,labels,plant_id = util.open_file(os.path.join(directory,file))
        pc_array = np.asarray(points)
        pc = visualise.array_to_o3d_pointcloud(pc_array)

        vis.add_geometry(pc)
        ctr = vis.get_view_control()
        ctr.rotate(0, -410.0) #(i*10.0, -410.0)
        vis.update_geometry(pc)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(f'images/{i}.png')
        vis.clear_geometries()

def error_animation():
    import leaf_encoding
    import visualise
    import open3d as o3d

    import leaf_matching
    import packages.pheno4d_util as putil
    import numpy as np
    import copy
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.cm as cm

    directory = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D', '_processed', 'pca_input')
    PCAH, test_ds, test_labels = leaf_encoding.get_encoding(train_split=0.754, random_split=False, directory=directory, standardise=True,  \
                                location=True, rotation=True, scale=True, as_features=False)
    data = test_ds
    labels = test_labels
    annotations = 3 # 0 for plant number, 1 for timestep, 2 for day, 3 for leaf number

    c_directory = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D', '_processed', 'transform_log')
    centroids, centroid_labels = leaf_matching.get_location_info(directory) # already sorted

    indeces = np.asarray([np.argwhere((centroid_labels == leaf).all(axis=1)) for leaf in labels]).flatten()
    centroids = centroids[indeces,:]
    centroid_labels = centroid_labels[indeces,:]
    components = 23

    for lf in range(35):
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        #ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax_3d = [ax,ax]
        #ax_3d_2 = [ax2,ax2]
        datasets = []
        centroid_lst = []
        for ID in range(2):
            subset, sub_labels = leaf_encoding.select_subset(data, labels, plant_nr = 1, timestep=lf+ID, day=None, leaf=None)
            sub_c, sub_c_labels = leaf_encoding.select_subset(centroids, centroid_labels, plant_nr = 1, timestep=lf+ID, day=None, leaf=None)
            datasets.append((subset,sub_labels))
            centroid_lst.append((sub_c,sub_c_labels))
            if subset.size == 0:
                print('WARNING: One of your selected subsets is empty!')
                break

        data_copy = copy.deepcopy(datasets)
        save_pcs = False

        for ax in ax_3d:
            ax.clear()
        # for ax in ax_3d_2:
        #     ax.clear()

        j=0
        for subset, ax in zip(data_copy, ax_3d):
            ax.grid(False)
            ax.axis('off')
            if j==0:
                col = 'grey'
            else:
                col = 'lightgrey'
            j+=1
            for leaf,label in zip(subset[0],subset[1]):
                outline = leaf_encoding.reshape_coordinates_and_additional_features(leaf)[0] # 500x3

                if save_pcs:
                    pcd = visualise.array_to_o3d_pointcloud(outline)
                    o3d.io.write_point_cloud(f"pcds/{label}.pcd", pcd)

                scatterplot = ax.scatter(xs=outline[:,0], ys=outline[:,1], zs=outline[:,2], s=1, color=col)
                start_point = ax.scatter(xs=outline[0][0], ys=outline[0][1], zs=outline[0][2], s=5, color='black')
                out = ax.text(outline[0,0], outline[0,1], outline[0,2], s=str(label[annotations]), color='black')
            visualise.set_axes_equal(ax)

        # get assignments
        odist = leaf_matching.make_fs_dist_matrix(datasets[0][0], datasets[1][0], PCAH, mahalanobis_dist = True, draw=False, components=components)
        match, match_array, o_legible_matches = leaf_matching.compute_assignment(odist, datasets[0][1], datasets[1][1])
        for match in o_legible_matches:
            before_point = datasets[0][0][np.where(datasets[0][1][:,-1] == match[0,-1]),:].flatten()
            after_point = datasets[1][0][np.where(datasets[1][1][:,-1] == match[1,-1]),:].flatten()

            if match[0,-1] == match[1,-1]:
                colour = 'blue'
            else:
                colour = 'red'
            pair_line = np.stack((before_point, after_point), axis = 0)
            ax.plot(pair_line[:,0], pair_line[:,1], pair_line[:,2], linewidth=3, c=colour)
            print(pair_line)
        #
        # j=0
        # for subset, ax in zip(centroid_lst, ax_3d_2):
        #     ax.grid(False)
        #     ax.axis('off')
        #     if j==0:
        #         col = 'grey'
        #     else:
        #         col = 'lightgrey'
        #     j+=1
        #
        #     scatterplot = ax.scatter(xs=subset[0][:,0], ys=subset[0][:,1], zs=subset[0][:,2], s=1, color=col)
        #     visualise.set_axes_equal(ax)
        #
        # # get assignments
        # cdist = leaf_matching.make_dist_matrix(centroid_lst[0][0], centroid_lst[1][0], centroids, mahalanobis_dist = False, draw=False)
        # match, match_array, c_legible_matches = leaf_matching.compute_assignment(cdist, centroid_lst[0][1], centroid_lst[1][1])
        # for match in c_legible_matches:
        #     before_point = centroid_lst[0][0][np.where(centroid_lst[0][1][:,-1] == match[0,-1]),:].flatten()
        #     after_point = centroid_lst[1][0][np.where(centroid_lst[1][1][:,-1] == match[1,-1]),:].flatten()
        #     if match[0,-1] == match[1,-1]:
        #         colour = 'blue'
        #     else:
        #         colour = 'red'
        #     pair_line = np.stack((before_point, after_point), axis = 0)
        #     ax.plot(pair_line[:,0], pair_line[:,1], pair_line[:,2], linewidth=3, c=colour)
        #     print(pair_line)
        plt.show()


if __name__ == "__main__":
    error_animation()
