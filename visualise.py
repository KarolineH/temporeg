import os
import open3d as o3d
import numpy as np
import re
import leaf_encoding
import copy
import packages.pheno4d_util as util

def read_ply_cloud(file):
    '''
    Load a point cloud from a .ply file
    '''
    try:
        pcd = o3d.io.read_point_cloud(file)
    except:
        print('File not available: %s' %file)
        return
    return pcd

def read_ply_mesh(file):
    '''
    Load a 3D mesh object from a .ply file
    '''
    try:
        mesh = o3d.io.read_triangle_mesh(file)
    except:
        print('File not available: %s' %file)
        return
    return mesh

def read_obj_with_no_faces(file):
    '''
    Load a 3D mesh object from a .obj file, which has been saved only with verteces and connecting edges,
    but does not have any faces. This usually trips up standard methods.
    '''
    #read from file
    try:
        content = open(file, 'r')
    except:
        print('File not available: %s' %file)
        return
    lines = content.readlines()
    vertices = np.array([np.array(info.split('\n')[0].split(' ')[1:], dtype='<f8') for info in lines if info[0] == 'v'])
    vertices[:,[2, 1]] = vertices[:,[1, 2]]
    vertices[:,1] = -vertices[:,1]
    edges = np.array([np.array(info.split('\n')[0].split(' ')[1:], dtype='int') for info in lines if info[0] == 'l'])
    edges = edges -1

    # make into o3D plottable object
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(vertices), lines=o3d.utility.Vector2iVector(edges))
    return line_set

def read_sampled_outline(file):
    '''
    Load our own sampled outline format from a .npy file.
    '''
    try:
        data = np.load(file)
    except:
        return None
    return data

def array_to_o3d_pointcloud(data):
    '''
    Transform a numpy array into an Open3D Point Cloud object
    '''
    if data is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data)
        return pcd
    else:
        return None

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def assemble_geometries(dir, standardise=True, visualise=True, plant_nr=0, timestep=3, leaf_nr=3):
    '''
    This loads all data necessary to later create a plot of the whole preprocessing pipeline,
    stacking multiple geometry representations into the same 3D plot.
    Includes the originap point cloud, meshed object, original and cleaned boundary objects, and the sampled outline (PCA input).
    '''
    pca_input_directory = os.path.join(dir, 'pca_input')
    PCAH, test_ds, test_labels = leaf_encoding.get_encoding(train_split=0, directory=pca_input_directory, standardise=standardise, location=False, rotation=False, scale=False, as_features=False)

    subset, subset_labels = leaf_encoding.select_subset(PCAH.training_data, PCAH.training_labels, plant_nr=plant_nr, timestep=timestep, leaf=leaf_nr)
    subset_labels = subset_labels[0]

    if subset.shape[0] > 0:
        pc_file = os.path.join(dir, 'aligned', 'plant' + str(subset_labels[0]) + '_step' + str(subset_labels[1]) + '_day' + str(subset_labels[2]) + '_leaf' + str(subset_labels[3]) + '.ply')
        pcd = read_ply_cloud(pc_file)

        mesh_file = os.path.join(dir, 'meshed', 'plant' + str(subset_labels[0]) + '_step' + str(subset_labels[1]) + '_day' + str(subset_labels[2]) + '_leaf' + str(subset_labels[3]) + '.ply')
        mesh = read_ply_mesh(mesh_file)
        mesh.compute_vertex_normals()

        outline_file = os.path.join(dir, 'outline', 'plant' + str(subset_labels[0]) + '_step' + str(subset_labels[1]) + '_day' + str(subset_labels[2]) + '_leaf' + str(subset_labels[3]) + '_full.obj')
        outline_filtered_file = os.path.join(dir, 'outline', 'plant' + str(subset_labels[0]) + '_step' + str(subset_labels[1]) + '_day' + str(subset_labels[2]) + '_leaf' + str(subset_labels[3]) + '_main.obj')
        outline_full = read_obj_with_no_faces(outline_file)
        outline_filtered = read_obj_with_no_faces(outline_filtered_file)

        sampled_file = os.path.join(dir, 'pca_input', 'plant' + str(subset_labels[0]) + '_step' + str(subset_labels[1]) + '_day' + str(subset_labels[2]) + '_leaf' + str(subset_labels[3]) + '_main.npy')
        sampled_outline = array_to_o3d_pointcloud(read_sampled_outline(sampled_file))

        geometries = [pcd, mesh, outline_full, outline_filtered, sampled_outline]
        if visualise:
            draw(geometries, 'name', offset = True)
            draw2(geometries, 'name', offset = True)

        return geometries
    else:
        print('No leaves in this subset')
        return None

def draw(geometries, file, offset = False, labels=None):
    '''
    Takes Open3D gemoetry objects and plots them using the o3D visualiser.
    '''
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=file)
    width = 0
    for i,geo in enumerate(geometries):
        geocopy = copy.deepcopy(geo)
        try:
            if offset:
                geocopy.translate((width,0,0))
                width += 1.2 * (geo.get_max_bound()[0] - geo.get_min_bound()[0])
            vis.add_geometry(geocopy)
        except:
            pass
    vis.run()
    vis.destroy_window()

def draw2(geometries, name, offset = False, labels=None):
    '''
    Takes Open3D gemoetry objects and plots them using the o3D visualization.gui.
    Version 2 with improved offset calculation.
    '''
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer(name, 1024, 768)
    vis.show_settings = True
    width = 0
    for i,geo in enumerate(geometries):
        geocopy = copy.deepcopy(geo)
        try:
            if offset:
                if i!=0:
                    width += 0.6 * (geometries[i-1].get_max_bound()[0] - geometries[i-1].get_min_bound()[0]) + 0.6 * (geometries[i].get_max_bound()[0] - geometries[i].get_min_bound()[0])
                geocopy.translate((width,0,0))
            vis.add_geometry("{}".format(i),geocopy)
            if labels is not None:
                vis.add_3d_label(geocopy.get_min_bound(),"{}".format(labels[i]))
            # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=geo.get_max_bound()[0] - geo.get_min_bound()[0], origin=geocopy.get_min_bound())
            # vis.add_geometry("box{}".format(i),mesh_frame)
        except:
            pass
    vis.reset_camera_to_default()
    app.add_window(vis)
    app.run()

def timestep_comparison(directory=None, standardise=True):
    '''
    Plots all leaf outlines of one timestep at a time, cycling through all plants and time-steps.
    '''
    if directory is None:
        directory = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D', '_processed', 'pca_input')

    PCAH, test_ds, test_labels = leaf_encoding.get_encoding(train_split=0, directory=directory, standardise=standardise, location=False, rotation=False, scale=False, as_features=False)
    labels = PCAH.training_labels

    for plant in np.unique(labels[:,0]):
        timesteps = np.unique(labels[labels[:,0] == plant][:,1])
        for i,time_step in enumerate(timesteps):
            before, before_labels = leaf_encoding.select_subset(PCAH.training_data, labels, plant_nr = plant, timestep=time_step, leaf=None)
            #after, after_labels = leaf_encoding.select_subset(data, labels, plant_nr = plant, timestep=timesteps[i+1], leaf=None)

            stacked_before, before_add_f = leaf_encoding.reshape_coordinates_and_additional_features(before, nr_coordinates=500)
            #stacked_after, after_add_f = leaf_encoding.reshape_coordinates_and_additional_features(before, nr_coordinates=500)
            before_clouds = [array_to_o3d_pointcloud(outline) for outline in stacked_before]
            #after_clouds = [array_to_o3d_pointcloud(outline) for outline in stacked_after]
            draw2(before_clouds, "time step number {}".format(time_step), offset=True, labels=before_labels[:,2])

def same_leaf_across_time(directory=None):
    '''
    Plots all leaf outlines of the same individual leaf over all time-steps side-by-side.
    Cycles over the entire data set.
    '''
    if directory is None:
        directory = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D', '_processed', 'pca_input')
    PCAH, test_ds, test_labels = leaf_encoding.get_encoding(train_split=0, directory=directory, standardise=standardise, location=False, rotation=False, scale=False, as_features=False)
    labels = PCAH.training_labels

    for plant in np.unique(labels[:,0]):
        for leaf in np.unique(labels[:,-1]):
            subset, sub_labels = leaf_encoding.select_subset(PCAH.training_data, labels, plant_nr = plant, leaf=leaf)
            stacked, add_f = leaf_encoding.reshape_coordinates_and_additional_features(subset, nr_coordinates=500)
            clouds = [array_to_o3d_pointcloud(outline) for outline in stacked]
            draw(clouds, "leaf number {}".format(leaf), offset=True, labels=sub_labels[:,1])

if __name__== "__main__":
    #timestep_comparison()
    #same_leaf_across_time()

    data_directory = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D', '_processed')
    geometries = assemble_geometries(data_directory, standardise = True, visualise=True, plant_nr=0, timestep=0, leaf_nr=3)
