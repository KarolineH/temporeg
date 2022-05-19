import os
import open3d as o3d
import numpy as np
import re
import leaf_encoding
import copy

# load point cloud (.ply)
def read_ply_cloud(file):
    try:
        pcd = o3d.io.read_point_cloud(file)
    except:
        print('File not available: %s' %file)
        return
    return pcd

def read_ply_mesh(file):
    try:
        mesh = o3d.io.read_triangle_mesh(file)
    except:
        print('File not available: %s' %file)
        return
    return mesh

def read_obj_with_no_faces(file):
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
    try:
        data = np.load(file)
    except:
        return None
    return data

def array_to_o3d_pointcloud(data):
    if data is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data)
        return pcd
    else:
        return None

def standardise_pc_scale(data):
    original_shape = data.shape
    mins = np.asarray([np.min(leaf, axis=0) for leaf in data])
    maxes = np.asarray([np.max(leaf, axis=0) for leaf in data])
    ranges = np.asarray([np.ptp(leaf, axis=0) for leaf in data])
    flat_data = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
    scale_factor = np.max(abs(np.concatenate((mins, maxes), axis=1)), axis=1)
    standardised = flat_data / scale_factor[:,None]
    out = standardised.reshape(original_shape)
    return out

def assemble_geometries(dir, plant_nr, step_nr, leaf_nr):
    pc_file = os.path.join(dir, 'aligned', 'plant' + str(plant_nr) + '_step' + str(step_nr) + '_leaf' + str(leaf_nr) + '.ply')
    pcd = read_ply_cloud(pc_file)
    x_offset = -1.2 * (pcd.get_max_bound()[0] - pcd.get_min_bound()[0])
    # if pcd is not None:
    #     pcd.translate((x_offset,0,0))

    mesh_file = os.path.join(dir, 'meshed', 'plant' + str(plant_nr) + '_step' + str(step_nr) + '_leaf' + str(leaf_nr) + '.ply')
    mesh = read_ply_mesh(mesh_file)
    mesh.compute_vertex_normals()

    outline_file = os.path.join(dir, 'outline', 'plant' + str(plant_nr) + '_step' + str(step_nr) + '_leaf' + str(leaf_nr) + '_full.obj')
    outline_filtered_file = os.path.join(dir, 'outline', 'plant' + str(plant_nr) + '_step' + str(step_nr) + '_leaf' + str(leaf_nr) + '_main.obj')
    outline_full = read_obj_with_no_faces(outline_file)
    outline_filtered = read_obj_with_no_faces(outline_filtered_file)
    #
    # if outline_full is not None:
    #     outline_full.translate((-x_offset,0,0))
    # if outline_filtered is not None:
    #     outline_filtered.translate((-2*x_offset,0,0))

    sampled_file = os.path.join(dir, 'pca_input', 'plant' + str(plant_nr) + '_step' + str(step_nr) + '_leaf' + str(leaf_nr) + '_main.npy')
    sampled_outline = array_to_o3d_pointcloud(read_sampled_outline(sampled_file))

    return [pcd, mesh, outline_full, outline_filtered, sampled_outline]

def draw(geometries, file, offset = False, labels=None):
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

def timestep_comparison(directory=None):
    if directory is None:
        directory = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D', '_processed', 'pca_input')
    data, names = leaf_encoding.load_inputs(directory)
    labels = leaf_encoding.get_labels(names)
    standardised = standardise_pc_scale(data)

    sorted_data = standardised[np.lexsort((labels[:,2], labels[:,1],labels[:,0])),:]
    sorted_labels = labels[np.lexsort((labels[:,2], labels[:,1],labels[:,0])),:]

    for plant in np.unique(labels[:,0]):
        timesteps = np.unique(labels[labels[:,0] == plant][:,1])
        for i,time_step in enumerate(timesteps):
            before, before_labels = leaf_encoding.select_subset(sorted_data, sorted_labels, plant_nr = plant, timestep=time_step, leaf=None)
            #after, after_labels = leaf_encoding.select_subset(data, labels, plant_nr = plant, timestep=timesteps[i+1], leaf=None)
            before_clouds = [array_to_o3d_pointcloud(outline) for outline in before]
            #after_clouds = [array_to_o3d_pointcloud(outline) for outline in after]
            draw2(before_clouds, "time step number {}".format(time_step), offset=True, labels=before_labels[:,2])

def same_leaf_across_time(directory=None):
    if directory is None:
        directory = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D', '_processed', 'pca_input')
    data, names = leaf_encoding.load_inputs(directory)
    standardised = standardise_pc_scale(data)
    labels = leaf_encoding.get_labels(names)

    sorted_data = standardised[np.lexsort((labels[:,2], labels[:,1],labels[:,0])),:]
    sorted_labels = labels[np.lexsort((labels[:,2], labels[:,1],labels[:,0])),:]

    for plant in np.unique(labels[:,0]):
        for leaf in np.unique(labels[:,2]):
            subset, sub_labels = leaf_encoding.select_subset(sorted_data, sorted_labels, plant_nr = plant, leaf=leaf)
            clouds = [array_to_o3d_pointcloud(outline) for outline in subset]
            # draw(before_clouds, 'Timestep Leaf comparison', offset = True)
            # draw(after_clouds, 'Timestep Leaf comparison', offset = True)
            draw(clouds, "leaf number {}".format(leaf), offset=True, labels=sub_labels[:,1])

# TODO:
# Show leaves of the same plant
# or instances of the same leaf over time together

if __name__== "__main__":
    #timestep_comparison()
    same_leaf_across_time()

    data_directory = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D', '_processed')
    all_leaves = os.listdir(os.path.join(data_directory, 'aligned'))

    for file in all_leaves:
        numbers = np.array(re.findall(r'\d+', file), dtype='int')
        geometries = assemble_geometries(data_directory, numbers[0], numbers[1], numbers[2])
        draw(geometries, file, offset = True)
        draw2(geometries, file, offset = True)
