import os
import open3d as o3d
import numpy as np
import re

# load point cloud (.ply)
def read_ply_cloud(file):
    try:
        pcd = o3d.io.read_point_cloud(file)
    except:
        print('File not available: %s' %file)
        return
    return pcd

def read_mesh(file):
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

def assemble_geometries(dir, plant_nr, step_nr, leaf_nr):
    pc_file = os.path.join(dir, 'aligned', 'plant' + str(plant_nr) + '_step' + str(step_nr) + '_leaf' + str(leaf_nr) + '.ply')
    pcd = read_ply_cloud(pc_file)
    x_offset = -1.2 * (pcd.get_max_bound()[0] - pcd.get_min_bound()[0])
    if pcd is not None:
        pcd.translate((x_offset,0,0))

    mesh_file = os.path.join(dir, 'meshed', 'plant' + str(plant_nr) + '_step' + str(step_nr) + '_leaf' + str(leaf_nr) + '.ply')
    mesh = read_mesh(mesh_file)
    mesh.compute_vertex_normals()

    outline_file = os.path.join(dir, 'outline', 'plant' + str(plant_nr) + '_step' + str(step_nr) + '_leaf' + str(leaf_nr) + '_full.obj')
    outline_filtered_file = os.path.join(dir, 'outline', 'plant' + str(plant_nr) + '_step' + str(step_nr) + '_leaf' + str(leaf_nr) + '_main.obj')
    outline_full = read_obj_with_no_faces(outline_file)
    outline_filtered = read_obj_with_no_faces(outline_filtered_file)

    if outline_full is not None:
        outline_full.translate((-x_offset,0,0))
    if outline_filtered is not None:
        outline_filtered.translate((-2*x_offset,0,0))

    return [pcd, mesh, outline_full, outline_filtered]

def draw(geometries, file):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=file)
    for geo in geometries:
        try:
            vis.add_geometry(geo)
        except:
            pass
    vis.run()
    vis.destroy_window()

# TODO:
# Show leaves of the same plant
# or instances of the same leaf over time together

if __name__== "__main__":

    plant_nr = 0
    step_nr = 0
    leaf_nr = 2

    data_directory = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D', '_processed')
    all_leaves = os.listdir(os.path.join(data_directory, 'aligned'))

    for file in all_leaves:
        numbers = np.array(re.findall(r'\d+', file), dtype='int')
        geometries = assemble_geometries(data_directory, numbers[0], numbers[1], numbers[2])
        draw(geometries, file)
