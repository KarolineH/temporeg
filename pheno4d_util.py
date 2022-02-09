import os
import numpy as np
from palettable.cartocolors.qualitative import Prism_10
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import open3d.visualization.gui as gui
import open3d as o3d

#outpth = os.path.join('/media', 'user','Data','karo','datasets', 'Pheno4D', 'karo_pickles')

def get_file_locations(dir):
    '''
    Finds all files in the Pheno4D directory
    Sorts them into lists with and without annotations
    '''

    plants = os.listdir(dir)
    plants.sort()
    maize_plants = [plant for plant in plants if 'Maize' in plant]
    tomato_plants = [plant for plant in plants if 'Tomato' in plant]

    all_locations = []
    all_annotated_locations = []

    counter = 0
    for set in [maize_plants,tomato_plants]:
        for i,plant in enumerate(set):
            records = os.listdir(os.path.join(dir, plant))
            records.sort()
            annotated_records = [record for record in records if '_a' in record]

            file_paths = [os.path.join(dir, plant, rec) for rec in records]
            annotated_file_paths = [os.path.join(dir, plant, rec) for rec in annotated_records]

            all_locations.append(file_paths)
            all_annotated_locations.append(annotated_file_paths)

            counter+= len(file_paths)
    print("found a total of %i point clouds" % counter)
    return all_locations, all_annotated_locations

def open_file(path):
    '''Read the point cloud and labels from the txt file'''

    print('Opening file %s'%(path))
    file = open(path,"r")
    content = file.read()
    lines = content.split('\n') # split into a list single rows
    if lines[-1] == '': # trim the last row off if it is empty
        lines = lines[:-1]
    raw = lines
    coordinates = np.array([[float(entry)for entry in line.split(' ')[:3]] for line in raw])
    instance_labels = np.array([[float(entry)for entry in line.split(' ')[3:]] for line in raw])
    return coordinates,instance_labels

def draw_cloud(cloud, labels, draw=True):
    '''
    Visualises a single point cloud
    input: numpy array
    '''
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)

    try:
        pcd = colour_by_labels(pcd, labels)
    except:
        print("Failed to apply colour by labels")
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    if draw ==True:
        vis.run()
        #vis.destroy_window()
    return vis

def colour_by_labels(pcd,labels):
    colours = np.array(Prism_10.mpl_colors)
    colour_array = colours[labels[:,0].astype(int)]
    pcd.colors = o3d.utility.Vector3dVector(colour_array)
    return pcd

def compare_visual(cloud1, labels1, cloud2, labels2):
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(cloud1)
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(cloud2)

    try:
        pcd1 = colour_by_labels(pcd1, labels1)
    except:
        print("Failed to apply colour by labels")

    try:
        pcd2 = colour_by_labels(pcd2, labels2)
    except:
        print("Failed to apply colour by labels")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name='left', width=600, height=540, left=0, top=0)
    vis.add_geometry(pcd1)
    vis2 = o3d.visualization.VisualizerWithEditing()
    vis2.create_window(window_name='right', width=600, height=540, left=800, top=0)
    vis2.add_geometry(pcd2)

    while True:
        vis.update_geometry(pcd1)
        if not vis.poll_events():
            break
        vis.update_renderer()

        vis2.update_geometry(pcd2)
        if not vis2.poll_events():
            break
        vis2.update_renderer()

    vis.destroy_window()
    vis2.destroy_window()

if __name__ == "__main__":
    data_directory = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D')
    all_files, annotated_files = get_file_locations(data_directory)
    points,labels = open_file(annotated_files[0][0])
    draw_cloud(points, labels)
