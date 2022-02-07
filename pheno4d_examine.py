import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import open3d as o3d
import matplotlib.pyplot as plt

#outpth = os.path.join('/media', 'user','Data','karo','datasets', 'Pheno4D', 'karo_pickles')

def get_file_locations(dir):
    plants = os.listdir(data_directory)
    plants.sort()
    maize_plants = [plant for plant in plants if 'Maize' in plant]
    tomato_plants = [plant for plant in plants if 'Tomato' in plant]

    all_locations = []
    all_annotated_locations = []

    counter = 0
    for set in [maize_plants,tomato_plants]:
        for i,plant in enumerate(set):
            records = os.listdir(os.path.join(data_directory, plant))
            records.sort()
            annotated_records = [record for record in records if '_a' in record]

            file_paths = [os.path.join(data_directory, plant, rec) for rec in records]
            annotated_file_paths = [os.path.join(data_directory, plant, rec) for rec in annotated_records]

            all_locations.append(file_paths)
            all_annotated_locations.append(annotated_file_paths)

            counter+= len(file_paths)
    print("found a total of %i point clouds" % counter)
    return all_locations, all_annotated_locations

def open_file(path):
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

def draw_cloud(pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    vis.run()
    vis.destroy_window()

def pca(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    #do I need scaling here?

    pca = PCA(n_components=2)
    pca.fit(scaled_data)
    import pdb; pdb.set_trace()

    principalComponents = pca.fit_transform(scaled_data)
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)

    plt.plot(principalComponents[:,0], principalComponents[:,1], 'o', color='black');

    import pdb; pdb.set_trace()

    return None

    #file = os.path.join(outpth, '{}plant{}.pickle'.format(rec,i))
    #pickle.dump( coordinates, open( file, "wb" ))

if __name__ == "__main__":
    data_directory = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D')
    all_files, annotated_files = get_file_locations(data_directory)

    points,labels = open_file(annotated_files[0][0])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    #draw_cloud(pcd)
    pca(points)
