import numpy as np
import os
import packages.pheno4d_util as util
from sklearn.decomposition import PCA

def fit_soil_plane(plant_cloud, labels):
    # takes a full plant point cloud
    soil_cloud = plant_cloud[(labels.astype(int).flatten()==0),:]
    centroid = np.mean(soil_cloud, axis=0)

    # Translate the pointcloud to its centroid
    pointcloud_0 = soil_cloud - centroid

    # find the plane and its normal vector via pca
    pca = PCA(n_components=3)
    pca.fit(pointcloud_0)
    vector_1, vector_2, normal = pca.components_

    # normalise the normal vector (to unit length)
    n = normal / np.linalg.norm(normal)
    return n, centroid

def find_closest_stem_point(plant_cloud, labels, normal, soil_centroid):
    # Adapted from https://stackoverflow.com/questions/55189333/how-to-get-distance-from-point-to-plane-in-3d
    # takes a full plant point cloud
    stem_cloud = plant_cloud[(labels.astype(int).flatten()==1),:]
    # find perpendicular distance for each stem point to the soil plane
    # dot product: make vector from any point on plane to the point of interest, dot with unit normal
    dist = abs(np.dot((stem_cloud-soil_centroid), normal))
    closest_point_index = np.argmin(dist)
    emergence_point = stem_cloud[closest_point_index]
    return emergence_point

def pipeline(plant_cloud, labels):
    normal, soil_centroid = fit_soil_plane(plant_cloud, labels) # takes a full plant point cloud
    emergence_point = find_closest_stem_point(plant_cloud, labels, normal, soil_centroid)
    return emergence_point

if __name__ == "__main__":
    data_directory = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D', 'Tomato01', 'T01_0305_a.txt')
    points,labels,id = util.open_file(data_directory)

    normal, soil_centroid = fit_soil_plane(points, labels) # takes a full plant point cloud
    emergence_point = find_closest_stem_point(points, labels, normal, soil_centroid)
    print(emergence_point)
