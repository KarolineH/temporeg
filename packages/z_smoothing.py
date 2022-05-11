import os
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
import copy
import gc

def z_smoothing_operation(in_dir, out_dir, radius=1.2, smoothIterations=8, smoothFactor=0.2):
    gc.collect()

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    leaf_clouds = os.listdir(in_dir)
    done = os.listdir(out_dir)
    for leaf in leaf_clouds:
        in_file = os.path.join(in_dir, leaf)
        out_file = os.path.join(out_dir, leaf)
        if not leaf in done:
            pc = o3d.io.read_point_cloud(in_file)
            coordinates = np.asarray(pc.points)
            smoothed_pc = smoothing(coordinates, radius, smoothIterations, smoothFactor)

            x_offset = -1.2 * (pc.get_max_bound()[0] - pc.get_min_bound()[0])
            pointSet = o3d.geometry.PointCloud()
            pointSet.points = o3d.utility.Vector3dVector(smoothed_pc)
            #pointSet.translate((x_offset,0,0))
            #o3d.visualization.draw_geometries([pc, pointSet])
            o3d.io.write_point_cloud(out_file, pointSet)
            print('Saved z-smoothed point cloud %s' %out_file)
            pointSet = None
            smoothed_pc = None
            coordinates = None

def smoothing(cloud, radius=1.2, smoothIterations=8, smoothFactor=0.2):
    smoothed_cloud = copy.deepcopy(cloud)
    cloud = None
    for i in range(smoothIterations):
        tree = KDTree(smoothed_cloud, leaf_size=2)
        ind = tree.query_radius(smoothed_cloud, r=radius) # find each point's neighbourhoods by radius
        z_push_vectors = np.asarray([np.mean((smoothed_cloud[query,2]-smoothed_cloud[j,2])) for j,query in enumerate(ind)])
        smoothed_z = smoothed_cloud[:,2] + (smoothFactor*z_push_vectors)
        smoothed_cloud[:,2] = smoothed_z

    tree = None
    return smoothed_cloud

if __name__ == "__main__":
        in_dir= os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D', '_processed', 'quick_test_set')
        out_dir = os.path.join(in_dir, 'smoothed')
        loopover(in_dir, out_dir)
