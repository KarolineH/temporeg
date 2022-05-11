import cloudComPy as cc
import os
import numpy as np
import open3d as o3d

def mesh_and_smooth(in_dir, out_dir, maxLength, smoothIterations=20, smoothFactor=0.2):

    cc.initCC()  # to do once before using plugins
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    leaf_clouds = os.listdir(in_dir)
    for leaf in leaf_clouds:
        in_file = os.path.join(in_dir, leaf)
        out_file = os.path.join(out_dir, leaf)
        cloud = cc.loadPointCloud(in_file) #load point cloud from file

        mesh_obj = cc.ccMesh.triangulate(cloud, cc.TRIANGULATION_TYPES.DELAUNAY_2D_AXIS_ALIGNED, updateNormals=True, maxEdgeLength=maxLength)
        mesh_obj.laplacianSmooth(nbIteration=smoothIterations, factor=smoothFactor)
        cc.SaveMesh(mesh_obj, out_file) #file extensions (.ma .dxf .off .stl .vtk .obj .ply .bin .fbx)

if __name__ == "__main__":
        in_dir= os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D', '_processed', 'aligned')
        out_dir = '.'
        mesh_and_smooth(in_dir, out_dir, maxLength = 1)
