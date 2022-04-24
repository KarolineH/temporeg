import cloudComPy as cc

def mesh_and_smooth(in_file, out_file, maxLength, smoothIterations=20, smoothFactor=0.2):
    cc.initCC()  # to do once before using plugins
    cloud = cc.loadPointCloud(in_file) #load point cloud from file
    mesh_a = cc.ccMesh.triangulate(cloud, cc.TRIANGULATION_TYPES.DELAUNAY_2D_AXIS_ALIGNED, updateNormals=True, maxEdgeLength=maxLength)
    mesh_a.laplacianSmooth(nbIteration=smoothIterations, factor=smoothFactor)
    cc.SaveMesh(mesh_a, out_file) #file extensions (.ma .dxf .off .stl .vtk .obj .ply .bin .fbx)

if __name__== "__main__":
    in_file = 'test_pc.ply'
    out_file = 'test_mesh.ply'
    mesh_and_smooth(in_file, out_file, maxLength = 1)
