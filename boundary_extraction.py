import cloudComPy as cc
cc.initCC()  # to do once before using plugins
cloud = cc.loadPointCloud("test_pc.ply") #load point cloud from file
mesh_a = cc.ccMesh.triangulate(cloud, cc.TRIANGULATION_TYPES.DELAUNAY_2D_AXIS_ALIGNED, updateNormals=True, maxEdgeLength=0)
mesh_a.size()
mesh_a.laplacianSmooth(nbIteration=20, factor=0.2)
