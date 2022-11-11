import os
import visualise

directory = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D', '_processed')

parameters = [0,8,16,24]# [0,4,8,11] #[0,6,12,15] #[0,4,8,11] [0,8,16,24]
plant = parameters[0]
time = parameters[1]
day = parameters[2]
leaf = parameters[3]


pc_file = os.path.join(directory, 'aligned', 'plant' + str(plant) + '_step' + str(time) + '_day' + str(day) + '_leaf' + str(leaf) + '.ply')
pcd = visualise.read_ply_cloud(pc_file)

mesh_file = os.path.join(directory, 'meshed', 'plant' + str(plant) + '_step' + str(time) + '_day' + str(day) + '_leaf' + str(leaf) + '.ply')
mesh = visualise.read_ply_mesh(mesh_file)
mesh.compute_vertex_normals()

outline_file = os.path.join(directory, 'outline', 'plant' + str(plant) + '_step' + str(time) + '_day' + str(day) + '_leaf' + str(leaf) + '_full.obj')
outline_filtered_file = os.path.join(directory, 'outline', 'plant' + str(plant) + '_step' + str(time) + '_day' + str(day) + '_leaf' + str(leaf) + '_main.obj')
outline_full, outline_full_pc = visualise.read_obj_with_no_faces(outline_file)
outline_filtered, outline_filtered_pc = visualise.read_obj_with_no_faces(outline_filtered_file)

geometries = [pcd, mesh, (outline_full, outline_full_pc), (outline_filtered, outline_filtered_pc)]
#visualise.draw(geometries, 'name', offset = True)

for geo in geometries:
    visualise.draw(geo, 'name', offset = True)
