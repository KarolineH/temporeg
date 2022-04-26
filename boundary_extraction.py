import bpy
import os
import sys

def open_and_split(in_file, out_file_full, out_file_main_only):
    cleanup()
    try:
        imported_object = bpy.ops.import_mesh.ply(filepath=in_file)
        input_mesh = bpy.context.selected_objects[0] #Fix variable for the selected object
    except:
        print('No such file %s' %in_file)

    # get the full outline
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.region_to_loop()
    bpy.ops.mesh.separate(type='SELECTED') # separate the loop from the rest of the mesh
    outline = bpy.context.selected_objects[1]
    outline.select_set(True) # fix a variable for the outline
    bpy.context.view_layer.objects.active = outline
    export_outline(outline, out_file_full)

    # clean up and reduce small non-connected outline elements
    bpy.data.objects.remove(input_mesh, do_unlink=True) # delete the original mesh object, only keep boundaries
    bpy.ops.mesh.separate(type='LOOSE') # sperate into connected outline components
    bpy.ops.object.select_all(action='DESELECT') # deselect all objects
    all_objects = list(bpy.data.objects)
    sizes = [len(obj.data.vertices) for obj in all_objects] #find the size of each connected component
    # take all object for which the size is equal to the max
    largest_components = [all_objects[index] for index in range(len(all_objects)) if sizes[index] == max(sizes)]
    bpy.ops.object.select_all(action='DESELECT') # deselect all objects
    for obj in largest_components:
        obj.select_set(True)
    main_component = largest_components[0]
    bpy.context.view_layer.objects.active = main_component
    try:
        bpy.ops.object.join()
    except:
        pass
    export_outline(main_component, out_file_main_only)

def cleanup():
    bpy.ops.wm.read_factory_settings()

    bpy.ops.object.select_all(action='DESELECT') # deselect all objects
    all_objects = list(bpy.data.objects)
    for obj in all_objects:
        obj.select_set(True)
        bpy.ops.object.delete()

def export_outline(outline, out_file):
    bpy.ops.object.select_all(action='DESELECT') # deselect all objects
    outline.select_set(True)
    bpy.ops.export_scene.obj(filepath=out_file, use_selection=True, use_materials=False)

if __name__ == "__main__":
    in_dir = sys.argv[-2]
    out_dir = sys.argv[-1]

    leaf_meshes = os.listdir(in_dir)
    leaf_meshes.sort()
    for file in leaf_meshes:
        in_file = os.path.join(in_dir, file)
        out_file_full = os.path.join(out_dir, file.split('.')[0]+ '_full' + '.obj')
        out_file_main_only = os.path.join(out_dir, file.split('.')[0]+ '_main'+ '.obj')
        open_and_split(in_file, out_file_full, out_file_main_only)
