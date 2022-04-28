import bpy
import os
import sys
import numpy as np

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
    areas = [(np.array(obj.dimensions)[0]*np.array(obj.dimensions)[1]) for obj in all_objects] #find the x-y spread area of each connected component, from its bounding box corners

    # keep only the largest (or if there are multiple of the same size == max
    largest_components = [all_objects[index] for index in range(len(all_objects)) if areas[index] == max(areas)]
    bpy.ops.object.select_all(action='DESELECT') # deselect all objects
    main_component = largest_components[0]
    if len(largest_components) > 1:
        import pdb; pdb.set_trace()
        for obj in largest_components:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = main_component
        try:
            bpy.ops.object.join()
        except:
            pass
    else:
        largest_components[0].select_set(True)


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
