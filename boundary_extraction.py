import bpy
import os
        
def open_and_split(in_file):
    cleanup()
    try:
        imported_object = bpy.ops.import_mesh.ply(filepath=in_file)
        input_mesh = bpy.context.selected_objects[0] #Fix variable for the selected object
    except:
        print('No such file %s' %in_file)
    
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.region_to_loop()
    bpy.ops.mesh.separate(type='SELECTED') # separate the loop from the rest of the mesh 
    outline = bpy.context.selected_objects[1]
    outline.select_set(True) # fix a variable for the outline
    
    bpy.data.objects.remove(input_mesh, do_unlink=True)
    
    bpy.ops.mesh.separate(type='LOOSE')
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
    
    #VERIFY that this worked
    
    # export the main_component to ply file
    

def cleanup():
    bpy.ops.object.select_all(action='DESELECT') # deselect all objects
    all_objects = list(bpy.data.objects)
    for obj in all_objects:
        obj.select_set(True)
        bpy.ops.object.delete()

def export_outline(outline, out_file):
    # TO DO
    pass

if __name__ == "__main__":
    file = os.path.join(os.path.expanduser('~'), 'workspace', 'dev', 'temporeg', 'test_mesh.ply')
    open_and_split(file) #Load plant objects in a loop and perform operations on them