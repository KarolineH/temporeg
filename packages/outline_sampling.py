import os
import numpy as np

def sampling_pca_input(in_dir, out_dir, normalise=False, n=200):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    outlines = os.listdir(in_dir)
    file_names = [file.split('.')[0] + '.npy' for file in outlines if 'main' in file]
    cleaned_outlines = [os.path.join(in_dir, file) for file in outlines if 'main' in file]

    for i,file in enumerate(cleaned_outlines):
        vertices, edges = read_outlines(file)
        if vertices is not None:
            loop = order_outline(vertices, edges)
            if loop is not None:
                points, normalised_points = uniformly_sample_loop(loop, n)

                # save to file
                # note that all outlines are in centroid leaf coordinate frame
                if normalise:
                    save_outline_to_file(normalised_points, file_names[i], out_dir)
                else:
                    save_outline_to_file(points, file_names[i], out_dir)

def read_outlines(file):
    try:
        content = open(file, 'r')
    except:
        print('File not available: %s' %file)
        return None, None
    lines = content.readlines()
    vertices = np.array([np.array(info.split('\n')[0].split(' ')[1:], dtype='<f8') for info in lines if info[0] == 'v'])
    vertices[:,[2, 1]] = vertices[:,[1, 2]]
    vertices[:,1] = -vertices[:,1]
    edges = np.array([np.array(info.split('\n')[0].split(' ')[1:], dtype='int') for info in lines if info[0] == 'l'])
    edges = edges -1
    return vertices, edges

def order_outline(vertices, edges):
    # Sort the vertices in order of the loop as defined by their edges
    loop = None
    sort_edges = []
    #starting_idx = np.argmin(np.linalg.norm(vertices, axis=1)) # start with the point closest to 0 which is the leaf centroid
    ''' Starting point is the highest point in y-direction'''
    #south_most_points = np.where(vertices[:,1] == vertices[:,1].min())
    tip_points = np.where(vertices[:,1] == vertices[:,1].max())

    if len(tip_points) > 1:
        starting_idx = int(tip_points[np.argmin(vertices[tip_points,0])]) # lowest in x-direction too if there are multiple
    else:
        starting_idx = int(tip_points[0])

    doneidxs = [starting_idx]
    curridx = starting_idx
    unique, counts = np.unique(edges[:,:], return_counts=True)
    if not np.any(np.greater(counts,2)):
        while (len(doneidxs) != len(vertices)):
            for e in edges:
                if e[0]==curridx or e[1]==curridx:
                    if not e[0] in doneidxs:
                        toidx= e[0]
                        break;
                    else:
                        if not e[1] in doneidxs:
                            toidx= e[1]
                            break;
            sort_edges.append([curridx, toidx])
            doneidxs.append(toidx)
            curridx = toidx
        if not ([sort_edges[0][0], sort_edges[-1][-1]] in edges or [sort_edges[-1][-1], sort_edges[0][0]] in edges):
            print('Warning: Edge loop does not close')
        sort_edges.append([sort_edges[-1][-1], sort_edges[0][0]])
        vertex_order = np.array(sort_edges)[:,0]
        loop = vertices[vertex_order,:]
        #loop = loop - loop[0,:] # Transform all points with respect to the first point at [0,0,0]
    return loop

def uniformly_sample_loop(loop, n=200):
    shifted_loop = np.append(loop, [loop[0]], axis=0)
    shifted_loop = np.delete(shifted_loop, (0), axis=0)

    # Check if the outline is in clockwise order, otherwise flip
    clockwise = sum((shifted_loop[:,0]-loop[:,0]) * (shifted_loop[:,1]+loop[:,1])) > 0
    if not clockwise:
        loop = np.flip(loop, axis = 0)
        shifted_loop = np.append(loop, [loop[0]], axis=0)
        shifted_loop = np.delete(shifted_loop, (0), axis=0)

    edge_vectors = shifted_loop-loop # vectors from point to point along the loop
    vector_length = np.linalg.norm(edge_vectors, axis=1)
    unit_vectors = np.divide(edge_vectors, np.transpose([vector_length]))

    cumulative_edges = np.cumsum(vector_length) #get the cumulative distance from start to each point
    total_length = cumulative_edges[-1]

    #sample = np.sort(np.random.uniform(0,1,n)*total_length)
    sample = np.linspace(0, total_length, num=n, endpoint=False)

    nearest_lower_vertex = np.searchsorted(cumulative_edges, sample)
    extended_cumulative_edges = np.insert(cumulative_edges, 0, 0)

    remainder = sample - extended_cumulative_edges[nearest_lower_vertex]
    remainder_vectors = unit_vectors[nearest_lower_vertex] * remainder[:, None]
    points = loop[nearest_lower_vertex] + remainder_vectors
    normalised_points = points/total_length
    return points, normalised_points

def save_outline_to_file(points, file_name, out_dir):
    file = os.path.join(out_dir, file_name)
    np.save(file, points)
    return

if __name__== "__main__":
    in_dir = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D', '_processed', 'outline')
    out_dir = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D', '_processed', 'pca_input')
    sampling_pca_input(in_dir, out_dir, normalise=False)
