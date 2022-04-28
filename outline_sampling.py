import os
import numpy as np

def read_outlines(files):
    for file in files:
        try:
            content = open(file, 'r')
        except:
            print('File not available: %s' %file)
            continue
        lines = content.readlines()
        vertices = np.array([np.array(info.split('\n')[0].split(' ')[1:], dtype='<f8') for info in lines if info[0] == 'v'])
        vertices[:,[2, 1]] = vertices[:,[1, 2]]
        vertices[:,1] = -vertices[:,1]
        edges = np.array([np.array(info.split('\n')[0].split(' ')[1:], dtype='int') for info in lines if info[0] == 'l'])
        edges = edges -1

        # Sort the vertices in order of the loop as defined by their edges
        sort_edges = []
        doneidxs = [edges[0][0]]
        curridx = edges[0][0]
        unique, counts = numpy.unique(edges[:,:], return_counts=True)
        occurences = dict(zip(unique, counts))

        while (len(doneidxs) != len(vertices)):
            for e in edges:
                if e[0]==curridx or e[1]==curridx:
                    #if not occurences[e[0]] == 0:
                    if not e[0] in doneidxs:
                        toidx= e[0]
                        break;
                    else:
                        #if not occurences[e[1]] == 0:
                        if not e[1] in doneidxs:
                            toidx= e[1]
                            break;
            if not curridx == toidx:
                sort_edges.append([curridx, toidx])
            occurences[toidx] -= 2
            doneidxs.append(toidx)
            curridx = toidx
        if not ([sort_edges[0][0], sort_edges[-1][-1]] in edges or [sort_edges[-1][-1], sort_edges[0][0]] in edges):
            print('Warning: Edge loop does not close')
        sort_edges.append([sort_edges[-1][-1], sort_edges[0][0]])
        vertex_order = np.array(sort_edges)[:,0]
        loop = vertices[vertex_order,:]
        import pdb; pdb.set_trace()



if __name__== "__main__":

    dir = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D', '_processed', 'outline')
    outlines = os.listdir(dir)
    outlines = ['plant4_step5_leaf10_full.obj','plant4_step5_leaf10_main.obj']
    cleaned_outlines = [os.path.join(dir, file) for file in outlines if 'main' in file]
    read_outlines(cleaned_outlines)
