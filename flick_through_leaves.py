import os
import visualise

directory = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D', '_processed', 'pca_input')
outlines = os.listdir(directory)
outlines.sort()

for leaf in outlines:
    file = os.path.join(directory, leaf)
    data = visualise.array_to_o3d_pointcloud(visualise.read_sampled_outline(file))
    visualise.draw(data, str(leaf), offset = False)
