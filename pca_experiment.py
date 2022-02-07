import os
import pheno4d_util as util
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def pca(points, labels):
    scaler = MinMaxScaler()
    scaled_points = scaler.fit_transform(points)

    #util.compare_visual(points, labels, scaled_points, labels)

    pca = PCA(n_components=3)
    pca.fit(points)
    reduced_dim = pca.fit_transform(points)
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)

    plt.plot(reduced_dim[:,0], reduced_dim[:,1], 'o', color='black');

    util.compare_visual(points, labels, reduced_dim, labels)



if __name__== "__main__":
    data_directory = os.path.join('/home', 'karolineheiwolt','workspace', 'data', 'Pheno4D')
    all_files, annotated_files = util.get_file_locations(data_directory)
    points,labels = util.open_file(annotated_files[0][0])

    #util.draw_cloud(points, labels)
    pca(points, labels)
