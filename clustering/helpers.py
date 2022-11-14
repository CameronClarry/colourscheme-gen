import matplotlib.pyplot as plt
import numpy as np

from clustering.algorithms import dbscan, kmeans, kmeans_separation, kdist

# http://www.easyrgb.com/en/math.php
# Converts RGB to XYZ
def rgb_to_xyz(r, g, b):
    r = r/255
    g = g/255
    b = b/255

    r = r**2.19921875
    g = g**2.19921875
    b = b**2.19921875

    r = r*100
    g = g*100
    b = b*100

    x = r * 0.57667 + g * 0.18555 + b * 0.18819
    y = r * 0.29738 + g * 0.62735 + b * 0.07527
    z = r * 0.02703 + g * 0.07069 + b * 0.99110

    return (x, y, z)

# Converts XYZ to RGB
def xyz_to_rgb(x, y, z):
    x = x/100
    y = y/100
    z = z/100
    
    r = x*2.04137 + y*(-0.56495) + z*(-0.34469)
    g = x*(-0.96927) + y*1.87601 + z*0.04156
    b = x*0.01345 + y*(-0.11839) + z*1.01541
    
    r = r**(1/2.19921875)
    g = g**(1/2.19921875)
    b = b**(1/2.19921875)
    
    r = r*255
    g = g*255
    b = b*255
    
    return r, g, b

# Converts XYZ to L*ab
def xyz_to_lab(x, y, z):
    # sRGB/aRGB (D65)
    reference_x = 94.811
    reference_y = 100.000
    reference_z = 107.304

    x = x/reference_x
    y = y/reference_y
    z = z/reference_z

    mask = x > 0.008856
    x[mask] = x[mask]**(1/3)
    x[~mask] = 7.787*x[~mask] + 16/116

    mask = y > 0.008856
    y[mask] = y[mask]**(1/3)
    y[~mask] = 7.787*y[~mask] + 16/116

    mask = z > 0.008856
    z[mask] = z[mask]**(1/3)
    z[~mask] = 7.787*z[~mask] + 16/116

    L = 116*y - 16
    a = 500*(x - y)
    b = 200*(y - z)

    return (L, a, b)

# Converts L*ab to XYZ
def lab_to_xyz(L, a, b):
    # sRGB/aRGB (D65)
    reference_x = 94.811
    reference_y = 100.000
    reference_z = 107.304
    
    y = (L+16)/116
    x = a/500 + y
    z = y - b/200
    
    mask = x**3 > 0.008856
    x[mask] = x[mask]**3
    x[~mask] = (x[~mask] - 16/116)/7.787
    
    mask = y**3 > 0.008856
    y[mask] = y[mask]**3
    y[~mask] = (y[~mask] - 16/116)/7.787
    
    mask = z**3 > 0.008856
    z[mask] = z[mask]**3
    z[~mask] = (z[~mask] - 16/116)/7.787

    x = x*reference_x
    y = y*reference_y
    z = z*reference_z
    
    return x, y, z


def cluster_plot(x, labels, n_clusters, filename=None):
    fig, axs = plt.subplots(1,2)
    for i in range(-2, n_clusters):
        mask = labels == i
        axs[0].scatter(x[mask][:,1], x[mask][:,2], label=i)
        axs[1].scatter(x[mask][:,1], x[mask][:,0], label=i)
    
    if filename is not None:
        fig.savefig(filename, format='pdf')

def cluster_info(x, labels, counts):
    # Find the size of each cluster and sort them
    _labels, _counts = np.unique(labels, return_counts=True)
    sizes = np.zeros(_labels.size)
    for i, label in enumerate(_labels):
        sizes[i] = np.sum(counts[labels == label])
    sort_indices = np.argsort(sizes)
    
    counts_dict = {}
    for label in _labels:
        counts_dict[label] = np.sum(counts[labels == label])
    
    sorted_labels = _labels[sort_indices]
    sorted_sizes = sizes[sort_indices]
    
    return sorted_labels[sorted_labels >= 0], sorted_sizes[sorted_labels >= 0]

# Create a dummy set of data and cluster using various algorithms
def cluster_test(n_clusters):
    # create our own data
    means = []
    sigmas = []
    points_arrs = []
    for i in range(10):
        points_x = np.random.normal(np.random.uniform(-40,40,1), np.random.uniform(1,10, 1), 200)
        points_y = np.random.normal(np.random.uniform(-40,40,1), np.random.uniform(1,10, 1), 200)
        points_z = np.random.normal(np.random.uniform(-40,40,1), np.random.uniform(1,10, 1), 200)
        points_arrs.append(np.stack([points_x, points_y, points_z], axis=1))

    points = np.concatenate(points_arrs, axis=0)

    # Clustering using DBSCAN
    min_points = 6
    dists = kdist(points, min_points)
    epsilon = np.percentile(dists, 80)
    labels, n_clusters = dbscan(points, np.ones(points.shape[0]), epsilon, min_points)
    cluster_plot(points, labels, n_clusters, 'random-dbscan.pdf')

    # Clustering using k-means
    n_clusters = 10
    labels, n_clusters = kmeans(points, np.ones(points.shape[0]), n_clusters)
    cluster_plot(points, labels, n_clusters, 'random-kmeans.pdf')

# Perform clustering on points in L*ab space and converts results to RGB
# TODO change this to take a string specifying the clustering method
def kmeans_colours(x, counts, n_colours, filename=None, separation=None, repulsion=None):
    if separation is None:
        labels, n_clusters, means = kmeans(x, counts, n_colours, repulsion)
    else:
        labels, n_clusters, means = kmeans_separation(x, counts, n_colours, separation)

    if filename is not None:
        cluster_plot(x, labels, n_clusters, filename)

    # Sort by luminance, increasing
    sort_indices = np.argsort(means[:,0])
    means = means[sort_indices,:]
    r, g, b = xyz_to_rgb(*lab_to_xyz(means[:,0], means[:,1], means[:,2]))
    colours = np.stack([r, g, b], axis=1)
    return colours
