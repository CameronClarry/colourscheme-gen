import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Use argparse to get the image file, repulsion, and output format
parser = argparse.ArgumentParser(description='Generate a colourscheme based on an image')
parser.add_argument('path', help='path to the image that the colourscheme will be generated from', type=str)
parser.add_argument('-r', '--repulsion', help='force that means repel each other with', default=None, type=float)
parser.add_argument('--xresources', dest='xresources', help='output in xresources format', default=False, action='store_true')
parser.add_argument('--seed', help='seed for initial kmeans centroids, set to 0 for random', default=0, type=int)
parser.add_argument('-s', '--separation', help='force that means repel each other with', default=None, type=float)

args = parser.parse_args()
img_path = args.path
repulsion = args.repulsion
use_xresources = args.xresources
separation = args.separation
seed = args.seed


if seed != 0:
    np.random.seed(seed)

# Shrink the image to nxn
img = Image.open(img_path)
img.thumbnail((300, 300))
# Get colour values, and convert to CIELAB

# http://www.easyrgb.com/en/math.php
# Takes normalized rgb, returns xyz
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

def dist(x, point):
    return np.sum((x-point)**2, axis=1)

def kmeans(x, counts, n_clusters, repulsion=None):
    # Initialize
    sample_indices = np.random.choice(x.shape[0], n_clusters)
    means = x[sample_indices,:]
    
    def assignment(means):
        # Calculate distances
        dist_arrs = []
        for i in range(n_clusters):
            dist_arrs.append(dist(x, means[i]))
        
        dist_arr = np.stack(dist_arrs, axis=1)
        
        # Take the argmin along the mean axis
        labels = np.argmin(dist_arr, axis=1)
        
        return labels
    
    def update(labels):
        means = []
        for i in range(n_clusters):
            means.append(np.average(x[labels == i], weights=counts[labels == i], axis=0))

        means = np.stack(means, axis=0)

        # Repulsion step
        if repulsion is not None:
            forces = []
            for i in range(n_clusters):
                mean = means[i]
                force = np.zeros(mean.shape)
                for j in range(n_clusters):
                    if i == j:
                        continue
                    force = force + repulsion*(mean-means[j])/(np.sqrt(np.sum((mean-means[j])**2)))**3
                forces.append(force)

            means = means + np.array(forces)
                
        
        return means
    
    for i in range(20):
        labels = assignment(means)
        new_means = update(labels)
        changes = np.sqrt(np.sum((new_means-means)**2, axis=1))
        means = new_means
    
    return labels, n_clusters, means

def kmeans_separation(x, counts, n_clusters, s):
    # Initialize
    sample_indices = np.random.choice(x.shape[0], n_clusters)
    means = x[sample_indices,:]
    
    def assignment(means):
        # Calculate distances
        dist_arrs = []
        for i in range(n_clusters):
            dist_arrs.append(dist(x, means[i]))
        
        dist_arr = np.stack(dist_arrs, axis=1)
        
        # Take the argmin along the mean axis
        labels = np.argmin(dist_arr, axis=1)
        
        return labels
    
    def update(labels, oldMeans):
        # Calculate special weights: difference between the closest/second closest centroid distances,
        # divided by the distance between those centroids
        dist_arrs = []
        for i in range(n_clusters):
            dist_arrs.append(dist(x, oldMeans[i]))
        
        dist_arr = np.stack(dist_arrs, axis=1)

        if n_clusters > 2:
            closest = np.argpartition(dist_arr, 2, axis=1)
            centroid_separations = dist(oldMeans[closest[:,0]], oldMeans[closest[:,1]])
            weights = (dist_arr[np.arange(x.shape[0]),closest[:,0]] - dist_arr[np.arange(x.shape[0]),closest[:,1]])/centroid_separations
        else:
            weights = np.ones(x.shape[0])

        means = []
        for i in range(n_clusters):
            mask = labels == i
            means.append(np.average(x[mask], weights=counts[mask]*weights[mask]**s, axis=0))

        means = np.stack(means, axis=0)
        
        return means
    
    for i in range(20):
        labels = assignment(means)
        new_means = update(labels, means)
        changes = np.sqrt(np.sum((new_means-means)**2, axis=1))
        means = new_means
    
    return labels, n_clusters, means

def dbscan(x, counts, epsilon, min_points):
    print('Starting DBSCAN clustering')
    print(epsilon)
    print(x.shape)
    print(counts.shape)
    clusters = 0
    labels = np.repeat(-1, x.shape[0])
    for i in range(x.shape[0]):
        #print(i)
        if labels[i] != -1:
            continue
        
        neighbours = dist(x, x[i]) < epsilon
        neighbours = np.where(neighbours)[0]
        if np.sum(counts[neighbours]) < min_points:
            #print('Found noise points')
            #print(epsilon)
            #print(neighbours.size)
            #print(dist(x, x[i]))
            labels[i] = -2
            continue
        
        labels[i] = clusters
        clusters = clusters + 1
        
        neighbours = neighbours.tolist()
        
        while len(neighbours) > 0:
            j = neighbours.pop(0)
            
            if labels[j] == -2:
                labels[j] = labels[i]
            
            if labels[j] != -1:
                continue
            
            labels[j] = labels[i]
            
            # find new neighbours
            new_neighbours = dist(x, x[j]) < epsilon
            new_neighbours = np.where(new_neighbours)[0]
            if np.sum(counts[neighbours]) >= min_points:
                new_neighbours = new_neighbours[labels[new_neighbours] < 0]
                neighbours = neighbours + new_neighbours.tolist()
    
    return labels, clusters

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

def kdist(x, k, filename=None):
    fig, ax = plt.subplots(1)
    kdist_arr = np.zeros(x.shape[0])
    for i in range(0, x.shape[0]):
        distances = dist(x, x[i])
        kdist_arr[i] = np.partition(distances, k)[k-1]
    
    if filename is not None:
        ax.hist(kdist_arr, np.linspace(0.0, np.percentile(kdist_arr, 95), 100))
        fig.savefig(filename, format='pdf')
    
    return kdist_arr

def kmeans_colours(x, counts, n_colours, filename=None):
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


counts = []
colours = []
width, height = img.size
for count, colour in img.getcolors(width*height):
    counts.append(count)
    colours.append(colour)

colours = np.array(colours)
counts = np.array(counts)
r = colours[:,0]
g = colours[:,1]
b = colours[:,2]
L, a, b = xyz_to_lab(*rgb_to_xyz(r, g, b))

# Split data into categories based on luminosity percentiles
mask_background = L < np.percentile(L, 10)
mask_background2 = np.logical_and(L < np.percentile(L, 40), L > np.percentile(L, 20))
mask_foreground = L > np.percentile(L, 90)
mask_accent = np.logical_and(L < np.percentile(L, 80), L > np.percentile(L, 50))


# Perform clustering on each category
x = np.stack([L, a, b], axis=1)

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



#min_points = 6
#dists = kdist(x_foreground, min_points)
#epsilon = np.percentile(dists, 30)
#labels, n_clusters = dbscan(x_foreground, counts_foreground, epsilon, min_points)
#cluster_plot(x_foreground, labels, n_clusters, 'colour-dbscan.pdf')
#cluster_info(x_foreground, labels, counts_foreground)

# Taking the largest cluster and using it to generate 3 foreground colours
# Identify the largest cluster
#sorted_labels, sorted_sizes = cluster_info(x_foreground, labels, counts_foreground)
#largest_cluster = sorted_labels[-1]
#print('largest cluster: %d'%largest_cluster)
#x_f = x_foreground[labels == largest_cluster]
#counts_f = counts_foreground[labels == largest_cluster]

# Get the 3 foreground colours
x_f = x[mask_foreground]
counts_f = counts[mask_foreground]
means_f = kmeans_colours(x_f, counts_f, 3)

# Get the 10 accent colours
x_a = x[mask_accent]
counts_a = counts[mask_accent]
means_a = kmeans_colours(x_a, counts_a, 10)

# Get the 3 background colours
x_b = x[mask_background]
counts_b = counts[mask_background]
means_b = kmeans_colours(x_b, counts_b, 1)

x_b2 = x[mask_background2]
counts_b2 = counts[mask_background2]
means_b2 = kmeans_colours(x_b2, counts_b2, 2)
print(means_b2)

def output_colours(bg, bg2,  a, fg, xresources=False):
    if xresources:
        num_str = '*.color%d: #%02x%02x%02x'
        bg_str = '*.background: #%02x%02x%02x'
        fg_str = '*.foreground: #%02x%02x%02x'
        cursor_str = '*.cursor: #ffffff'
    else:
        num_str = '#define COLOUR%d #%02x%02x%02x'
        bg_str = '#define BACKGROUND #%02x%02x%02x'
        fg_str = '#define FOREGROUND #%02x%02x%02x'
        cursor_str = '#define CURSOR #ffffff'
    
    colours = np.concatenate([bg, bg2, a, fg], axis=0)
    for i in range(colours.shape[0]):
        colour = colours[i]
        print(num_str%(i, int(colour[0]), int(colour[1]), int(colour[2])))

    print(bg_str%(int(bg[0,0]), int(bg[0,1]), int(bg[0,2])))
    print(fg_str%(int(fg[-1,0]), int(fg[-1,1]), int(fg[-1,2])))
    print(cursor_str)

output_colours(means_b, means_b2,  means_a, means_f, use_xresources)
