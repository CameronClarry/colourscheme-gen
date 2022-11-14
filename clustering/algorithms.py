import numpy as np

# Takes an array of points, and one particular point
# Calculates the distance from each point in the array to the given point
def dist(x, point):
    return np.sum((x-point)**2, axis=1)

# Takes an array of n points of shape (n, ...)
# Returns an array with the shape (n) where the ith entry is the distance from the ith point to the kth nearest point
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


# Perform k-means clustering
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

# Perform modified k-means clustering
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

# Perform clustering with DBSCAN
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

