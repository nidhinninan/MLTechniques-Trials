# K-means clustering for 1D data
import numpy as np

# Step 1: Initialize data and cluster means
X = np.array([1, 2, 2, 3, 6, 10, 11, 16, 18])
m = np.array([3, 8, 15])  # Initial cluster means

# Step 2: Cluster assignment based on squared Euclidean distance
def assign_clusters(X, means):
    distances = np.array([[np.square(x - mean) for mean in means] for x in X])
    cluster_assignments = np.argmin(distances, axis=1)
    return cluster_assignments

clusters = assign_clusters(X, m)

# Step 3: Update cluster means
def update_means(X, clusters, k):
    new_means = []
    for i in range(k):
        cluster_points = X[clusters == i]
        new_mean = np.mean(cluster_points) if len(cluster_points) > 0 else 0
        new_means.append(new_mean)
    return np.array(new_means)

updated_means = update_means(X, clusters, len(m))

# Step 4: Compute error (objective function)
def compute_error(X, clusters, means):
    error = 0
    for i, x in enumerate(X):
        cluster_mean = means[clusters[i]]
        error += np.square(x - cluster_mean)
    return error

error = compute_error(X, clusters, updated_means)

# Print results
print("Initial Cluster Means:", m)
print("Cluster Assignments:", clusters)
print("Updated Cluster Means:", updated_means)
print("Objective Function (Error):", error)
