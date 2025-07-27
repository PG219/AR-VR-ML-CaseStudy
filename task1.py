import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN

# Generate synthetic data (blobs)
X, y = make_blobs(n_samples=50, centers=3, n_features=2, random_state=0)

# Function to plot clusters
def plot_clusters(X, labels, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, edgecolor='k')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.grid(True)
    plt.show()

# Plot original unclustered data
plot_clusters(X, np.zeros(X.shape[0]), "Unclustered Data")

# Apply K-Means
kmeans = KMeans(n_clusters=5, random_state=42)
labels_kmeans = kmeans.fit_predict(X)
plot_clusters(X, labels_kmeans, "K-Means Clustering[5]")

# Apply DBSCAN
dbscan = DBSCAN(eps=1.0, min_samples=5)
labels_dbscan = dbscan.fit_predict(X)
plot_clusters(X, labels_dbscan, "DBSCAN Clustering[1.5]")



