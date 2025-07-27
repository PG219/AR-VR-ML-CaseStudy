import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN


X = pd.read_csv('train/X_train.txt', delim_whitespace=True, header=None)
y = pd.read_csv('train/y_train.txt', delim_whitespace=True, header=None)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("\nFirst few rows of X:")
print(X.head())

print("\nActivity Label Counts:")
print(y[0].value_counts().sort_index())

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y[0], cmap='Set1', s=10)
plt.title("PCA Projection (Color: Actual Activities)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()


kmeans = KMeans(n_clusters=10, random_state=42)
labels_k = kmeans.fit_predict(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_k, cmap='tab10', s=10)
plt.title("K-Means Clustering on Sensor Data")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()


dbscan = DBSCAN(eps=1.5, min_samples=10)
labels_d = dbscan.fit_predict(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_d, cmap='tab10', s=10)
plt.title("DBSCAN Clustering on Sensor Data")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()





