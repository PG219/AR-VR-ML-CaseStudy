import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, load_iris
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

iris = load_iris()
X = iris.data
y = iris.target
title = "Iris Dataset"

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', s=40)
plt.title(f"PCA Projection of {title}")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar()
plt.grid(True)
plt.show()


tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(8, 5))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=40)
plt.title(f"t-SNE Projection of {title}")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.colorbar()
plt.grid(True)
plt.show()
