import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_moons
from sklearn.datasets.samples_generator import make_blobs
import New_Naive

#X, y_true = make_blobs(n_samples=300, centers=4,
#                       cluster_std=0.60, random_state=0)
X, y = make_moons(n_samples=250, noise=0.05, random_state=42)
plt.scatter(X[:, 0], X[:, 1], s=50)

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
labels1 = kmeans.predict(X)
centers1 = kmeans.cluster_centers_
centers2, labels2, initial_centers = New_Naive.Naive_k_means(X,2,addInitial=True)
plt.scatter(X[:, 0], X[:, 1], c=labels1, s=50, cmap='viridis')
plt.scatter(centers1[:, 0], centers1[:, 1], c='black', s=200, alpha=0.5)
plt.show()

print(labels2)
plt.scatter(X[:, 0], X[:, 1], c=labels2, s=50, cmap='viridis')
plt.scatter(centers2[:, 0], centers2[:, 1], c='black', s=50, cmap='viridis')
plt.scatter(initial_centers[:, 0], initial_centers[:, 1], c='purple', s=200, alpha=0.5)
plt.show()

#centers, labels = find_clusters(X, 4)
#plt.scatter(X[:, 0], X[:, 1], c=labels,
#           s=50, cmap='viridis');