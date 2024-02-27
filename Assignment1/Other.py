from sklearn.cluster import OPTICS
from sklearn import datasets
import numpy as np
 
# Load the iris dataset
iris = datasets.load_iris()
X = iris.data
 
# Fit the OPTICS model
clustering = OPTICS(min_samples=20, xi=.05, min_cluster_size=.05)
clustering.fit(X)

# Extract the clusters
labels = clustering.labels_
 
# Print the cluster labels
print("Cluster Labels:", labels)
