import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
# import seaborn as sns 
from sklearn.cluster import KMeans
from sklearn.metrics  import silhouette_score 
from sklearn.preprocessing import MinMaxScaler 

name = 'IRIS.csv'
dataset = pd.read_csv(name, sep=",", header=None)
dataset.columns = ["sepal_length","sepal_width","petal_length","petal_width","label","species"]

print(dataset.info())
print("\n============================================\n")
print(dataset)

X = dataset.iloc[:, 0:4].values
Y = dataset.iloc[:, 4:5].values


# # sns.FacetGrid(dataset,hue="species").map(sns.distplot,"petal_length").add_legend() 
# # sns.FacetGrid(dataset,hue="species").map(sns.distplot,"petal_width").add_legend() 
# # sns.FacetGrid(dataset,hue="species").map(sns.distplot,"sepal_length").add_legend() 
# # plt.show()



# # sns.set_style("whitegrid") 
# # sns.pairplot(dataset,hue="species",size=3); 
# # plt.show()




# from sklearn.cluster import KMeans 

# # inertia = [] 

# # for i in range(1, 15): 
# #     kmeans_plus = KMeans(n_clusters = i, init = 'k-means++', max_iter = 400, n_init = 10, random_state = 0) 
# #     kmeans_plus.fit(X) 
# #     inertia.append(kmeans_plus.inertia_)

# # plt.plot(range(1, 15), inertia, marker="o", color='red')
# # plt.title('The elbow method') 
# # plt.show()


kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 500, n_init = 10).fit(X)
y_kmeans = kmeans.fit_predict(X) 


print("\n==================k-means++==========================\n")

silhouette_score_average = silhouette_score(X, y_kmeans)
print(f"silhouette_score_average => {silhouette_score_average}")


T = 0
F = 0

for i in range(X.shape[0]):
    if y_kmeans[i] == Y[i]:
        T += 1
    else:
        F += 1 

print(f"True : {T} || False : {F}")
print(f"score_TF => {(T/150)}")


# plt.scatter(X[:, 0], X[:, 1], marker="o", c=Y, s=40)
# plt.title("real data")
# plt.show()

# #Visualising the clusters 
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 40, c = 'purple', label = 'Iris-setosa') 
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 40, c = 'orange', label = 'Iris-versicolour') 
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 40, c = 'green', label = 'Iris-virginica') 
#Plotting the centroids of the clusters 
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'red', label = 'Centroids') 
plt.title("clustering data")
plt.show()















import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from numpy.random import uniform
from sklearn.datasets import make_blobs
import seaborn as sns
import random

def euclidean(point, data):
    """
    Euclidean distance between point & data.
    Point has dimensions (m,), data has dimensions (n,m), and output will be of size (n,).
    """
    return np.sqrt(np.sum((point - data)**2, axis=1))


class KMeans:    
    def __init__(self, n_clusters=8, max_iter=500):
        self.n_clusters = n_clusters
        self.max_iter = max_iter    
    def fit(self, X_train):        
        # Initialize the centroids, using the "k-means++" method, where a random datapoint is selected as the first,
        # then the rest are initialized w/ probabilities proportional to their distances to the first
        # Pick a random point from train data for first centroid
        self.centroids = [random.choice(X_train)]        
        for _ in range(self.n_clusters-1):
            # Calculate distances from points to the centroids
            dists = np.sum([np.sqrt(np.sum((centroid - X_train)**2, axis=1)) for centroid in self.centroids], axis=0)
            # Normalize the distances
            dists /= np.sum(dists)
            # Choose remaining points based on their distances
            new_centroid_idx, = np.random.choice(range(len(X_train)), size=1, p=dists)
            self.centroids += [X_train[new_centroid_idx]]        
            # This initial method of randomly selecting centroid starts is less effective
        # min_, max_ = np.min(X_train, axis=0), np.max(X_train, axis=0)
        # self.centroids = [uniform(min_, max_) for _ in range(self.n_clusters)]        
        # Iterate, adjusting centroids until converged or until passed max_iter
        iteration = 0
        prev_centroids = None
        while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
            # Sort each datapoint, assigning to nearest centroid
            sorted_points = [[] for _ in range(self.n_clusters)]
            for x in X_train:
                dists = euclidean(x, self.centroids)
                centroid_idx = np.argmin(dists)
                sorted_points[centroid_idx].append(x)            
                # Push current centroids to previous, reassign centroids as mean of the points belonging to them
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():  
                    # Catch any np.nans, resulting from a centroid having no points
                    self.centroids[i] = prev_centroids[i]
            iteration += 1    
    def evaluate(self, X):
        centroids = []
        centroid_idxs = []
        for x in X:
            dists = euclidean(x, self.centroids)
            centroid_idx = np.argmin(dists)
            centroids.append(self.centroids[centroid_idx])
            centroid_idxs.append(centroid_idx)        
        return centroids, centroid_idxs
    




# Create a dataset of 2D distributions
centers = 3
# X_train, true_labels = make_blobs(n_samples=1000, centers=centers, random_state=42)
X_train = X
true_labels = Y
X_train = StandardScaler().fit_transform(X_train)
# Fit centroids to dataset
kmeans = KMeans(n_clusters=centers)
kmeans.fit(X_train)


# View results
class_centers, classification = kmeans.evaluate(X_train)
# sns.scatterplot(data=dataset,x="sepal_length",y="sepal_width",hue=true_labels)

print("\n================== k-means++ normalized ==========================\n")

silhouette_score_average = silhouette_score(X_train, classification)
print(f"silhouette_score_average => {silhouette_score_average}")


T = 0
F = 0

for i in range(X.shape[0]):
    if classification[i] == Y[i]:
        T += 1
    else:
        F += 1 

print(f"True : {T} || False : {F}")
print(f"score_TF => {(T/150)}")

plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=classification, s=40)
plt.plot([x[0] for x in kmeans.centroids],[y[1] for y in kmeans.centroids],'k+',markersize=10,)
plt.title("clustering normalize data")
plt.show()

