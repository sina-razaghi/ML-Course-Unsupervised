###################################################### Read Dataset ######################################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

name = 'dataset2.txt'
dataset = pd.read_csv(name, sep=",", header=None)
dataset.columns = ['X1', 'X2', 'Y']
print(dataset)

X = dataset.iloc[:, 0:2].values
Y = dataset.iloc[:, 2:3].values

# # plt.scatter(X[:, 0], X[:, 1], marker="o", c=Y, s=10)
# plt.scatter(X[:, 0], X[:, 1], marker="o", s=10)
# plt.show()

###################################################### Elbow Chart ######################################################

# from sklearn.cluster import KMeans
# import warnings
# warnings.filterwarnings('ignore')

# cluster_list = []
# classes = range(1,16)

# for i in classes:
#     clustering = KMeans(n_clusters=i)
#     clustering.fit(X)
#     cluster_list.append(clustering.inertia_)

# plt.plot(classes, cluster_list, marker="o", color='red')
# plt.show()

###################################################### K-Means in SKLeaen ######################################################

# import numpy as np

# kmeans = KMeans(n_clusters=2, max_iter=4)
# kmeans.fit(X)

# plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='r')
# plt.show()


# def evaluate(kmeans, X):
#         centroids = []
#         centroid_idxs = []
#         for x in X:
#             dists = euclidean(x, kmeans.centroids)
#             centroid_idx = np.argmin(dists)
#             centroids.append(kmeans.centroids[centroid_idx])
#             centroid_idxs.append(centroid_idx)
#         return centroids, centroid_idx

# def euclidean(point, data):
#     """
#     Euclidean distance between point & data.
#     Point has dimensions (m,), data has dimensions (n,m), and output will be of size (n,).
#     """
#     return np.sqrt(np.sum((point - data)**2, axis=1))


# kmeans = KMeans(n_clusters=3)
# kmeans.fit(X)
# # View results
# class_centers, classification = evaluate(kmeans, X)

# import seaborn as sns
# sns.scatterplot(x=[X[0] for X in X],
#                 y=[X[1] for X in X],
#                 hue=Y,
#                 style=classification,
#                 palette="deep",
#                 legend=None
#                 )
# plt.plot([x for x, _ in kmeans.centroids],
#          [y for _, y in kmeans.centroids],
#          '+',
#          markersize=10,
#          )
# plt.show()



# import random


# # Initialize the centroids, using the "k-means++" method, where a random datapoint is selected as the first,
# # then the rest are initialized w/ probabilities proportional to their distances to the first
# # Pick a random point from train data for first centroid
# kmeans.centroids = [random.choice(X)]
# for _ in range(3):
#     # Calculate distances from points to the centroids
#     dists = np.sum([euclidean(centroid, X) for centroid in kmeans.centroids], axis=0)
#     # Normalize the distances
#     dists /= np.sum(dists)
#     # Choose remaining points based on their distances
#     new_centroid_idx, = np.random.choice(range(len(X)), size=1, p=dists)
#     kmeans.centroids += [X[new_centroid_idx]]

# print("normalize")

# plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, linewidths=3, color='r')
# plt.show()



###################################################### K-Means Algorithem ######################################################

# import numpy as np
# import matplotlib.pyplot as plt

# def kmeans(X, K, max_iters=400):
#     '''ُ
#     پیاده سازی الگوریتم k-means برای داده های X و تعداد خوشه های K
#     '''

#     # تعداد داده های موجود در دیتاست
#     m = X.shape[0]

#     # تعیین تصادفی مقدار اولیه مراکز خوشه
#     centers = X[np.random.choice(m, K), :]

#     # شروع حلقه های تکراری الگوریتم
#     for iter in range(max_iters):
#         # محاسبه فاصله هر داده از مراکز خوشه
#         distances = np.sqrt(((X - centers[:, np.newaxis])**2).sum(axis=2))

#         # نام خوشه هایی که هر داده در آنها قرار دارد
#         cluster_ids = np.argmin(distances, axis=0)

#         # ذخیره مراکز خوشه در متغیر جدید
#         centers_new = np.zeros_like(centers)

#         # به روز رسانی مراکز خوشه
#         for i in range(K):
#             centers_new[i] = np.mean(X[cluster_ids == i], axis=0)

#         # بررسی شرط خاتمه الگوریتم
#         if np.allclose(centers, centers_new):
#             break

#         # به روز رسانی مراکز خوشه برای حلقه بعدی
#         centers = centers_new.copy()

#     # برگرداندن نتایج
#     return centers, cluster_ids

# # تولید داده های تصادفی
# # X = np.random.rand(100, 2)

# # اعمال الگوریتم k-means
# K = 2
# centers, cluster_ids = kmeans(X, K)

# # رسم نتایج
# colors = ['red', 'green', 'blue', 'pink', 'black']
# # for i in range(K):
# #     cluster = X[cluster_ids == i]
# #     plt.scatter(cluster[:, 0], cluster[:, 1], color=colors[i])

# plt.scatter(X[:, 0], X[:, 1], marker="o", c=cluster_ids, s=10)

# plt.scatter(centers[:, 0], centers[:, 1], color='k', marker='x')
# plt.show()


# توضیحات:

# ابتدا، کتابخانه‌های NumPy و Matplotlib را وارد می‌کنیم.
# تابع kmeans را با ورودی‌های داده X و تعداد خوشه‌های K تعریف می‌کنیم. همچنین، مقدار پیش‌فرض برای تعداد حداکثر تکرار‌های

import math

def PointsInCircum(r, n=100):
    return [(math.cos(2*math.pi/n*x)*r+np.random.normal(-2,2),
             math.sin(2*math.pi/n*x)*r+np.random.normal(-2,2)) for x in range(1,n+1)]
df = pd.DataFrame(PointsInCircum(15,900))
df = df.append(PointsInCircum(2,300))
# df = df.append(PointsInCircum(100,300))

# df=df.append([(np.random.randint(-600,600), 
#                np.random.randint(-600,600)) for i in range(300)])

plt.figure(figsize=(7,7))
plt.scatter(df[0], df[1])
plt.xticks(size=10)
plt.yticks(size=10)
plt.show()

import matplotlib

# from hdbscan import HDBSCAN
# hDBSAN = HDBSCAN()
# hDBSAN.fit(df[[0,1]])
# df['HDBSCAN_opt_labels']=hDBSAN.labels_ 


from sklearn.cluster import OPTICS
cluster = OPTICS()
cluster.fit(df[[0,1]])
cluster_labels = cluster.labels_

plt.figure(figsize=(10,10))
plt.scatter(df[0],df[1],c=cluster_labels,s=10)
plt.legend()
plt.show()

