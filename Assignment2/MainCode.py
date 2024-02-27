from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


Data = np.zeros(shape=(10304, ), dtype= np.int8)
Label = np.array([])

for i in range(1,42):
    for a in range(1,11):
        sina_img = Image.open(f'./dataset/s{i}/{a}.pgm')
        data = np.asarray(sina_img).reshape(-1)
        Data = np.vstack((Data,data))
        Label = np.append(Label,i)

Label = Label.astype('int8')
Data = Data[1:]

print(f'All Data shape: {Data.shape}')

sina_img = Image.open(f'./2.pgm')
test_sina_x = np.asarray(sina_img).reshape(-1)

someSamples = Data[np.random.choice(Data.shape[0], 
                                size=24, 
                                replace=False),:]
    
print("Faces shape:", (someSamples[0].reshape(112,92)).shape)

plt.figure(figsize=(10,7))
plt.suptitle(f"Some Data from dataset")
i=1
for s in someSamples:
    plt.subplot(4,6,i)
    plt.imshow(s.reshape(112,92),cmap='gray')
    i+=1
plt.show()

train_x, test_x, train_y, test_y = train_test_split(Data, Label, test_size=0.048, random_state=42)

print(f'=========================')
print(f'Train Data shape: {train_x.shape}')
print(f'=========================')
print(f'Train Data shape: {test_x.shape}')

# Preprocess the data
n_samples, n_features = train_x.shape

# Define number of principal components to use for reconstruction
n_components_range = [10, 25, 50, 100, 200, train_x.shape[0]]

# Loop over different number of principal components
for n_components in n_components_range:

    # Perform PCA on the data
    pca = PCA(n_components=n_components, whiten=False)
    pca.fit(train_x[:5])

    # Reconstruct the images using the principal components

    sina_pca = pca.transform([test_sina_x,])
    sina_reconstructed = pca.inverse_transform(sina_pca)

    plt.figure(figsize=(5,4))
    plt.suptitle(f"It has seen something like it")
    plt.subplot(1,2,1)
    plt.title("Test Sample")
    plt.imshow(test_sina_x.reshape(112,92), cmap='gray')
    plt.subplot(1,2,2)
    plt.title(f'PCA_Component={n_components}')
    plt.imshow(sina_reconstructed.reshape(112,92), cmap='gray')
    plt.show()

    X_pca = pca.transform(test_x)
    X_reconstructed = pca.inverse_transform(X_pca)

    EigenFaces = pca.components_[:n_components]
    print('Top 10 eigenfaces:\n')
    imgs = EigenFaces[:10]
    L = len(imgs)
    plt.figure(figsize=(7,7))
    plt.suptitle(f"Top 10 eigenfaces {n_components} components")
    for i in range(L):
        plt.subplot(2,5,i+1)
        plt.imshow(imgs[i].reshape(112,92), cmap='gray')
    plt.show()

    # Plot the reconstructed images
    fig, axes = plt.subplots(nrows=5, ncols=8, figsize=(13, 7),
                             subplot_kw={'xticks': [], 'yticks': []})
    a = 0
    for i, ax in enumerate(axes.flat):
        if i%2==0:
            ax.imshow(test_x[a].reshape(112,92), cmap='gray')
            ax.set_title(f'Original_{a}')
        else:
            ax.imshow(X_reconstructed[a].reshape(112,92), cmap='gray')
            ax.set_title(f'PCA_Component={n_components}')
            a += 1
    plt.tight_layout()
    plt.show()
