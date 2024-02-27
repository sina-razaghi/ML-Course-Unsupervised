# Import the necessary libraries
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



def display_data(D,y,pltTitle):
    print(f'\n=========================\n{pltTitle}:')
    print(f'Data shape: {D.shape}')

    someSamples = D[np.random.choice(D.shape[0], 
                                size=6, 
                                replace=False),:]
    
    print("Faces shape:", (someSamples[0].reshape(112,92)).shape)

    print('\nSample from data:')
    plt.figure(figsize=(10,7))
    plt.title(pltTitle)
    i=1
    for s in someSamples:
        plt.subplot(2,3,i)
        plt.imshow(s.reshape(112,92),cmap='gray')
        i+=1
    plt.show()



Data = np.zeros(shape=(10304, ), dtype= np.int8)
Label = np.array([])



for i in range(1,42):
    for j in range(1,11):
        img = Image.open(f'./dataset/s{i}/{j}.pgm')
        data = np.asarray(img).reshape(-1)
        Data = np.vstack((Data,data))
        Label = np.append(Label,i)


Label = Label.astype('int8')
Data = Data[1:]


img = Image.open(f'./dataset/s41/3.pgm')
data = np.asarray(img).reshape(-1)
test_sina_x = data
# plt.imshow(test_sina_x.reshape(112,92),cmap='gray')
# plt.show()


train_x, test_x, train_y, test_y = train_test_split(Data, Label, test_size=0.05, random_state=0)


# display_data(train_x,train_y,'Train data')
# display_data(test_x,test_y,'Test data')



from sklearn.decomposition import PCA
pca = PCA().fit(train_x)

EigenFaces = pca.components_

# print('Top 10 eigenfaces:\n')
# imgs = EigenFaces[:10]
# L = len(imgs)
# plt.figure(figsize=(7,7))
# for i in range(L):
#     plt.subplot(2,5,i+1)
#     plt.title(f"EigenFaces{i+1}")
#     plt.imshow(imgs[i].reshape(112,92), cmap='gray')
# plt.show()

avrgFace = pca.mean_
# plt.title("Avrage Face")
# plt.imshow(avrgFace.reshape(112,92), cmap='gray')
# plt.show()

n_component = 300



X_eta = (np.asarray(train_x) - avrgFace)
X_i_list = []
for i in range(len(X_eta)):
    data = np.dot(EigenFaces[i].T, X_eta[i])
    X_i_list.append(data)

X_i = np.asarray(X_i_list)

sigma = avrgFace + np.dot(X_i[:n_component], EigenFaces[:n_component])

Components = np.sum(sigma, axis=0)
plt.title("Output Face")
plt.imshow(sigma.reshape(112,92), cmap='gray')
plt.show()

Output = avrgFace + Components


plt.title("Output Face")
plt.imshow(Output.reshape(112,92), cmap='gray')
plt.show()