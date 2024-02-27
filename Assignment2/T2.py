# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt


# f = './1.pgm'

# img = Image.open(f)
# data = np.asarray(img).reshape(-1)


# plt.imshow(data.reshape(112,92),cmap='gray')
# plt.show()





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
plt.imshow(test_sina_x.reshape(112,92),cmap='gray')
plt.show()


train_x, test_x, train_y, test_y = train_test_split(Data, Label, test_size=0.05, random_state=0)


display_data(train_x,train_y,'Train data')
display_data(test_x,test_y,'Test data')



from sklearn.decomposition import PCA
pca = PCA().fit(train_x)

numberComponents = 395
EigenFaces = pca.components_[:numberComponents]


print('Top 10 eigenfaces:\n')
imgs = EigenFaces[:10]
L = len(imgs)
plt.figure(figsize=(7,7))
for i in range(L):
    plt.subplot(2,5,i+1)
    plt.imshow(imgs[i].reshape(112,92), cmap='gray')
plt.show()

weightsFaces = EigenFaces @ (train_x - pca.mean_).T


def find_best_match_face(test):
    '''Find face that's best match !'''
    X_i = np.dot(EigenFaces.T, (np.asarray([test,]) - pca.mean_))
    result = np.argmin(np.linalg.norm(weightsFaces - X_i, axis=0))
    print(f"The Best match is '{train_y[result]}'")
    return result

# Test a data
# firstTest = find_best_match_face(test_x[-1])
firstTest = find_best_match_face(test_sina_x)

imgs = [test_sina_x, train_x[firstTest]]
plt.figure(figsize=(5,3))
plt.subplot(1,2,1)
plt.title("test_sample")
plt.imshow(imgs[0].reshape(112,92), cmap='gray')
plt.subplot(1,2,2)
plt.title("match_data")
plt.imshow(imgs[1].reshape(112,92), cmap='gray')
plt.show()


# Test All Test datas
i = 0
plt.figure(figsize=(14,14))
for test in test_x:
    i += 1
    plt.subplot(4,12,i)
    plt.title("test_sample")
    plt.imshow(test.reshape(112,92), cmap='gray')
    firstTest = find_best_match_face(test)
    i += 1
    plt.subplot(4,12,i)
    plt.title("match_data")
    plt.imshow(train_x[firstTest].reshape(112,92), cmap='gray')
plt.show()

