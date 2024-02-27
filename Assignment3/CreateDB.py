import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def DataSet(showResult=True):
    Data = np.zeros(shape=(10304, ), dtype= np.int8)
    Label = np.array([])

    for i in range(0,2):
        for a in range(1,11):
            image = Image.open(f'./dataset/s{i}/{a}.pgm')
            for d in range(-15, 16, 1):
                r_img = image.rotate(d)
                data = np.asarray(r_img).reshape(-1)
                Data = np.vstack((Data,data))
                Label = np.append(Label,int(i))

    Label = Label.astype('int8')
    Data = Data[1:]

    print(f'All Data shape: {Data.shape}')

    if showResult:
        choose = np.random.choice(Data.shape[0], size=24, replace=False)
        someSamples , someSamplesLabel = Data[choose] , Label[choose]

        plt.figure(figsize=(10,7))
        plt.suptitle(f"Some Data from dataset")
        i=1
        for s in someSamples:
            plt.subplot(3,8,i)
            plt.axis('off')
            plt.title(f"L:{someSamplesLabel[i-1]}")
            plt.imshow(s.reshape(112,92),cmap='gray')
            i+=1
        plt.show()

    return Data , Label