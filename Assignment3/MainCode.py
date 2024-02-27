from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from CreateDB import DataSet
import numpy as np


# class CustomPCA:
#     def __init__(self, n_components=None):
#         self.n_components = n_components
    
#     def fit_transform(self, X):
#         # محاسبه ماتریس ویژگی-ویژگی
#         cov_matrix = np.cov(X.T)
        
#         # محاسبه بردار ویژه و مقادیر ویژه
#         eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
#         # مرحله 3: مرتب‌سازی و انتخاب n_components بردار ویژه با بیشترین مقادیر ویژه
#         indices = np.argsort(eigenvalues)[::-1]
#         selected_eigenvectors = eigenvectors[:, indices[:self.n_components]]

#         # مرحله 4: تبدیل داده‌ها به فضای کاهش یافته
#         transformed_data = X.dot(selected_eigenvectors)

#         return transformed_data



def CustomPCA(X, n_components):
    # محاسبه میانگین ستون‌ها
    mean_vector = np.mean(X, axis=0)

    # مرحله 1: تطبیق داده‌ها با میانگین
    centered_data = X - mean_vector

    # محاسبه ماتریس کوواریانس
    covariance_matrix = np.dot(centered_data.T, centered_data) / (X.shape[0] - 1)

    # مرحله 2: محاسبه بردار ویژه‌ها و مقادیر ویژه
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # مرتب‌سازی و انتخاب n_components بردار ویژه با بیشترین مقادیر ویژه
    indices = np.argsort(eigenvalues)[::-1]
    selected_eigenvectors = eigenvectors[:, indices[:n_components]]

    # مرحله 3: تبدیل داده‌ها به فضای کاهش یافته
    transformed_data = np.dot(centered_data, selected_eigenvectors)

    return transformed_data




Data , Label = DataSet(showResult=False)
print(f'All Data shape: {Data.shape}')
print(f'All Label shape: {Label.shape}')

# pca = CustomPCA(n_components=700)
# X_pca = pca.fit_transform(Data)
X_pca = CustomPCA(Data, 700)
print(f'All Data shape after PCA: {X_pca.shape}')

train_x, test_x, train_y, test_y = train_test_split(X_pca, Label, test_size=0.1, random_state=42)

train_x = np.real(train_x)
test_x = np.real(test_x)

count_fold = 0
weights = []
final_weights = []
accuracies = []

print(f'=========================')
print(f'Train Data shape: {train_x.shape}')
print(f'=========================')
print(f'Test Data shape: {test_x.shape}')


from sklearn import svm
from sklearn.model_selection import KFold
kf = KFold(n_splits = 10); kf.get_n_splits(train_x)

model = svm.SVC(kernel='rbf')

for trainFoldIndex, CVFoldIndex in kf.split(train_x):
    print(f"\n============== Fold {count_fold+1} ==============")
    count_fold += 1
    
    model.fit(train_x[trainFoldIndex], train_y[trainFoldIndex])
    print(f"=> SVM Train Fold_{count_fold} Completed")

    scores = model.score(train_x[trainFoldIndex], train_y[trainFoldIndex])
    print(f"=> Train Scores = {scores}")

    scores = model.score(train_x[CVFoldIndex], train_y[CVFoldIndex])
    print(f"=> CV Scores = {scores}")


test_score = model.score(test_x, test_y)
print(f"\n============== Final Test ==============")
print(f"=> SVM Test Scores = {test_score}")
