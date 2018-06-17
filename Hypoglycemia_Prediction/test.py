import numpy as np
import csv
from dataprocess import dataprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
import random

def main():
##    a = np.matrix([[23,4,2,9], [2,3,45,0], [3,4,2,9]])
##    print(a)
##    temp = a[:, 0].T
##    print(temp.shape)
##    for i in range(a.shape[1]-1):
##        temp = np.hstack((temp, a[:, i+1].T))
##    print(temp)
##
#    features, targets = dataprocessing("Subject_2_part1.csv")
#    features_std = StandardScaler().fit_transform(features)
#    pca = decomposition.PCA(n_components = 56)
#    sklearn_pca_x = pca.fit_transform(features_std)
#    print(sklearn_pca_x)

    features = np.hstack((features, np.matrix([targets]).T))
    count = int(targets.shape[0]/k)
    arr_one = np.array([])
    class_1 = 0
    class_0 = 1
    for i in range(len(targets)):
        if (targets[i] == 1):
            class_1 += 1
            arr_one = np.append(arr_one, i)
        else:
            class_0 += 1
    for j in range(class_0):
        features = np.vstack((features, features[random.choice(arr_one)]))
    for k in range(4):
        features = np.random.shuffle(features)

    return features[count:len(features), :], features[0:count, :]

#    print(pca.explained_variance_ratio_)
if __name__ == '__main__':
    main()
