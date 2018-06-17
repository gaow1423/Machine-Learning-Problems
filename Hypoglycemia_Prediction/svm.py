import numpy as np
import csv
from dataprocess import dataprocessing, writeintocsv
import math
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from sklearn.datasets import load_iris
from sklearn.svm import SVC
import matplotlib
import math
import matplotlib.pyplot as plt


def main():
#    features, targets = dataprocessing("Subject_7_part1.csv")

    features_a, targets_a = dataprocessing("./data_a/Subject_1.csv")
    features_b, targets_b = dataprocessing("./data_b/Subject_4.csv")
    features_c, targets_c = dataprocessing("./data_c/Subject_6.csv")
    features_d, targets_d = dataprocessing("./data_d/Subject_9.csv")

    features = features_a
    features = np.vstack((features, features_b, features_c, features_d))

    targets = targets_a
    targets = np.r_[targets, targets_b, targets_c, targets_d]
    X = features
    y = targets

    label = np.array([])
    score = np.array([])
#    ran = range(0,1000,100)
#    for k in ran:
    weight = abs(np.random.randn(len(X)))
    for i in range(len(X)):
        if(targets[i] == 1):
            weight[i] *= 400
    clf_weights = SVC(probability=False)
    clf_weights.fit(X, y, sample_weight = weight)

    tesdata = np.array(list(csv.reader(open("general_test_instances.csv", "r"), delimiter=",")))
    tesdata = np.delete(tesdata, [0, 1, 2, 3, 4, 5, 6], 1)
    tesdata = tesdata.astype(np.float64())
    count = 0
    for j in range(len(tesdata)):
        label = np.append(label, clf_weights.predict([tesdata[j]]))
        score = np.append(score, clf_weights.decision_function([tesdata[j]]))
    res = np.hstack((np.matrix(score).T, np.matrix(label).T))
    writeintocsv(res, "./prediction_file/general_pred1.csv")
if __name__ == '__main__':
    main()
