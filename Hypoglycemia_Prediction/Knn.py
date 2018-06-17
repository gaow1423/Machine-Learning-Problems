import numpy as np
import copy
import csv
from dataprocess import dataprocessing, writeintocsv
import matplotlib
import math
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#from sklearn import decomposition



def main():
    tesdata = np.array(list(csv.reader(open("general_test_instances.csv", "r"), delimiter=",")))
    tesdata = np.delete(tesdata, [0, 1, 2, 3, 4, 5, 6], 1)
    
#    features, targets = dataprocessing("Subject_7_part1.csv")
    features_a, targets_a = dataprocessing("./data_a/Subject_1.csv")
    features_b, targets_b = dataprocessing("./data_b/Subject_4.csv")
    features_c, targets_c = dataprocessing("./data_c/Subject_6.csv")
    features_d, targets_d = dataprocessing("./data_d/Subject_9.csv")
    features = features_a
    features = np.vstack((features, features_b, features_c, features_d))
    targets = targets_a
    targets = np.r_[targets, targets_b, targets_c, targets_d]
    
    features_std = StandardScaler().fit_transform(features.astype(np.float64))
    tesdata_std = StandardScaler().fit_transform(tesdata.astype(np.float64))
    predict_res = np.array([])
    predict_score = np.array([])
#14 dimensions work better
#    a = np.array([])
#    for w in range(1, 57):
    pca = PCA(n_components = 43)
    pca.fit(features_std)
    xtrain = pca.fit_transform(features_std)
    xtest = pca.fit_transform(tesdata_std)
#    print(pca.explained_variance_ratio_)
    ytrain = targets
    
    neigh = KNeighborsClassifier(n_neighbors=3, weights = 'distance')
    neigh.fit(xtrain, ytrain.ravel())
    count = 0
    for i in range(len(xtest)):
#            if (neigh.predict([xtest[i]]) == 1):
#                count += 1
#        a = np.append(a, count)

        predict_res = np.append(predict_res, neigh.predict([xtest[i]]))
        predict_score = np.append(predict_score, neigh.predict_proba([xtest[i]])[0, 1])
#        print(neigh.predict([xtest[i]]), neigh.predict_proba([xtest[i]]))
    predict_file = np.hstack((np.matrix(predict_score).T, np.matrix(predict_res).T))

#    print(predict_file)
    writeintocsv(predict_file, "./prediction_file/general_pred2.csv")
#    plt.plot(range(1, 57), a)
#    plt.show()


#    print(predict_res, predict_score)


#def getData():
#    features, targets = dataprocessing("Subject_2_part1.csv")#3130
#    xtrain = features
#    ytrain = targets
#    for i in range (0, 30):
#        min_tr = np.min(xtrain[:, i])
#        max_tr = np.max(xtrain[:, i])
#        xtrain[:, i] = (xtrain[:, i] - min_tr) / (max_tr - min_tr)
#    return xtrain, ytrain

#def Knn_eval(start, end, testd, traind, testdy, traindy):
#    count = 0
#    for i in range(0, testd.shape[0]):
#        x = copy.deepcopy(testd)
#        y = copy.deepcopy(traind)
#        dis = np.array([])
#        for j in range(0, traind.shape[0]):
#            y[j, :] = x[i, :]##every line in y is the same, has the same len with train.
#            y[j, :] = traind[j, :] - y[j, :]
#            dis =np.append(dis, np.matrix(y[j, :]) * np.matrix(y[j, :]).T)
#        dis = np.hstack((np.matrix(dis).T, np.matrix(traindy).T))
#        dis = dis[np.argsort(dis.A[:,0])]
##the number of zero in K
#        cnt_0 = 0
#        for k in range(start, end):
#            if (dis[k, 1] == 0):
#                cnt_0 += 1
#
#        if (cnt_0 >= (start/2)):
#            if(testdy[i] == 1):
#                count += 1
#        else:
#            if(testdy[i] == 0):
#                count += 1
#    return count


if __name__ == '__main__':
    main()
