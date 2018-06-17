import numpy as np
import csv

def testdataloader(i):
    return np.matrix(list(csv.reader(open("sampleinstance_{}.csv".format(i)))))

def traindataloader(filename):
    return np.array(list(csv.reader(open(filename, "r"), delimiter=",")))
#    m, n = a.shape
#    a = a[:, :, 0].reshape(m, n)
#    return a

def writeintocsv(data, filename):
    np.savetxt(filename, data, delimiter=',')

def insert(rowset, features, targets):
    temp = rowset[0][0:8]
    for i in range(len(rowset)-1):
        if (rowset[i][8] == 1):
            return features, targets
#        temp = np.hstack((temp, rowset[i+1][0:8]))
    temp = rowset[:, 0].T
    for i in range(1, rowset.shape[1]-1):
        temp = np.hstack((temp, rowset[:, i].T))
    targets = np.append(targets, rowset[rowset.shape[0]-1][8])
    if (features.shape[0] == 1):
        features = temp
    else:
        features = np.vstack((features, temp))
    return features, targets

#rearrane the training data so that one row represents one instance (one instance = size of window * 8)
def dataprocessing(filename):
    features = np.matrix([])
    targets = np.array([])
    input = traindataloader(filename)#3345*10
    data = np.zeros((input.shape[0], input.shape[1]-1))
    for i in range(input.shape[0]):
        row = input[i]
        for j in range(1, input.shape[1]):
            data[i][j-1] = row[j]
#    print (data, data.shape)
    for i in range(0, len(data)-8):
        rowset = data[i:i+7]
#        print(rowset.shape)
        features, targets = insert(rowset, features, targets)
#        print(features.shape)##3130, 56
#        print(targets.shape)
    return features, targets

