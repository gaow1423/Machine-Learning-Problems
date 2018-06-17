import numpy as np
import csv
from dataprocess import dataprocessing, writeintocsv
import torch
#import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import matplotlib
import math
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from sklearn.datasets import load_iris


def train_and_validation_set(features, targets, k):
    train = np.array([])
    valid_fe = np.array([])
    valid_label = np.array([])
    features_std = StandardScaler().fit_transform(features.astype(np.float64))
#    valid_std = StandardScaler().fit_transform(targets.astype(np.float64))
    pca = decomposition.PCA(n_components = 14)
    pca.fit(features_std)
    features = pca.fit_transform(features_std)
    features = np.hstack((features, np.matrix([targets]).T))
    
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
        features = np.vstack((features, features[int(random.choice(arr_one))]))

#    print(features[6000:6200, 14])
    for k in range(4):
        np.random.shuffle(features)
#    print(features[6000:6200, 14])
    count = int(features.shape[0]/k)
    return features


#features, targets = dataprocessing("Subject_2_part1.csv")#3130

features_a, targets_a = dataprocessing("./data_a/Subject_1.csv")
features_b, targets_b = dataprocessing("./data_b/Subject_4.csv")
features_c, targets_c = dataprocessing("./data_c/Subject_6.csv")
features_d, targets_d = dataprocessing("./data_d/Subject_9.csv")

features = features_a
features = np.vstack((features, features_b, features_c, features_d))

targets = targets_a
targets = np.r_[targets, targets_b, targets_c, targets_d]



train_data = train_and_validation_set(features, targets, 10)

valid_data = np.array(list(csv.reader(open("general_test_instances.csv", "r"), delimiter=",")))
valid_data = np.delete(valid_data, [0, 1, 2, 3, 4, 5, 6], 1)
valid_data_a = valid_data.astype(np.float64)
print(valid_data_a.shape)
pca = decomposition.PCA(n_components = 14)
pca.fit(valid_data_a)
valid_data = pca.fit_transform(valid_data_a)
#print("valid_data.shape", valid_data.shape)



#print(type(train_data[0,1]), type(valid_data[0,1]))
#print(type(train_data), type(np.matrix(valid_data)))
class Hpoglycemia_dataset_tr(Dataset):
    """ Diabetes dataset."""
    
    # Initialize your data, download, etc.
    def __init__(self):
        self.len = train_data.shape[0]
        self.x_data = torch.from_numpy(train_data[:, 0:-1])
        self.y_data = torch.from_numpy(train_data[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len

class Hpoglycemia_dataset_va(Dataset):
    #Initialize your data
    def __init__(self):
        self.len = valid_data.shape[0]
        self.x_data = torch.from_numpy(valid_data)
        self.y_data = torch.from_numpy(valid_data[:, [-1]])
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.len


dataset_tr = Hpoglycemia_dataset_tr()
dataset_va = Hpoglycemia_dataset_va()
train_loader = torch.utils.data.DataLoader(dataset = dataset_tr, batch_size = 32, shuffle=True, num_workers = 2)
valid_loader = torch.utils.data.DataLoader(dataset = dataset_va, batch_size = 32, shuffle=False, num_workers = 2)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(14, 6)
        self.l2 = torch.nn.Linear(6, 4)
        self.l3 = torch.nn.Linear(4, 2)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
        return F.log_softmax(y_pred, dim = 1)

model = Model()
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.2, momentum = 0.9)

#def train(epoch, k, model):
#    model.train()
#    for batch_idx, data in enumerate(train_loader, 0):
#        inputs, labels = data
#        inputs, labels = Variable(inputs).float(), Variable(labels).float()
#        output = model(inputs)
#        loss = criterion(output, labels)
#        optimizer = optim.SGD(model.parameters(), lr = k, momentum = 0.9)
#        optimizer.zero_grad()
#        loss.backward()
#        optimizer.step()

#def validate(loss_vector, accuracy_vector, epochs, model):
##    model.train(False)
#    model.eval()
#    total_cnt = 0
#    correct_cnt, ave_loss = 0, 0
#    for batch_idx, (x, target) in enumerate(valid_loader):
#        x, target = Variable(x, volatile = True).float(), Variable(target, volatile = True).float()
#        output = model(x)
#        loss = criterion(output, target)
#        target = target.long()
#        total_cnt += x.data.size()[0]
#        _, pred_label = torch.max(output.data, 1)
#        corr = 0
#        for k in range(len(pred_label)):
#            if(pred_label[k] == int(target.data[k])):
#                corr += 1
#        correct_cnt += corr
#        ave_loss = ave_loss * 0.9 + loss.data[0] * 0.1
#    loss_vector.append(ave_loss)
#    accuracy = 100. * correct_cnt / total_cnt
#    accuracy_vector.append(accuracy)
#    print('\nEpoch {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(epochs, ave_loss, correct_cnt, total_cnt, accuracy))
def convert(a):
    for i in range(len(a)):
        a[i, 0] = math.exp(a[i, 0])
        a[i, 1] = math.exp(a[i, 1])
    return a

def train(epoch, lr, mom, model):
    model.train()
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = inputs.float()
        labels = labels.long()
        inputs, labels = Variable(inputs), Variable(labels)
        y_pred = model(inputs)
#        print(y_pred)
        loss = criterion(y_pred, labels.squeeze(1))
#        print(epoch, i, loss.data[0])
        optimizer = torch.optim.SGD(model.parameters(), lr, mom)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(epoch, model, prediction_file):
    model.eval()
    correct_cnt = 0
    total_cnt = 0
    res_a = np.array([])
    res_b = np.array([])
    for i, data in enumerate(valid_loader, 0):
        inputs, labels = data
        inputs = inputs.float()
        labels = labels.long()
        total_cnt += inputs.size()[0]
        inputs, labels = Variable(inputs), Variable(labels)
        y_pred = model(inputs)
        _, pred_label = torch.max(y_pred.data, 1)
        corr = 0
        for k in range(len(pred_label)):
            if(int(pred_label[k]) == int(labels[k])):
                corr += 1
        correct_cnt += corr
        pre_acc = np.zeros((y_pred.size(0), y_pred.size(1)))
        for j in range(y_pred.size(0)):
            pre_acc[j, 0] = y_pred[j][0]
            pre_acc[j, 1] = y_pred[j][1]
        pre_acc = convert(pre_acc)
#        print(pre_acc)
        res_a = np.r_[res_a, pre_acc[:, 1]]
        res_b = np.r_[res_b, np.array(pred_label)]
    res = np.vstack((res_a, res_b))
#    print(res)
#    if (prediction_file.size == 0):
#        prediction_file = res
#    else:
#        prediction_file = np.hstack((prediction_file, res))
    prediction_file = res
    accuracy = 100 * correct_cnt / total_cnt
    print(epoch, correct_cnt,total_cnt,accuracy)
    return prediction_file

def main():
    prediction_file = np.array([])
    for epoch in range(450):
#            prediction_file = np.array([])
            train(epoch, 0.2, 0.9, model)
            prediction_file = test(epoch, model, prediction_file)
            print(prediction_file.T)
    count = 0
#    for i in range(len(prediction_file.T)):
#        if (prediction_file.T[i][1] == 1):
#            count += 1
#    print(count)
    writeintocsv(prediction_file.T, "./prediction_file/general_pred3.csv")
if __name__ == '__main__':
    main()
