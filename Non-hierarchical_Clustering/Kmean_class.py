import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random

class kmean:
    def __init__(self, inputfile, k):
        self.data = np.genfromtxt(inputfile, delimiter=",")
        self.k = k
        self.seeds = [self.data[x] for x in random.sample(range(0, len(self.data)), k)]
        self.SSE = 0
        self.rtime = 0
        self.iteset = 0
    
    def implement(self):
        self.SSE = 0
        row, col = self.data.shape
        center = np.zeros((self.k, col))
        count = np.zeros(len(self.seeds))
        
        label = np.zeros(row)
        min = np.zeros(row)
        for z in range(0, row):
            min[z] = float("inf")
        for k in range(0, self.k):
            temp = np.zeros((row, col))
            for r in range(0, len(self.data)):
                temp[r] = np.array([self.seeds[k]])##6000 rows of seed
            diff = self.data - temp
            diff = np.matrix(diff)
            for ind in range(0, len(self.data)):

                if (diff[ind] * (diff[ind].T) < min[ind]):
                    min[ind] = diff[ind] * (diff[ind]).T
                    label[ind] = k
        self.SSE = np.sum(min)
        #reassign the center
        label = label.astype(int)
        for i in range(0, row):
            count[label[i]] += 1

        for i in range(0, self.k):
            self.seeds[i] = self.seeds[i]*0
            for j in range(0, row):
                if (label[j] == i):
                    self.seeds[i] += self.data[j]
            self.seeds[i] = self.seeds[i]/count[i]
        self.rtime += 1
        return 0

    def itestop(self):
        previous = 0
        current = 1
        arr_sse = np.array([])
        if (self.iteset != 1):
            while(previous != current):
                self.implement()
                arr_sse = np.append(arr_sse, self.SSE)
                previous = current
                current = self.SSE
        else:
            while (self.rtime <= 9):
                self.implement()
                arr_sse = np.append(arr_sse, self.SSE)
                
        return arr_sse

    def setk(self, newk):
        self.k = newk
        self.rtime = 0
        self.seeds = [self.data[x] for x in random.sample(range(0, len(self.data)), newk)]
        return 0





