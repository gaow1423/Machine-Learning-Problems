##Reference: http://sebastianraschka.com/Articles/2014_pca_step_by_step.html#3-b-computing-the-covariance-matrix-alternatively-to-the-scatter-matrix
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la

def norm(a):
    return (a - a.min())/(a.max() - a.min())*255

if __name__ == '__main__':
    data = np.genfromtxt('data-1.txt',
                         delimiter=",")
    print(data.shape)
    w, v = la.eigh(np.cov(data.T))
    
    #Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(w[i]), v[:, i]) for i in range(len(w))]
    
    #Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    print(eig_pairs[0][0])

    '''
    352868.691256
    267895.866871
    227632.699244
    174703.490258
    130486.76236
    115542.502682
    99726.4367264
    90576.0578785
    85326.5368082
    71547.9660125
    '''

#calculate mean
    mean = np.mean(data, axis = 0)
#    pic = eig_pairs[1][0].T.reshape(28*28)
    pic = norm(np.array([eig_pairs[0][1]]).reshape(28, 28))
    for i in range(0, 4):
        pic = np.hstack((pic, norm(np.array([eig_pairs[i+1][1]])).reshape(28, 28)))
    pic_2 = norm(np.array([eig_pairs[5][1]]).reshape(28, 28))
    for i in range(0, 4):
        pic_2 = np.hstack((pic_2, norm(np.array([eig_pairs[i+6][1]])).reshape(28, 28)))
    eig_pic = np.array(np.vstack((pic, pic_2)), dtype = float)
    print(eig_pairs[0][1].shape)

    plt.subplot(2, 2, 1)
    plt.imshow(mean.reshape(28,28), cmap=plt.get_cmap('gray'))
    plt.title("The mean image")
    plt.subplot(2, 2, 2)
    plt.imshow(eig_pic, cmap=plt.get_cmap('gray'))
    plt.title("The images of top ten eigen-vector")

    matrix_w = np.hstack((eig_pairs[0][1].reshape(784,1), eig_pairs[1][1].reshape(784,1),eig_pairs[2][1].reshape(784,1),eig_pairs[3][1].reshape(784,1),eig_pairs[4][1].reshape(784,1),eig_pairs[5][1].reshape(784,1),eig_pairs[6][1].reshape(784,1),eig_pairs[7][1].reshape(784,1),eig_pairs[8][1].reshape(784,1),eig_pairs[9][1].reshape(784,1)))

    transformed = matrix_w.T.dot(data.T)
    transformed = transformed.T
    index = np.zeros(10)
    max = np.ones(10)
    for k in range(10):
        max[k] = float("-inf")

    for i in range(10):
        for j in range(6000):
#            print(transformed[j][i])
            if(transformed[j][i] > max[i]):
                max[i] = transformed[j][i]
                index[i] = j

    index = np.array(index, dtype = int)
    a = np.array(data[index[0]].reshape(28, 28))
    for i in range(0, 4):
        a = np.hstack((a, np.array(data[index[i+1]]).reshape(28, 28)))
    b = np.array(data[index[5]].reshape(28, 28))
    for i in range(0, 4):
        b = np.hstack((b, np.array(data[index[i+6]]).reshape(28, 28)))
    orig_img = np.array(np.vstack((a, b)), dtype = float)

#    plt.subplot(2,2,3)
#    plt.imshow(data[30].reshape(28,28), cmap=plt.get_cmap('gray'))
    plt.subplot(2, 2, 4)
    plt.imshow(orig_img, cmap=plt.get_cmap('gray'))
    plt.title("The original images that has the lagest value in those 10 dimension")

#    print(transformed.T.shape)

#    plt.legend()
#    plt.show()

