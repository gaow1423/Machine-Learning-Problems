import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import Kmean_class

if __name__ == '__main__':
    a = Kmean_class.kmean('data-1.txt', 2)
    arr_sse = a.itestop()
    plt.subplot(1, 2, 1)
    plt.plot(np.array(range(1, a.rtime+1)), arr_sse)
    plt.title('SSE as a function of the iterations')
    plt.xlabel('Iterations')
    plt.ylabel('SSE')
    
    plt.subplot(1, 2, 2)
    for i in range(2, 11):
        a.setk(i)
        a.iteset = 1
        arr_sse = a.itestop()
        plt.plot(np.array(range(1, 11)), arr_sse, label = "k = %f"%(i))
    plt.title('SSE as a function of the iterations')
    plt.xlabel('Iterations')
    plt.ylabel('SSE')
    plt.legend()
    plt.show()

