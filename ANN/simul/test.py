import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x, deriv):
    if(deriv == True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

X = np.array([ [0,0,1],
               [0,1,1],
               [1,0,1],
               [1,1,1] ])
Y = np.array([[0,0,1,1]]).T
np.random.seed(1)

w1 = 2*np.random.random((3,4))-1
w2 = 2*np.random.random((4,1))-1
log = list()
print("weights initi \n")
print(w1)
print(w2)
for i in range(60000):
    l0 = X
    l1 = sigmoid(np.dot(X, w1), False)
    l2 = sigmoid(np.dot(l1,w2), False)
    err2 = Y - l2
    log.append(np.mean(np.abs(err2)))
    err_delta1 = err2 * sigmoid(l2, True)
    err1 = err_delta1.dot(w2.T)
    err_delta2 = err1* sigmoid(l1, True)
    w2 += np.dot(l1.T, err_delta1)
    w1 += np.dot(l0.T,err_delta2)
print("weights end \n")
print(w1)
print(w2)
print(l2)

ind = np.array([i for i in range(60000)])
log = np.array(log)
plt.plot(ind, log)
plt.legend()
plt.show()
