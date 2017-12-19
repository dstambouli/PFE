import numpy as np
import matplotlib.pyplot as pyplot


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

w1 = 2*np.random.random((3,1))-1
print("weights initi \n")
print(w1)
for i in range(10000):
    l0 = X
    l1 = sigmoid(np.dot(X, w1), False)
    err = Y - l1
    err_delta = err * sigmoid(l1, True)
    w1 += np.dot(l0.T,err_delta)
print("weights end \n")
print(w1)

w2 = 2*np.random.random((2,3))-1
print(w2)
mean = [0, 0]
cov = [[1, 0], [0, 100]
s1, s2 = np.random.multivariate_normal(mean, cov, 5000).T

plt.plot(s1, s2, 'x1')
plt.axis('equal')
plt.show()
