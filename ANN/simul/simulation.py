import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(1)
# Charger les données
df = pd.read_csv('set1.csv', sep=';')

# split data 0.8 training, 0.2 test
train, test = train_test_split(df, test_size=0.2)
################################################################################
#                                Preprocessing
################################################################################

# récupérer les Input pour l'apprentissage
X_train = train[['b', 'x1', 'x2']]
X_train = X_train.values

# récupérer les Y pour l'apprentissage
Y_train = train.iloc[:,3]
Y_train = Y_train.values.reshape(len(Y_train),1)

# récupérer les Input pour le test
X_test = test[['b', 'x1', 'x2']]
X_test = X_test.values

# récupérer les Y pour le test
Y_test = test.iloc[:,3]
Y_test = Y_test.values.T

################################################################################
#                                     ANN
################################################################################

def sigmoid(x, deriv):
    if(deriv == True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

w1 = 2*np.random.random((3,4))-1
w2 = 2*np.random.random((4,1))-1
print("weights inti \n")
print(w1)
print(w2)
log_tr = list()

for i in range(100000):
    l0 = X_train
    l1 = sigmoid(np.dot(X_train, w1), False)
    #print("l1 shape \n",l1.shape)
    l2 = sigmoid(np.dot(l1,w2), False)
    #print("l2 shape \n",l2.shape)
    l2error = (Y_train - l2) + 0.2*np.linalg.norm(w2, ord = 2)
    #print("l2error\n", l2error.shape)
    l2_delta = np.multiply(l2error, sigmoid(l2, True))
    #print("l2delta\n",l2_delta.shape)
    log_tr.append(np.mean(np.abs(l2error)))
    l2_delta = l2error * sigmoid(l2, True)
    l1error = np.dot(l2_delta,w2.T) + 0.2*np.linalg.norm(w1, ord = 2)
    l1_delta = l1error* sigmoid(l1, True)
    w2 += np.dot(l1.T, l2_delta)
    w1 += np.dot(l0.T,l1_delta)

print("weights end \n")
print(w1)
print(w2)
print(np.trunc(l2[:10]))

ind = np.array([i for i in range(100000)])
log_tr = np.array(log_tr)
plt.plot(ind, log_tr)

plt.show()
