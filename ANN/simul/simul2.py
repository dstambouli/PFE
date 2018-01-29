_Derimport numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

np.random.seed(1)
# Charger les données
df = pd.read_csv('set1.csv', sep=';')

# split data 0.8 training, 0.2 test
train, test = train_test_split(df, test_size = 0.2)
################################################################################
#                                Preprocessing
################################################################################

# récupérer les Input pour l'apprentissage
X_train = train[['x1', 'x2']]
X_train = X_train.values
print("train shape : ",X_train.shape)
# récupérer les Y pour l'apprentissage
Y_train = train.iloc[:,2]
Y_train = Y_train.values.reshape(len(Y_train),1)

# récupérer les Input pour le test
X_test = test[['x1', 'x2']]
X_test = X_test.values

# récupérer les Y pour le test
Y_test = test.iloc[:,2]
Y_test = Y_test.values.T

################################################################################
#                                     ANN
################################################################################

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_Der(x):
    return sigmoid(x)*(sigmoid(x)-1)

class network(object):
        def __init__(self):
        #Define Hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        #Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)

    def forward(self, X):
        #Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = sigmoid(self.z3)
        return yHat

    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        return J

    def costFunction_Der(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)

        delta3 = np.multiply(-(y-self.yHat), sigmoid_Der(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T)*sigmoid_Der(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2
