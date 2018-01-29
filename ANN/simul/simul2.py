import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from scipy import optimize

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
class Neural_Network(object):
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
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))

    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return self.sigmoid(z)*(self.sigmoid(z)-1)

    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        return J

    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)

        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2



nn = Neural_Network()
