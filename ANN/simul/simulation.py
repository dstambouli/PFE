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
class Ann(object):
    def __init__(self):
        self.inputSize = 3
        self.outputSize = 1
        self.hiddenSize = 4
        # weights from a standard normal distribution (mean = 0, std =1)
        self.w1 = np.random.randn(self.inputSize, self.hiddenSize)
        self.w2 = np.random.rand(self.hiddenSize, self.outputSize)

    def sigmoid(self, x, deriv):
        if(deriv):
            return self.sigmoid(x)*(self.sigmoid(x)-1)
        else:
            return 1/(1+np.exp(-x))

    def forwardStep(self, X):
        self.signal1 = np.dot(X, self.w1)
        self.layer1 = self.sigmoid(self.signal1, False)
        self.signal2 = np.dot(self.layer1, self.w2)
        resOut = self.sigmoid(self.signal2, False)
        return resOut
    def CostDeriv(self, X, Y):
        yOut = self.forwardStep(X)
        deltaOut = np.multiply(-(Y-yOut), self.sigmoid(self.signal2,True))
        dCostdw2 = np.dot(self.layer1.T, deltaOut)


nn = Ann()
ysortie = nn.forwardStep(X_train)
print(ysortie - Y_train)
