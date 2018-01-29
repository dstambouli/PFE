import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras as k
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from keras import regularizers
np.random.seed(1)
# Charger les données
df = pd.read_csv('set1.csv', sep=';')

# split data 0.8 training, 0.2 test
train, test = train_test_split(df, test_size=0.2)
################################################################################
#                                Preprocessing
################################################################################

# récupérer les Input pour l'apprentissage
X_train = train[['x1', 'x2']]
X_train = X_train.values
X_train = preprocessing.scale(X_train)

# récupérer les Y pour l'apprentissage
Y_train = train.iloc[:,2]
Y_train = Y_train.values.reshape(len(Y_train),1)

# récupérer les Input pour le test
X_test = test[['x1', 'x2']]
X_test = X_test.values
X_test = preprocessing.scale(X_test)

# récupérer les Y pour le test
Y_test = test.iloc[:,2]
Y_test = Y_test.values



###############################################################################
#                   network
###############################################################################
# create model
model = Sequential()
model.add(Dense(3, input_dim=2, kernel_regularizer=regularizers.l2(0.05), activation='sigmoid'))
#model.add(Dense(3, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
sgd = optimizers.SGD(lr=0.7, clipnorm=1.)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mean_squared_error'])
# Fit the model
k = model.fit(x=X_train, y=Y_train, batch_size= 40, epochs=1000, verbose=0)
# evaluate the model
score = model.evaluate(X_test, Y_test)
print('Test score:', score[1])

print("weights")
for w in model.get_weights():
    print(w)
    print('\n')
