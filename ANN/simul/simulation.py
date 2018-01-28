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
X_train = train[['x1', 'x2']]
X_train = X_train.values

# récupérer les Y pour l'apprentissage
Y_train = train.iloc[:,3]
Y_train = Y_train.values.reshape(len(Y_train),1)

# récupérer les Input pour le test
X_test = test[['x1', 'x2']]
X_test = X_test.values

# récupérer les Y pour le test
Y_test = test.iloc[:,3]
Y_test = Y_test.values.T

################################################################################
#                                     ANN
################################################################################

def sigmoid(self, x, deriv):
    if(deriv):
        return sigmoid(x)*(sigmoid(x)-1)
    else:
        return 1/(1+np.exp(-x))

class Ann(object):
    def __init__(self, tailles):
        """
            - tailles : list with number of neurones per layer
            - no need for a bias vector for the input layer
            - weiths are initialized with a gaussian with mean = 0 and std = 1
        """
        self.nbCouches = len(tailles)
        self.tailles = tailles
        self.bias = [np.random.randn(j, 1) for j in tailles[1:]]
        self.weights = [np.random.randn(i,j)
                        for i, j in zip(tailles[:-1], tailles[1:])]


    def forwardStep(self, a):
          for b, w in zip(self.bias, self.weights):
            a = sigmoid(np.dot(w, a)+b, False)
            return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
        test_data=None):
    """
        - Train using mini-batch stochastic gradient descent.
        - The training_data is a list of tuples with the training inputs and
          the desired outputs.
        - eta the learning rate
        - if test_data specified then we use it with the neural n_test
    """
    if test_data: n_test = len(test_data)
    n = len(training_data)
    for j in range(epochs):
        random.shuffle(training_data)
        mini_batches = [
            training_data[k:k+mini_batch_size]
            for k in range(0, n, mini_batch_size)]
        for mini_batch in mini_batches:
            self.update_mini_batch(mini_batch, eta)
        if test_data:
            print "Epoch {0}: {1} / {2}".format(
                j, self.evaluate(test_data), n_test)
        else:
            print "Epoch {0} complete".format(j)
    def update_mini_batch(self, mini_batch, eta):
    """
        - Update the network's weights and biases by applying GD
           and backpropagation to a single mini batch
        - eta the learning rate
    """
    nabla_b = [np.zeros(b.shape) for b in self.bias]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    for x, y in mini_batch:
        delta_nabla_b, delta_nabla_w = self.backprop(x, y)
        nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
    self.weights = [w-(eta/len(mini_batch))*nw
                    for w, nw in zip(self.weights, nabla_w)]
    self.biases = [b-(eta/len(mini_batch))*nb
                    for b, nb in zip(self.biases, nabla_b)]
    def backprop(self, x, y):
        """
            - Return a tuple (nabla_b, nabla_w) : the gradient for the cost
              function.
            - nabla_b and nabla_w are layer-by-layer lists of numpy arrays
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        # list to store all the activations, layer by layer
        activations = [x]
        # list to store all the z vectors, layer by layer : the input signal for the layer
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = np.multiply(self.cost_derivative(activations[-1], y), sigmoid(zs[-1], True))
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z, True)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            return (nabla_b, nabla_w)

    """def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.forward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)"""

     def cost_derivative(self, output_act, y):
        return (output_act-y)


nn = Ann([2,3,1])

yhat = nn.forwardStep(X_train)
print(yhat)
