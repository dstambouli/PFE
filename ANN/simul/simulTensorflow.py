import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
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
Y_test = Y_test.values
log = []
x = tf.placeholder(tf.float32, [None, 3])
w = tf.Variable(tf.random_normal([3,4], stddev=0.25), name="weights")

y = tf.matmul(x,w)
yt = tf.placeholder(tf.float32,[None,1])
ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, yt))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializzer().run()
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(yt,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for i in range(6000):
    bat_xs = train[['b', 'x1', 'x2']]
    bat_ys = train.iloc[:,3]
    train_acc = accuracy.eval(feed_dict = {x: bat_xs, yt: bat_ys})
    log.append(1-train_acc)
    sess.run(train_step,feed_dict = {x: bat_xs, yt: bat_ys} )

print("test", sess.run(accuracy,feed_dict = {x: test[['b', 'x1', 'x2']], yt: test.iloc[:,3]} ))
ind = np.array([i for i in range(6000)])
log_tr = np.array(log)
plt.plot(ind, log)

plt.show()
