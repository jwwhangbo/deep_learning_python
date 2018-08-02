from __future__ import print_function, division
from builtins import range
# Inspired by https://github.com/lazyprogrammer/machine_learning_examples/ann_class2/dropout_tensorflow.py
# Dropout regularized ann with save functionality - save function developed by John Whangbo

# "For the class Data Science: Practical Deep Learning Concepts in Theano and TensorFlow"
# "https://deeplearningcourses.com/c/data-science-deep-learning-in-theano-tensorflow"
# "https://www.udemy.com/data-science-deep-learning-in-theano-tensorflow"
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from util import get_normalized_data
from sklearn.utils import shuffle

def error_rate(p, t):
    return np.mean(p != t)

class HiddenLayer(object):
    def __init__(self, M1, M2, f):
        self.M1 = M1
        self.M2 = M2
        self.f = f
        W = np.random.randn(M1, M2) * np.sqrt(2.0 / M1)
        b = np.zeros(M2)
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))
        self.params = [self.W, self.b]

    def forward(self, X):
        return self.f(tf.matmul(X, self.W) + self.b)


class ANN(object):
    def __init__(self, hidden_layer_sizes, p_keep, savefile):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_rates = p_keep
        self.savefile = savefile

    # defines self.session as input: session
    def set_session(self, session):
        self.session = session

    def fit(self, X, Y, Xvalid, Yvalid, activation=tf.nn.relu, lr=1e-4, mu=0.9, decay=0.9, epochs=5, batch_sz=100, print_every=50):
        X = X.astype(np.float32)
        Y = Y.astype(np.int64)
        Xvalid = Xvalid.astype(np.float32)
        Yvalid = Yvalid.astype(np.int64)

        # initialize hidden layers
        N, D = X.shape
        K = len(set(Y))
        self.hidden_layers = []
        M1 = D
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayer(M1, M2, activation)
            self.hidden_layers.append(h)
            M1 = M2
        W = np.random.randn(M1, K) * np.sqrt(2.0 / M1)
        b = np.zeros(K)
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))

        # collect params for later use
        self.params = []
        for h in self.hidden_layers:
            self.params += h.params
        
        # set up theano functions and variables
        inputs = tf.placeholder(tf.float32, shape=(None, D), name='inputs')
        labels = tf.placeholder(tf.int64, shape=(None,), name='labels')
        
        # for later use
        self.inputs = inputs

        # initiate saver
        self.saver = tf.train.Saver()

        # for testing
        test_logit = self.forward_test(inputs)

        test_cost =  tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=test_logit,
                labels=labels
            )
        )

        # for training
        logits = self.forward(inputs)

        cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=labels
            )
        )

        self.predict_op = tf.argmax(test_logit, 1, name="predict_op")

        train_op = tf.train.RMSPropOptimizer(lr, decay=decay, momentum=mu).minimize(cost)
        # train_op = tf.train.MomentumOptimizer(lr, momentum=mu).minimize(cost)
        # train_op = tf.train.AdamOptimizer(lr).minimize(cost)
        # prediction = self.predict(inputs)

        # validation cost will be calculated separately since nothing will be dropped
        

        n_batches = N // batch_sz
        costs = []
        self.session.run(tf.global_variables_initializer())
        for i in range(epochs):
            print("epoch:", i, "n_batches:", n_batches)
            X, Y = shuffle(X, Y)
            for j in range(n_batches):
                Xbatch = X[j*batch_sz:(j*batch_sz+batch_sz)]
                Ybatch = Y[j*batch_sz:(j*batch_sz+batch_sz)]

                self.session.run(train_op, feed_dict={inputs: Xbatch, labels: Ybatch})

                if j % print_every == 0:
                    c , p = self.session.run([test_cost, self.predict_op], feed_dict={inputs: Xvalid, labels: Yvalid})
                    costs.append(c)
                    e = error_rate(Yvalid, p)
                    print("i:", i, "j:", j, "nb:", n_batches, "cost:", c, "error rate:", e)
            print("Train acc:", self.score(X, Y), "Test acc:", self.score(Xvalid, Yvalid))

        self.saver.save(self.session, self.savefile)

        # plot costs
        plt.plot(costs)
        plt.show()

    def forward(self, X):
        # tf.nn.dropout scales inputs by 1/p_keep
        # therefore, during test time, we don't have to scale anything
        Z = X
        Z = tf.nn.dropout(Z, self.dropout_rates[0])
        for h, p in zip(self.hidden_layers, self.dropout_rates[1:]):
            Z = h.forward(Z)
            Z = tf.nn.dropout(Z, p)
        return tf.matmul(Z, self.W) + self.b

    def forward_test(self, X):
        Z = X
        for h in self.hidden_layers:
            Z = h.forward(Z)
        return tf.matmul(Z, self.W) + self.b

    def predict(self, X):
        return self.session.run(self.predict_op, feed_dict={self.inputs: X})

    def score(self, X, Y):
        e = error_rate(self.predict(X), Y)
        return 1 - e

    def save(self, filename):
        j = {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'p_keep': self.dropout_rates,
            'model' : self.savefile
        }
        with open(filename, 'w') as f: 
            json.dump(j, f)

    def restore_model(self):
        new_saver = tf.train.import_meta_graph(self.savefile + '.meta')
        new_saver.restore(self.session, tf.train.latest_checkpoint('./'))
        graph = tf.get_default_graph()
        self.predict_op = graph.get_tensor_by_name("predict_op:0")
        self.inputs = graph.get_tensor_by_name("inputs:0")

    @staticmethod
    def load(session, filename):
        with open(filename) as f:
            j = json.load(f)
        return ANN(j['hidden_layer_sizes'], j['p_keep'], j['model'])
        
    


def main():
    # step 1: get the data and define all the usual variables
    Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()

    ann = ANN([500, 300], [0.8, 0.5, 0.5], './tf.model')

    session = tf.InteractiveSession()
    ann.set_session(session)
    ann.fit(Xtrain, Ytrain, Xtest, Ytest)

    print("final train accuracy:", ann.score(Xtrain, Ytrain))
    print("final test accuracy:", ann.score(Xtest, Ytest))

    ann.save("my_saved_model.json")

    session.close()

    sess = tf.InteractiveSession()

    model = ANN.load(sess, "my_saved_model.json")
    model.set_session(sess)
    model.restore_model()
    print("final train accuracy (after reload):", model.score(Xtrain, Ytrain))
    print("final test accuracy (after reload):", model.score(Xtest, Ytest))


if __name__ == '__main__':
    main()
