from __future__ import division
__author__ = "Music"
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')

def readData(file):
    # read the data from the file
    # the outputs are data and label matrix
    logging.info("Loading Data")
    f = open(file, 'r')
    X = []
    y = []
    for line in f:
        data = map(float, line.strip().split(','))
        X.append(data[:-1])
        y.append(int(data[-1]))
    X = np.array(X)
    y = np.array(y)
    # convert y to one-hot vectors Y
    Y = np.zeros((y.shape[0], 10))
    Y[np.arange(y.shape[0]), y] = 1

    logging.info("The shape of the data matrix is (%s, %s)", X.shape[0], X.shape[1])
    logging.info("The shape of the label matrix is (%s, %s)", Y.shape[0], Y.shape[1])


    return X, Y, y

def plotData(X):
    import matplotlib.pyplot as plt
    for i in xrange(600,603):
        print X[i]
        # Row by row
        image = np.reshape(X[i], (28,28))
        plt.imshow(image, cmap='Greys_r')
        plt.show()

def one_layer_NN(training_file):
    # Training one layer neural networks
    # Procesure should be:
    # 1. Load data X
    # 2. Initialize hidden layer weights [784 * 100], bias [1 * 100]
    # 3. Initialize output layer weights [100 * 10], bias [1 * 10]
    # 4. Forward propagation. Get the final output [data * 10]
    # 5. Define loss function. Calculate loss.
    # 6. Calculate derivatives for output pre-activation
    # 7. calculate derivatives for each W and b
    # 8. update W and b
    # repeat 4-8 as a training epoch
    # 9. When testing, use the same W and b, do forward propagation.
    np.random.seed(2016)
    epoch = 200
    learning_rate = 0.5
    X, Y, y = readData(training_file)
    # Single hidden layer, we have two W's and b's
    num_d, num_h0 = X.shape[0], X.shape[1] # data number and feature number
    num_h1, num_o = 100, 10 # hidden layer size and output layer size

    # Initialization
    W1 = initialize_weights(num_h0, num_h1)  # [h0 * h1]
    W2 = initialize_weights(num_h1, num_o)
    b1 = initialize_biases(num_h1)
    b2 = initialize_biases(num_o)

    # Training
    for i in xrange(epoch):
        [_, _, _, o] = fprop(X, W1, b1, W2, b2)
        #print o
        loss = - np.sum(np.log(o) * Y) / Y.shape[0]
        print "Average negative log-likelihood:", loss
        y_pred = predict(o)
        pre = precision(y_pred, y)
        # # print o
        print pre

        [a1, h1, a2, o] = fprop(X, W1, b1, W2, b2)
        #print sum(o[1,:])
        W2_g, b2_g, W1_g, b1_g = bprop(X, W1, b1, W2, b2, a1, h1, a2, o, Y)
        # update
        #print W1_g
        W2 -= W2_g * learning_rate
        b2 -= b2_g * learning_rate
        W1 -= W1_g * learning_rate
        b1 -= b1_g * learning_rate
        #print W2_g


    # Testing
    #[_,_,_,o] = fprop(X, W1, b1, W2, b2)
    #print o
    Y_pred = predict(o)

def predict(X):
    y = np.argmax(X, axis=1)
    return y

def precision(y_pred, y):
    pre = sum(y_pred == y) / len(y)
    return pre

def fprop(X, W1, b1, W2, b2):
    g = sigmoid
    a1 = np.dot(X, W1) + b1
    h1 = g(a1)
    a2 = np.dot(h1, W2) + b2
    o  = softmax(a2)
    return [a1, h1, a2, o]

def bprop(X, W1, b1, W2, b2, a1, h1, a2, o, Y):
    # computer gradients
    # softmax loss is defaulted here
    num_data = X.shape[0]

    a2_g = (o - Y) / num_data # gradient of softmax loss [data * o]
    #print h1.shape
    #print a2_g.shape

    W2_g = np.dot(h1.T, a2_g) # [h1 * data] [data * o]
    b2_g = np.sum(a2_g, axis=0, keepdims=True) # [1 * o]

    h1_g = np.dot(a2_g, W2.T)  # [data * o, o * h1]
    a1_g = h1_g * sigmoid_grad(a1)
    # a1_g = h1_g * (1-np.power(h1,2))  # [data * h1]
    W1_g = np.dot(X.T, a1_g) # [h0 * h1]
    b1_g = np.sum(a1_g, axis=0, keepdims=True) # [1 * o]

    return W2_g, b2_g, W1_g, b1_g

def softmax(X):
    # Softmax Function
    return np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)

def sigmoid(X):
    # f = np.vectorize(lambda x: 1 / (1 + np.exp(-x)), otypes=[np.float])  # ReLU function
    # return f(X)
    #X = np.clip(X, -500, 500)
    return 1 / (1+np.exp(-X))

def sigmoid_grad(X):
    return sigmoid(X) * sigmoid(1-X)

def ReLU(X):
    # Rectified Linear Activation function
    f = np.vectorize(lambda x: x if x > 0 else 0, otypes=[np.float]) # ReLU function
    return f(X)

def ReLU_grad(X):
    # Gradient for Rectified Linear Activation function
    f = np.vectorize(lambda x: 1 if x > 0 else 0, otypes=[np.float])  # ReLU function
    return f(X)

def initialize_weights(h_prev, h):
    # sample Wk from U[-b,b] where b = sqr(6) / sqr(h + h_prev)
    # W_k: [h * h_prev]
    b = np.sqrt(6) / np.sqrt(h + h_prev)
    return np.random.uniform(low = -b, high = b, size=[h_prev, h])

def initialize_biases(h):
    return np.zeros((1,h))



if __name__ == "__main__":
    training_file = "data/digitstrain.txt"
    one_layer_NN(training_file)
    #plotData(X)

