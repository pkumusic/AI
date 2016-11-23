from __future__ import division
__author__ = "Music"
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')

import cPickle as pickle

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

def train_one_layer(train_file, file_name="",W1=None, W2=None, epoch=1000, learning_rate=0.5, seed=2016, hidden_size=[200],
                    output_size=10, show_stats=True, val_file=None, test_file=None,
                    L=False, L_lambda=0, plot=True, display_epoch=10, momentum=0.5, dropout=0.5):
    model = {}
    np.random.seed(seed)
    X, Y, y = readData(train_file)
    if val_file:
        X_val, Y_val, y_val = readData(val_file)
    if test_file:
        X_test, Y_test, y_test = readData(test_file)
    # Single hidden layer, we have two W's and b's
    num_d, num_h0 = X.shape[0], X.shape[1] # data number and feature number
    [num_h1]  = hidden_size # hidden layer size and output layer size
    num_o = output_size
    # Initialization
    if W1 == None:
        W1 = initialize_weights(num_h0, num_h1)  # [h0 * h1]
    if W2 == None:
        W2 = initialize_weights(num_h1, num_o)
    b1 = initialize_biases(num_h1)
    b2 = initialize_biases(num_o)
    model['W1'], model['W2'], model['b1'], model['b2'] = W1, W2, b1, b2
    pre_W2_g, pre_b2_g, pre_W1_g, pre_b1_g = np.zeros(W2.shape),np.zeros(b2.shape),np.zeros(W1.shape), np.zeros(b1.shape)

    # Stats collector
    if show_stats:
        loss_trains = []
        loss_vals = []
        loss_tests = []
        err_trains = []
        err_vals = []
        err_tests = []
    # Training

    for i in xrange(epoch):
        # Show stats
        if show_stats:
            [_, _, _, o] = fprop(X, model, dropout, deterministic=True)
            loss_train = loss(o, Y)
            err_train = precision(predict(o), y)
            #print i, "th Training loss:", loss_train
            #print precision(predict(o), y)
            if val_file:
                [_, _, _, o_val] = fprop(X_val, model, dropout, deterministic=True)
                loss_val = - np.sum(np.log(o_val) * Y_val) / Y_val.shape[0]
                err_val = precision(predict(o_val), y_val)
                loss_vals.append(loss_val)
                err_vals.append(err_val)

            if test_file:
                [_, _, _, o_test] = fprop(X_test, model, dropout, deterministic=True)
                loss_test = - np.sum(np.log(o_test) * Y_test) / Y_test.shape[0]
                err_test = precision(predict(o_test), y_test)
                loss_tests.append(loss_test)
                err_tests.append(err_test)

            if i % display_epoch == 0:
                print i, "th Training loss:", loss_train
                print precision(predict(o), y)
                if val_file:
                    print i, "th validation loss:", loss_val
                    print precision(predict(o_val), y_val)
                if test_file:
                    print i, "th test loss:", loss_test
                    print precision(predict(o_test), y_test)


            loss_trains.append(loss_train)
            err_trains.append(err_train)

        [a1, h1, a2, o] = fprop(X, model, dropout)
        # update
        W2_g, b2_g, W1_g, b1_g = bprop(X, model, a1, h1, a2, o, Y, L=L, L_lambda=L_lambda)
        W2 -= (W2_g + momentum * pre_W2_g) * learning_rate
        b2 -= (b2_g + momentum * pre_b2_g) * learning_rate
        W1 -= (W1_g + momentum * pre_W1_g) * learning_rate
        b1 -= (b1_g + momentum * pre_b1_g) * learning_rate

        pre_W2_g, pre_b2_g, pre_W1_g, pre_b1_g = W2_g, b2_g, W1_g, b1_g

    if plot:
        #plot_train_val_loss(loss_trains, loss_vals, "pretrain")
        plot_train_val_err(err_trains, err_vals, file_name)
        #plot_W(W1, "pretrain")
    return model


def train_two_layer(train_file, epoch=1000, learning_rate=0.5, seed=2016, hidden_size=[100,100],
                    output_size=10, show_stats=True, val_file=None, test_file=None,
                    L=False, L_lambda=0, plot=True, display_epoch=10, momentum=0.5, dropout=0.5):
    model = {}
    np.random.seed(seed)
    X, Y, y = readData(train_file)
    if val_file:
        X_val, Y_val, y_val = readData(val_file)
    if test_file:
        X_test, Y_test, y_test = readData(test_file)
    # Single hidden layer, we have two W's and b's
    num_d, num_h0 = X.shape[0], X.shape[1] # data number and feature number
    [num_h1, num_h2]  = hidden_size # hidden layer size and output layer size
    num_o = output_size
    # Initialization
    W1 = initialize_weights(num_h0, num_h1)  # [h0 * h1]
    W2 = initialize_weights(num_h1, num_h2)
    W3 = initialize_weights(num_h2, num_o)
    b1 = initialize_biases(num_h1)
    b2 = initialize_biases(num_h2)
    b3 = initialize_biases(num_o)
    model['W1'], model['W2'], model['W3'], model['b1'], model['b2'], model['b3'] = W1, W2, W3, b1, b2, b3
    pre_W2_g, pre_b2_g, pre_W1_g, pre_b1_g, pre_W3_g, pre_b3_g = np.zeros(W2.shape),np.zeros(b2.shape),np.zeros(W1.shape), np.zeros(b1.shape), np.zeros(W3.shape), np.zeros(b3.shape)

    # Stats collector
    if show_stats:
        loss_trains = []
        loss_vals = []
        err_trains = []
        err_vals = []
    # Training

    for i in xrange(epoch):
        # Show stats
        if show_stats:
            [_, _, _, _, _, o] = fprop2(X, model, dropout, deterministic=True)
            loss_train = loss(o, Y)
            err_train = precision(predict(o), y)
            #print i, "th Training loss:", loss_train
            #print precision(predict(o), y)
            if val_file:
                [_, _, _,_,_, o_val] = fprop2(X_val, model, dropout, deterministic=True)
                loss_val = - np.sum(np.log(o_val) * Y_val) / Y_val.shape[0]
                err_val = precision(predict(o_val), y_val)
                loss_vals.append(loss_val)
                err_vals.append(err_val)

            if test_file:
                [_, _, _,_,_, o_test] = fprop2(X_test, model, dropout, deterministic=True)
                loss_test = - np.sum(np.log(o_test) * Y_test) / Y_test.shape[0]
                err_test = precision(predict(o_test), y_test)
                loss_vals.append(loss_test)
                err_vals.append(err_test)

            if i % display_epoch == 0:
                print i, "th Training loss:", loss_train
                print precision(predict(o), y)
                if val_file:
                    print i, "th validation loss:", loss_val
                    print precision(predict(o_val), y_val)
                if test_file:
                    print i, "th test loss:", loss_test
                    print precision(predict(o_test), y_test)


            loss_trains.append(loss_train)
            err_trains.append(err_train)

        [a1, h1, a2, h2, a3, o] = fprop2(X, model, dropout)
        # update
        W3_g, b3_g, W2_g, b2_g, W1_g, b1_g = bprop2(X, model, a1, h1, a2, h2, a3, o, Y, L=L, L_lambda=L_lambda)
        W3 -= (W3_g + momentum * pre_W3_g) * learning_rate
        b3 -= (b3_g + momentum * pre_b3_g) * learning_rate
        W2 -= (W2_g + momentum * pre_W2_g) * learning_rate
        b2 -= (b2_g + momentum * pre_b2_g) * learning_rate
        W1 -= (W1_g + momentum * pre_W1_g) * learning_rate
        b1 -= (b1_g + momentum * pre_b1_g) * learning_rate

        pre_W3_g, pre_b3_g, pre_W2_g, pre_b2_g, pre_W1_g, pre_b1_g = W3_g, b3_g, W2_g, b2_g, W1_g, b1_g

    if plot:
        #plot_train_val_loss(loss_trains, loss_vals)
        #plot_train_val_err(err_trains, err_vals)
        plot_W(W1)
    return model


def plot_W(W, file_name):
    # 784 * 100
    import matplotlib.pyplot as plt
    import numpy.random as rnd
    W = np.transpose(W)
    W = np.reshape(W, (-1,28,28))

    fig = plt.figure()
    for i in xrange(W.shape[0]):
        plt.subplot(10,10,i+1)
        plt.axis('off')
        plt.imshow(W[i],cmap=plt.cm.binary)
    #plt.show()
    fig.savefig(file_name+'_W.png')


def plot_train_val_loss(trains, vals, file_name):
    import matplotlib.pyplot as plt
    from matplotlib.legend_handler import HandlerLine2D
    fig = plt.figure()
    l1,=plt.plot(trains, label = 'Train')
    l2,=plt.plot(vals, label = 'Validation')
    plt.ylabel('Cross-entropy error')
    plt.xlabel('#Epochs')
    plt.legend()
    #plt.show()
    fig.savefig(file_name+'_loss.png')

def plot_train_val_err(trains, vals, file_name):
    import matplotlib.pyplot as plt
    from matplotlib.legend_handler import HandlerLine2D
    fig = plt.figure()
    l1,=plt.plot(trains, label = 'Train')
    l2,=plt.plot(vals, label = 'Validation')
    plt.ylabel('Classification Error')
    plt.xlabel('#Epochs')
    plt.legend()
    #plt.show()
    fig.savefig(file_name + '_error.png')

def loss(predict, true):
    return - np.sum(np.log(predict) * true) / true.shape[0]

def predict(X):
    y = np.argmax(X, axis=1)
    return y

def precision(y_pred, y):
    pre = 1 - sum(y_pred == y) / len(y)
    return pre

def fprop(X, model, dropout, deterministic=False):
    # Deterministic is True for testing and false for training
    W1, W2, b1, b2 = model['W1'], model['W2'], model['b1'], model['b2']
    g = sigmoid
    a1 = np.dot(X, W1) + b1

    if not deterministic:
        # Training Time
        m1 = np.random.choice(2, a1.shape, p=[dropout, 1-dropout])
    else:
        m1 = np.empty(a1.shape)
        m1.fill(1-dropout)
    h1 = g(a1) * m1
    a2 = np.dot(h1, W2) + b2
    o  = softmax(a2)
    return [a1, h1, a2, o]

def fprop2(X, model, dropout, deterministic=False):
    # Deterministic is True for testing and false for training
    W1, W2, W3, b1, b2, b3 = model['W1'], model['W2'], model['W3'], model['b1'], model['b2'], model['b3']
    g = sigmoid
    a1 = np.dot(X, W1) + b1
    if not deterministic:
        # Training Time
        m1 = np.random.choice(2, a1.shape, p=[dropout, 1-dropout])
    else:
        m1 = np.empty(a1.shape)
        m1.fill(1-dropout)
    h1 = g(a1) * m1
    a2 = np.dot(h1, W2) + b2
    if not deterministic:
        # Training Time
        m2 = np.random.choice(2, a2.shape, p=[dropout, 1-dropout])
    else:
        m2 = np.empty(a2.shape)
        m2.fill(1-dropout)
    h2 = g(a2) * m2
    a3 = np.dot(h2, W3) + b3
    o  = softmax(a3)
    return [a1, h1, a2, h2, a3, o]

def bprop(X, model, a1, h1, a2, o, Y, L=False, L_lambda=0):
    # computer gradients
    # softmax loss is defaulted here
    W1, W2, b1, b2 = model['W1'], model['W2'], model['b1'], model['b2']
    num_data = X.shape[0]

    a2_g = (o - Y) / num_data # gradient of softmax loss [data * o]
    W2_g = np.dot(h1.T, a2_g) # [h1 * data] [data * o]
    b2_g = np.sum(a2_g, axis=0, keepdims=True) # [1 * o]

    h1_g = np.dot(a2_g, W2.T)  # [data * o, o * h1]
    a1_g = h1_g * sigmoid_grad(a1)
    W1_g = np.dot(X.T, a1_g) # [h0 * h1]
    b1_g = np.sum(a1_g, axis=0, keepdims=True) # [1 * o]

    if L == "L2":
        W2_g += L_lambda * 2 * W2
        W1_g += L_lambda * 2 * W1
    if L == "L1":
        W2_g += L_lambda * np.sign(W2)
        W1_g += L_lambda * np.sign(W1)

    return W2_g, b2_g, W1_g, b1_g

def bprop2(X, model, a1, h1, a2, h2, a3, o, Y, L=False, L_lambda=0):
    # computer gradients
    # softmax loss is defaulted here
    W1, W2, W3, b1, b2, b3 = model['W1'], model['W2'], model['W3'], model['b1'], model['b2'], model['b3']
    num_data = X.shape[0]

    a3_g = (o - Y) / num_data  # gradient of softmax loss [data * o]
    W3_g = np.dot(h2.T, a3_g)  # [h1 * data] [data * o]
    b3_g = np.sum(a3_g, axis=0, keepdims=True)  # [1 * o]

    h2_g = np.dot(a3_g, W3.T)  # [data * o, o * h1]
    a2_g = h2_g * sigmoid_grad(a2) # gradient of softmax loss [data * o]
    W2_g = np.dot(h1.T, a2_g) # [h1 * data] [data * o]
    b2_g = np.sum(a2_g, axis=0, keepdims=True) # [1 * o]

    h1_g = np.dot(a2_g, W2.T)  # [data * o, o * h1]
    a1_g = h1_g * sigmoid_grad(a1)
    W1_g = np.dot(X.T, a1_g) # [h0 * h1]
    b1_g = np.sum(a1_g, axis=0, keepdims=True) # [1 * o]

    if L == "L2":
        W3_g += L_lambda * 2 * W3
        W2_g += L_lambda * 2 * W2
        W1_g += L_lambda * 2 * W1

    if L == "L1":
        W3_g += L_lambda * np.sign(W3)
        W2_g += L_lambda * np.sign(W2)
        W1_g += L_lambda * np.sign(W1)

    return W3_g, b3_g, W2_g, b2_g, W1_g, b1_g

def softmax(X):
    # Softmax Function
    return np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)

def sigmoid(X):
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

def rbm_sample(path):
    model = {}
    model['c'] = np.load("c_" + path)
    model['b'] = np.load("b_" + path)
    model['W'] = np.load("W_" + path)
    X = np.empty((784, 100))
    X.fill(0.5)
    X = np.random.binomial(1, X)
    x, h = gibbs_sampling(X, model, k=1000)
    plot_W(x, "rbm_sampling")

def train_rbm(train_file, val_file=None, save_W=False, batch_size=10, hidden_size=100, epoch=100, learning_rate=0.1, seed=1, CD_k=1):
    model = {}
    np.random.seed(seed)
    X, Y, y = readData(train_file)
    X_val, _, _ = readData(val_file)
    X = binarize(X)  # 3000 * 784
    train_size = X.shape[0]
    iter_times = int(train_size / batch_size)
    num_x = X.shape[1]
    num_h = hidden_size
    X = X.T          # 784 * 3000
    X_val = X_val.T
    W = initialize_weights(num_h, num_x)  # 100 * 784
    c = initialize_biases(num_x).reshape((num_x,1)) # bias for x  # 784 * 1
    b = initialize_biases(num_h).reshape((num_h,1)) # bias for h  # 100 * 1
    model['W'], model['c'], model['b'] = W, c, b
    train_losses = []
    if val_file:
        val_losses   = []
    for t in xrange(epoch):
        train_loss = cross_entropy(model, X, num_x, CD_k)
        val_loss   = cross_entropy(model, X_val, num_x, CD_k)
        train_losses.append(train_loss)
        if val_file:
            val_losses.append(val_loss)
        print t, train_loss
        #for x in X:
        #    x = x.reshape((num_x, 1))
        # mini-batch version
        for i in xrange(iter_times):
            rows = np.random.permutation(train_size)[:batch_size]
            X_ = X[:, rows]
            X_neg, _ = gibbs_sampling(X_, model, k=CD_k)
            rbm_update(model, X_, X_neg, learning_rate)
    file_name = "h"+str(hidden_size)+"_l"+str(learning_rate)+"_e"+str(epoch)+"_b"+str(batch_size)+"_k"+str(CD_k)
    #plot_W(W.T, file_name)
    if val_file:
        plot_train_val_loss(train_losses, val_losses, file_name)
    if save_W:
        np.save('W_'+file_name, W)
        np.save('c_' + file_name, c)
        np.save('b_' + file_name, b)

# class PCD(object):
#     def __init__(self, K, num_v, num_h1, num_h2):
#         self.num_v = num_v
#         self.num_h1 = num_h1
#         self.num_h2 = num_h2




def train_dbm(train_file, val_file=None, save_W=False,
              batch_size=10, num_h1=100, num_h2=100,
              epoch=100, learning_rate=0.01, seed=1,
              K=100, mean_field_update=10, gibbs_update=1,
              use_bias=False):
    """
    :param K: number of gibbs chains
    :param mean_field_update: number of iteration time for update mu
    :param gibbs_update: number of iteration time for update gibbs chains
    :return:
    """
    model = {}
    np.random.seed(seed)
    X, Y, y = readData(train_file)
    X_val, _, _ = readData(val_file)
    X = binarize(X)  # 3000 * 784
    train_size = X.shape[0]
    iter_times = int(train_size / batch_size)
    num_v = X.shape[1]
    X = X.T  # 784 * 3000
    X_val = X_val.T
    W1 = initialize_weights(num_v, num_h1)  # 784 * 100
    W2 = initialize_weights(num_h1, num_h2)
    #TODO: Add bias

    # Initialize persistent chains
    v_neg =  np.random.binomial(1, 0.5, (num_v, K))
    h1_neg = np.random.binomial(1, 0.5, (num_h1, K))
    h2_neg = np.random.binomial(1, 0.5, (num_h2, K))
    model['W1'], model['W2'] = W1, W2
    train_losses = []
    if val_file:
        val_losses = []
    for t in xrange(epoch):
        train_loss = dbm_cross_entropy(model, X, num_h2)
        val_loss   = dbm_cross_entropy(model, X, num_h2)
        train_losses.append(train_loss)
        if val_file:
            val_losses.append(val_loss)
        print t, train_loss
        # mini-batch mode
        for i in xrange(iter_times):
            rows = np.random.permutation(train_size)[:batch_size]
            v = X[:, rows]
            v_neg, h1_neg, h2_neg = dbm_gibbs_sampling(v_neg, h1_neg, h2_neg, model, gibbs_update)
            mu1, mu2 = mean_field_dbm_gibbs_sampling(v, num_h1, num_h2, model, mean_field_update, )
            # Update model
            W1 += learning_rate * (v.dot(mu1.T)/v.shape[1] - v_neg.dot(h1_neg.T)/v_neg.shape[1])
            W2 += learning_rate * (mu1.dot(mu2.T) / mu1.shape[1] - h1_neg.dot(h2_neg.T) / h1_neg.shape[1])
    file_name = "h1" + str(num_h1) + '_h2' + str(num_h2)+\
                "_l" + str(learning_rate) + "_e" + str(epoch) + \
                "_b" + str(batch_size) + "_k" + str(K)
    if val_file:
        plot_train_val_loss(train_losses, val_losses, file_name)

def dbm_cross_entropy(model, v, num_h2):
    W1, W2 = model['W1'], model['W2']
    h2 = np.random.rand(num_h2, v.shape[1])
    accum = 0
    _, h1, _ = dbm_gibbs_sampling(v, h2, h2, model, 1)
    v_pred = sigmoid(np.dot(W1, h1))
    accum += np.sum(v * np.log(v_pred))
    accum += np.sum((1 - v) * np.log(1 - v_pred))
    accum /= v.shape[1]
    return -accum

def dbm_gibbs_sampling(v, h1, h2, model, iter_times):
    W1, W2 = model['W1'], model['W2']
    for i in xrange(iter_times):
        h1 = sigmoid(np.dot(W1.T, v) + np.dot(W2, h2))
        h1 = np.random.binomial(1, h1)
        h2 = sigmoid(np.dot(W2.T, h1))
        h2 = np.random.binomial(1, h2)
        v = sigmoid(np.dot(W1, h1))
        v = np.random.binomial(1, v)
    return v,h1,h2

def mean_field_dbm_gibbs_sampling(v, num_h1, num_h2, model, iter_times):
    W1, W2 = model['W1'], model['W2']
    batch_size = v.shape[1]
    mu1 = np.random.rand(num_h1, batch_size)
    mu2 = np.random.rand(num_h2, batch_size)
    for i in xrange(iter_times):
        mu1 = sigmoid(np.dot(W1.T, v) + np.dot(W2, mu2))
        mu2 = sigmoid(np.dot(W2.T, mu1))
    return mu1, mu2




def cross_entropy(model, X, num_x, CD_k):
    W, c, b = model['W'], model['c'], model['b']
    accum = 0
    _, h = gibbs_sampling(X, model, k=CD_k)
    accum += np.sum(X * np.log(sigmoid(c + np.dot(W.T,h))))
    accum += np.sum((1-X) * np.log(1-sigmoid(c + np.dot(W.T,h))))
    accum /= X.shape[1]
    return -accum

def rbm_update(model, x, x_neg, learning_rate):
    W, c, b = model['W'], model['c'], model['b']
    hx = sigmoid(b+np.dot(W,x)) # (100, 3000)
    hx_neg =sigmoid(b+np.dot(W,x_neg))
    W += learning_rate * (np.dot(hx, x.T) - np.dot(hx_neg, x_neg.T)) / x.shape[1]
    b += np.sum(learning_rate * (hx - hx_neg), axis=1).reshape(-1,1) / x.shape[1]
    c += np.sum(learning_rate * (x - x_neg), axis=1).reshape(-1,1) / x.shape[1]

def gibbs_sampling(x, model, k=1):
    W, c, b = model['W'], model['c'], model['b']
    for i in xrange(k):
        h = sigmoid(b + np.dot(W,x))
        h = np.random.binomial(1, h)
        x = sigmoid(c + np.dot(W.T,h))
        x = np.random.binomial(1, x)
    return x, h

def binarize(X):
    X[X>=0.5] = 1
    X[X<0.5]  = 0
    return X



def autoencoder(train_file, denoising=0.25, epoch=100, batch_size=10, learning_rate=0.1, seed=1, hidden_size=[100],
                    output_size=784, show_stats=True, val_file=None,
                    L=False, L_lambda=0, plot=True, display_epoch=10, momentum=0.5, dropout=0):
    model = {}
    np.random.seed(seed)
    X, _, _ = readData(train_file)
    X = binarize(X)
    Y = X  # Reconstruction
    if val_file:
        X_val, _, _ = readData(val_file)
        X_val = binarize(X_val)
        Y_val = X_val
    # Single hidden layer, we have two W's and b's
    num_d, num_h0 = X.shape[0], X.shape[1] # data number and feature number
    iter_times = int(num_d / batch_size)
    [num_h1]  = hidden_size # hidden layer size and output layer size
    num_o = output_size
    # Initialization
    W1 = initialize_weights(num_h0, num_h1)  # [h0 * h1]
    W2 = initialize_weights(num_h1, num_o)
    b1 = initialize_biases(num_h1)
    b2 = initialize_biases(num_o)
    model['W1'], model['W2'], model['b1'], model['b2'] = W1, W2, b1, b2
    pre_W2_g, pre_b2_g, pre_W1_g, pre_b1_g = np.zeros(W2.shape),np.zeros(b2.shape),np.zeros(W1.shape), np.zeros(b1.shape)

    # Stats collector
    if show_stats:
        loss_trains = []
        loss_vals = []
    # Training

    for i in xrange(epoch):
        # Show stats
        if show_stats:
            [_, _, _, o] = fprop_auto(X, model, dropout, deterministic=True)
            loss_train = loss_auto(o, Y)
            if val_file:
                [_, _, _, o_val] = fprop_auto(X_val, model, dropout, deterministic=True)
                loss_val = loss_auto(o_val, Y_val)
                loss_vals.append(loss_val)

            if i % display_epoch == 0:
                print i, "th Training loss:", loss_train
                if val_file:
                    print i, "th validation loss:", loss_val

            loss_trains.append(loss_train)

        # mini-batch
        for i in xrange(iter_times):
            rows = np.random.permutation(num_d)[:batch_size]
            X_ = X[rows, :]
            if denoising:
                m = np.random.choice(2, X_.shape, p=[denoising, 1-denoising])
                X_noise = m * X_
            Y_ = X_
            if denoising:
                [a1, h1, a2, o] = fprop_auto(X_noise, model, dropout)
            else:
                [a1, h1, a2, o] = fprop_auto(X_, model, dropout)
            # update
            if denoising:
                W2_g, b2_g, W1_g, b1_g = bprop_auto(X_noise, model, a1, h1, a2, o, Y_, L=L, L_lambda=L_lambda)
            else:
                W2_g, b2_g, W1_g, b1_g = bprop_auto(X_, model, a1, h1, a2, o, Y_, L=L, L_lambda=L_lambda)
            W2 -= (W2_g + momentum * pre_W2_g) * learning_rate
            b2 -= (b2_g + momentum * pre_b2_g) * learning_rate
            W1 -= (W1_g + momentum * pre_W1_g) * learning_rate
            b1 -= (b1_g + momentum * pre_b1_g) * learning_rate

            pre_W2_g, pre_b2_g, pre_W1_g, pre_b1_g = W2_g, b2_g, W1_g, b1_g

    if plot:
        file_name = "autoencoder_h" + str(learning_rate) + "_e" + str(epoch) + "_h" + str(hidden_size)
        if denoising:
            file_name = "deno_" + str(denoising) + file_name
        plot_train_val_loss(loss_trains, loss_vals, file_name)
        plot_W(W1, file_name)
        np.save("W1_"+file_name, W1)
        np.save("W2_"+file_name, W2)
    return model

def fprop_auto(X, model, dropout, deterministic=False):
    # Deterministic is True for testing and false for training
    W1, W2, b1, b2 = model['W1'], model['W2'], model['b1'], model['b2']
    g = sigmoid
    a1 = np.dot(X, W1) + b1

    if not deterministic:
        # Training Time
        m1 = np.random.choice(2, a1.shape, p=[dropout, 1-dropout])
    else:
        m1 = np.empty(a1.shape)
        m1.fill(1-dropout)
    h1 = g(a1) * m1
    a2 = np.dot(h1, W2) + b2
    o  = g(a2)
    return [a1, h1, a2, o]

def bprop_auto(X, model, a1, h1, a2, o, Y, L=False, L_lambda=0):
    # computer gradients
    # softmax loss is defaulted here
    W1, W2, b1, b2 = model['W1'], model['W2'], model['b1'], model['b2']
    num_data = X.shape[0]

    a2_g = - (sigmoid(1-a2) * Y - sigmoid(a2) * (1-Y))/ num_data # gradient of softmax loss [data * o]
    W2_g = np.dot(h1.T, a2_g) # [h1 * data] [data * o]
    b2_g = np.sum(a2_g, axis=0, keepdims=True) # [1 * o]

    h1_g = np.dot(a2_g, W2.T)  # [data * o, o * h1]
    a1_g = h1_g * sigmoid_grad(a1)
    W1_g = np.dot(X.T, a1_g) # [h0 * h1]
    b1_g = np.sum(a1_g, axis=0, keepdims=True) # [1 * o]

    if L == "L2":
        W2_g += L_lambda * 2 * W2
        W1_g += L_lambda * 2 * W1
    if L == "L1":
        W2_g += L_lambda * np.sign(W2)
        W1_g += L_lambda * np.sign(W1)

    return W2_g, b2_g, W1_g, b1_g

def loss_auto(predict, true):
    return - (np.sum(np.log(predict) * true) + np.sum(np.log(1-predict) * (1-true))) / true.shape[0]



if __name__ == "__main__":
    train_file = "../DNN/data/digitstrain.txt"
    val_file = "../DNN/data/digitsvalid.txt"
    test_file = "../DNN/data/digitstest.txt"
    train_dbm(train_file, val_file=val_file)
    #train_rbm(train_file, val_file=val_file, save_W=True)
    #autoencoder(train_file, val_file=val_file)
    #path = "h100_l0.1_e100_b10_k1.npy"
    #rbm_sample(path)
    #W1 = np.load('W1_deno_0.25autoencoder_h0.1_e100_h[100].npy')
    #W1 = np.load('W_h100_l0.1_e100_b10_k1.npy').T
    #W1 = np.load("W1_autoencoder_h0.1_e100_h[100].npy")
    #train_one_layer(train_file, W1=W1, file_name="auto", epoch=2000, learning_rate=0.1, seed=0, hidden_size=[100],
    #                output_size=10, show_stats=True, val_file=val_file, test_file=test_file,
    #                L=None, L_lambda=0, plot=True, display_epoch=10, momentum=0.5, dropout=0)
    #plotData(X)

    #train_two_layer(train_file, epoch=2000, learning_rate=0.5, seed=2016, hidden_size=[100,100],
    #                output_size=10, show_stats=True, val_file=val_file, test_file=test_file,
    #                L="L2", L_lambda=0.002, plot=True, display_epoch=10, momentum=0.5, dropout=0)
