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

def train_one_layer(train_file, epoch=1000, learning_rate=0.5, seed=2016, hidden_size=[200],
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
        err_trains = []
        err_vals = []
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

        [a1, h1, a2, o] = fprop(X, model, dropout)
        # update
        W2_g, b2_g, W1_g, b1_g = bprop(X, model, a1, h1, a2, o, Y, L=L, L_lambda=L_lambda)
        W2 -= (W2_g + momentum * pre_W2_g) * learning_rate
        b2 -= (b2_g + momentum * pre_b2_g) * learning_rate
        W1 -= (W1_g + momentum * pre_W1_g) * learning_rate
        b1 -= (b1_g + momentum * pre_b1_g) * learning_rate

        pre_W2_g, pre_b2_g, pre_W1_g, pre_b1_g = W2_g, b2_g, W1_g, b1_g

    if plot:
        #plot_train_val_loss(loss_trains, loss_vals)
        #plot_train_val_err(err_trains, err_vals)
        plot_W(W1)
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


def plot_W(W):
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
    plt.show()


def plot_train_val_loss(trains, vals):
    import matplotlib.pyplot as plt
    from matplotlib.legend_handler import HandlerLine2D
    l1,=plt.plot(trains, label = 'Train')
    l2,=plt.plot(vals, label = 'Validation')
    plt.ylabel('Cross-entropy error')
    plt.xlabel('#Epochs')
    plt.legend()
    plt.show()

def plot_train_val_err(trains, vals):
    import matplotlib.pyplot as plt
    from matplotlib.legend_handler import HandlerLine2D
    l1,=plt.plot(trains, label = 'Train')
    l2,=plt.plot(vals, label = 'Validation')
    plt.ylabel('Classification Error')
    plt.xlabel('#Epochs')
    plt.legend()
    plt.show()

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



if __name__ == "__main__":
    train_file = "data/digitstrain.txt"
    val_file = "data/digitsvalid.txt"
    test_file = "data/digitstest.txt"
    train_one_layer(train_file, epoch=2000, learning_rate=0.1, seed=0, hidden_size=[50],
                    output_size=10, show_stats=True, val_file=val_file, test_file=test_file,
                    L="L2", L_lambda=0.001, plot=True, display_epoch=10, momentum=0.5, dropout=0)
    #plotData(X)

    #train_two_layer(train_file, epoch=2000, learning_rate=0.5, seed=2016, hidden_size=[100,100],
    #                output_size=10, show_stats=True, val_file=val_file, test_file=test_file,
    #                L="L2", L_lambda=0.002, plot=True, display_epoch=10, momentum=0.5, dropout=0)
