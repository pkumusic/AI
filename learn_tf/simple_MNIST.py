__author__ = "Music"
# MNIST For Experts
# https://www.tensorflow.org/versions/r0.9/tutorials/mnist/pros/index.html

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()
seed = 2016
features = 784
classes = 10
epoch = 1000

x = tf.placeholder(tf.float32, shape=[None, features])
y_ = tf.placeholder(tf.float32, shape=[None, classes])
#W = tf.Variable(tf.zeros([features, classes]))
W = tf.Variable(tf.truncated_normal([features, classes], stddev=0.1, seed=seed))
b = tf.Variable(tf.truncated_normal([classes], stddev=0.1, seed=seed))

sess.run(tf.initialize_all_variables())

#print sess.run(W)
y = tf.nn.softmax(tf.matmul(x,W) + b)
# lose function: cross-entropy
cross_entropy = tf.reduce_mean(tf.reduce_sum(-y_ * tf.log(y), 1)) # 1 = reduction_indices

# update function
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for i in xrange(epoch):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x:batch[0], y_:batch[1]}) # update W and b
    #print '%dth iteration, '%(i)
    #print 'Cross entropy over minibatch', cross_entropy.eval(feed_dict={x:batch[0], y_:batch[1]})
    #print 'Cross entropy over entire training data', cross_entropy.eval(feed_dict={x:mnist.train.images, y_:mnist.train.labels})

true_labels = tf.argmax(y_, 1)
predicted_labels = tf.argmax(y, 1)
#print true_labels.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels})
#print predicted_labels.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels})
correct_prediction = tf.equal(true_labels, predicted_labels)
#print correct_prediction.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels})
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print "Softmax regression"
print "train accuracy:", accuracy.eval(feed_dict={x:mnist.train.images, y_:mnist.train.labels})
print "test accuracy:", accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels})

# multilayer conv net
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, seed=seed)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# convolutional layer
def conv2d(x, W): # W filter
    # stride: [batch, height, width, channel]
    # W:filter_height, filter_width, channel, output_channel
    return tf.nn.conv2d(x, W, [1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, [1,2,2,1], [1,2,2,1], padding='SAME')

W_conv1 = weight_variable([5,5,1,32]) #32 filters with 5*5 receptive field
b_conv1 = bias_variable([32])

# input: 28 * 28 * 1
x_image = tf.reshape(x, [-1, 28, 28, 1])
# After conv with 32 filters: (28-5+2*2)/1 + 1 = 28
# 28 * 28 * 32
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# After pooling: 14 * 14 * 32
h_pool1 = max_pool_2x2(h_conv1)

# After conv2: 14*14*64
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# After pooling2: 7 * 7 * 64
h_pool2 = max_pool_2x2(h_conv2)

# Dense layer  : hidden 1024
W_fc1  = weight_variable([7*7*64, 1024])
b_fc1  = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

# Loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
# Training function
train_step = tf.train.AdagradOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))




