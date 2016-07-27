__author__ = "Music"
# Learning Basic Usage of Tf
# https://www.tensorflow.org/versions/r0.9/get_started/basic_usage.html
import tensorflow as tf

print "Construct Graph"
matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],[2]])
# TO MY UNDERSTANDING: constant is added to the default graph as op (node), matrix1 represents the output of the op.

product = tf.matmul(matrix1, matrix2)

print "Launch the default graph"
sess = tf.Session()
result = sess.run(product)
print result
sess.close()

# Interactive mode  Use op.run() and tensor.eval()
sess= tf.InteractiveSession()
x = tf.Variable([1,2])
a = tf.constant([3,3])
x.initializer.run()

sub = tf.sub(x, a)
print sub.eval()
sess.close()

state = tf.Variable(0, name='counter')
one = tf.constant(1)
update = tf.assign(state, tf.add(one, state))
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    print sess.run(state)
    for _ in xrange(3):
        sess.run(update)
        print sess.run([state, update, state])


# input1 = tf.constant([3.0])
# input2 = tf.constant([2.0])
# input3 = tf.constant([5.0])
# intermed = tf.add(input2, input3)
# mul = tf.mul(input1, intermed)
#
# with tf.Session() as sess:
#   result = sess.run([mul, intermed])
#   print(result)


# feed
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.mul(input1, input2)

with tf.Session() as sess:
  print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))

# output:
# [array([ 14.], dtype=float32)]
