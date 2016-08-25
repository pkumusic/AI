import numpy as np
import theano
import theano.tensor as T
from theano import shared
rng = np.random
rng.seed(2016)

N = 20
features = 5
# Random generate dataset. (input matrix, ground_truth classes)
D = (rng.randn(N, features), rng.randint(size=N, low=0, high=2))
training_steps = 10

x = T.dmatrix('x')
y = T.lvector('y')

w = shared(rng.randn(features), name='w')
b = shared(0.0, name='b')
print "initial model"
print w.get_value()
print b.get_value()
# Construct Theano expression graph
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))   # Probability that target = 1
prediction = p_1 > 0.5                    # The prediction thresholded
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
cost = xent.mean() + 0.01 * (w ** 2).sum()# The cost to minimize
gw, gb = T.grad(cost, [w, b])             # Compute the gradient of the cost
                                          # w.r.t weight vector w and
                                          # bias term b
                                          # (we shall return to this in a
                                          # following section of this tutorial)
precision = T.sum(T.eq(prediction, y)) / N

# Compile
train = theano.function(
          inputs=[x,y],
          outputs=[prediction, xent],
          updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
predict = theano.function(inputs=[x], outputs=prediction)
cal_precision = theano.function(inputs=[x,y], outputs=precision)


# Train
for i in range(training_steps):
    pred, err = train(D[0], D[1])
    precision = cal_precision(D[0], D[1])
    print pred, err, precision

print("Final model:")
print(w.get_value())
print(b.get_value())
print("target values for D:")
print(D[1])
print("prediction on D:")
print(predict(D[0]))