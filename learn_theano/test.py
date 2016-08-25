import numpy as np
from theano import function, pp, In, shared
from theano.tensor.shared_randomstreams import RandomStreams
import theano.tensor as T

# x = T.dscalar('x')
# y = T.dscalar('y')
# z = x + y
# f = function([x,y], z)
# print(pp(z))
# print np.allclose(f(2,3), 5)
#
# # exercise
# a = T.ivector() # declare variable
# out = a + a ** 10               # build symbolic expression
# f = function([a], out)   # compile function
# print(f([0, 1, 2]))
#
# x = T.dmatrix('x')
# y = 1/(1+T.exp(-x))
# logistic = function([x],y)
# print logistic([[1,2],[3,4]])
# a,b = T.dmatrices('a','b')
# diff = a-b
# abs_diff = abs(diff)
# diff_square = diff ** 2
# f = function([a, In(b, value=[[1]])],[diff, abs_diff, diff_square])
# print f([[1]])
# state = shared(0)
# inc = T.iscalar('inc')
# accumulator = function([inc], state, updates={state:state+inc})
# print accumulator(1)
# print "internal State:", state.get_value()
# print accumulator(2)
# print "internal State:", state.get_value()

# random number generator
srng = RandomStreams(seed=234)
rv_u = srng.uniform((2,2))
rv_n = srng.normal((2,2))
f = function([], rv_u)
g = function([], rv_n, no_default_updates=True)
nearly_zeros = function([], rv_u + rv_u - 2 * rv_u)
#print g()
print f()
#print g()
#print nearly_zeros()
print rv_u.rng.get_value().get_state()