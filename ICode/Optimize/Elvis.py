#This is an Elvis code that help me for a first step in gradient controlling
import numpy as np
from numpy.random import rand, randn
g = lambda x : x
f = lambda x : .5 * x ** 2
check_grad(f, g, np.array([89]))
from scipy.optimize import fmin_l_bfgs_b, check_grad
check_grad(f, g, np.array([89]))
map(lambda x : check_grad(f, g, x), rand(100))
map(lambda x : check_grad(f, g, x), rand(100, 1))
map(lambda x : check_grad(lambda x: .5 * x * x, lambda x: x, x), rand(1000, 1))
map(lambda x : check_grad(lambda x: .5 * x * x, lambda x: x, x), rand(1000, 1))
fmin_l_bfgs_b??
fg = lambda x, **kwargs: (f(x), g(x))
fmin_l_bfgs_b(fg, randn(1))
fmin_l_bfgs_b(fg, randn(1), iprint=3)
fmin_l_bfgs_b(fg,  array([1e8]), iprint=3)