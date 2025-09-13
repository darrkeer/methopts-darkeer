import pytest
import numpy as np
from sys import stderr
from optimizers import gradient_descent, momentum_descent, adam_descent

def almost(a, b):
    return abs(a - b) < 1e-2

def check_all_variants(f, grad, x0, f0, max_iters, excepted):
    optimizers = [gradient_descent, momentum_descent, adam_descent]
    for optimizer in optimizers:
        a, b = optimizer(f, grad, x0, lr=1e-2, max_iters=max_iters)
        assert almost(a[0], excepted[0]) and almost(a[1], excepted[1])

def test_sample():
    f = lambda xy: (xy[0] + 2)**2 + (xy[1] - 4)**2
    grad = lambda xy: np.array([2*(xy[0]+2), 2*(xy[1]-4)])
    check_all_variants(f, grad, [1,1], None, 1000, [-2,4])

def test_zeros():
    f = lambda xy: xy[0]**2 + xy[1]**2 + xy[0]*xy[1]
    grad = lambda xy: np.array([2*xy[0] + xy[1], xy[0] + 2*xy[1]])
    check_all_variants(f, grad, [1,1], None, 1000, [0,0])

def test_multiple_vars():
    f = lambda x: (x[0] - 2)**2 + (x[1] + 3)**2 + (x[2] - 1)**2
    grad = lambda x: np.array([2*(x[0]-2), 2*(x[1]+3), 2*(x[2]-1)])
    check_all_variants(f, grad, [0, 0, 0], None, 1000, [2, -3, 1])

def test_sample2():
    f = lambda x: x[0]**2 + 2*x[0]*x[1] + 3*x[1]**2 + 4*x[0] + 5*x[1] + 6
    grad = lambda x: np.array([2*x[0] + 2*x[1] + 4, 2*x[0] + 6*x[1] + 5])
    check_all_variants(f, grad, [0, 0], None, 1500, [-1.75, -0.25])