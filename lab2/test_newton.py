import pytest
import numpy as np
from newton import newton_method

def almost_equal(a, b, eps=1e-6):
    return abs(a - b) <= eps

def test_sq_f():
    f = lambda x: 3*(x[0] - 2)**2 + 5*(x[1] + 1)**2
    grad = lambda x: np.array([6*(x[0] - 2), 10*(x[1] + 1)])
    hess = lambda x: np.array([[6, 0], [0, 10]])
    start_point = [0, 0]
    solution = newton_method(start_point, f, grad, hess)
    assert almost_equal(solution[0], 2) and almost_equal(solution[1], -1)

def test_sample():
    f = lambda x: x[0]**2 + x[0]*x[1] + x[1]**2
    grad = lambda x: np.array([2*x[0] + x[1], x[0] + 2*x[1]])
    hess = lambda x: np.array([[2, 1], [1, 2]])
    start_point = [3, 4]
    solution = newton_method(start_point, f, grad, hess)
    assert almost_equal(solution[0], 0) and almost_equal(solution[1], 0)

def test_single_var():
    f = lambda x: (x[0] - 5)**4
    grad = lambda x: np.array([4*(x[0] - 5)**3])
    hess = lambda x: np.array([[12*(x[0] - 5)**2]])
    start_point = [10]
    solution = newton_method(start_point, f, grad, hess)
    assert almost_equal(solution[0], 5, eps=1e-3)

def test_rosenbrock():
    f = lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
    grad = lambda x: np.array([
        -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
        200*(x[1] - x[0]**2)
    ])
    hess = lambda x: np.array([
        [2 - 400*x[1] + 1200*x[0]**2, -400*x[0]],
        [-400*x[0], 200]
    ])
    start_point = [-1, 2]
    solution = newton_method(start_point, f, grad, hess)
    assert almost_equal(solution[0], 1) and almost_equal(solution[1], 1)