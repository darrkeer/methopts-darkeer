import pytest
import lbfgs
import numpy as np

def rosenbrock_nd(x):
    n = len(x)
    if n % 2 != 0:
        raise ValueError("len must be divisable by 2")
    
    f = 0.0
    for i in range(n // 2):
        idx1 = 2 * i
        idx2 = 2 * i + 1
        f += 100 * (x[idx2] - x[idx1]**2)**2 + (1 - x[idx1])**2
    
    return f

def rosenbrock_nd_grad(x):
    n = len(x)
    grad = np.zeros(n)
    
    for i in range(n // 2):
        idx1 = 2 * i
        idx2 = 2 * i + 1
        grad[idx1] = -400 * x[idx1] * (x[idx2] - x[idx1]**2) - 2 * (1 - x[idx1])
        grad[idx2] = 200 * (x[idx2] - x[idx1]**2)
    
    return grad

def almost_eq(a, b, prec):
    return all(map(lambda x: abs(x[0] - x[1]) <= 1e-6, zip(a, b)))

def test_rosenbrock_small():
    N, m = 10, 5
    x0 = np.zeros(N)
    optimizer = lbfgs.LBFGS(m=m)
    x_opt, f_opt, history = optimizer.optimize(rosenbrock_nd, rosenbrock_nd_grad, x0, 2000, 1e-6)
    
    assert almost_eq(x_opt, [1] * N, 1e-3)

def test_quadratic_function():
    N, m = 8, 4
    x0 = np.zeros(N)
    f = lambda x: np.sum((x - np.arange(1, len(x) + 1))**2)
    grad = lambda x: 2 * (x - np.arange(1, len(x) + 1))
    
    optimizer = lbfgs.LBFGS(m=m)
    x_opt, f_opt, history = optimizer.optimize(f, grad, x0, 100, 1e-8)
    
    expected = np.arange(1, N + 1)
    assert almost_eq(x_opt, expected, 1e-6)
    assert f_opt < 1e-10


def test_rosenbrock_large():
    N, m = 100, 10
    x0 = np.zeros(N)
    
    optimizer = lbfgs.LBFGS(m=m)
    x_opt, f_opt, history = optimizer.optimize(rosenbrock_nd, rosenbrock_nd_grad, x0, 2000, 1e-6)
    
    expected = np.ones(N)
    assert almost_eq(x_opt, expected, 1e-3)
    assert f_opt < 1e-6

def test_different_m_values():
    N = 20
    x0 = np.zeros(N)

    for m in [3, 5, 10, 15]:
        optimizer = lbfgs.LBFGS(m=m)
        x_opt, f_opt, history = optimizer.optimize(rosenbrock_nd, rosenbrock_nd_grad, x0, 1500, 1e-6)
        
        assert almost_eq(x_opt, np.ones(N), 1e-3)
        assert f_opt < 1e-5

def test_memory():
    N, m = 50, 5
    x0 = np.zeros(N)
    optimizer = lbfgs.LBFGS(m=m)
    x_opt, f_opt, history = optimizer.optimize(rosenbrock_nd, rosenbrock_nd_grad, x0, 2000, 1e-6)
    
    assert len(optimizer.s_history) <= m
    assert len(optimizer.y_history) <= m    
    assert almost_eq(x_opt, np.ones(N), 1e-3)
