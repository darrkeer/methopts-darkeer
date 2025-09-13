import numpy as np

def f_quad_2d(x):
    return (x[0] - 3)**2 + (x[1] + 1)**2

def grad_quad_2d(x):
    return np.array([2 * (x[0] - 3), 2 * (x[1] + 1)], dtype=float)


def gradient_descent(f, grad, x0, lr=1e-2, max_iters=1000, tol=1e-6):
    x = np.array(x0, dtype=float)
    history = []
    for k in range(max_iters):
        g = grad(x)
        x -= lr * g
        history.append(f(x))
        if np.linalg.norm(g) < tol:
            break
    return x, history


def momentum_descent(f, grad, x0, lr=1e-2, beta=0.9, max_iters=1000, tol=1e-6):
    x = np.array(x0, dtype=float)
    v = np.zeros_like(x)
    history = []
    for k in range(max_iters):
        g = grad(x)
        v = beta * v + (1 - beta) * g
        x -= lr * v
        history.append(f(x))
        if np.linalg.norm(g) < tol:
            break
    return x, history


def adam_descent(f, grad, x0, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, max_iters=1000, tol=1e-6):
    x = np.array(x0, dtype=float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    history = []
    for t in range(1, max_iters+1):
        g = grad(x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g * g)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x -= lr * m_hat / (np.sqrt(v_hat) + eps)
        history.append(f(x))
        if np.linalg.norm(g) < tol:
            break
    return x, history
