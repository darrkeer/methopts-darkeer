import numpy as np

def newton_method(start, f, grad, hess, tol=1e-6, max_iter=100):
    x = np.array(start, dtype=float)
    
    for i in range(max_iter):
        g = grad(x)
        H = hess(x)

        delta = np.linalg.solve(H, g)
        x_new = x - delta
        if np.linalg.norm(delta) < tol:
            return x_new
        x = x_new
    
    return x


f = lambda x: (x[0] - 1)**2+ (x[1] + 2)**2
grad = lambda x: np.array([2 * (x[0] - 1), 2 * (x[1] + 2)])
hess = lambda xy: np.array([[2, 0], [0, 2]])

start_point = [0, 0]
solution = newton_method(start_point, f, grad, hess)
print("min point:", solution)
print("min value:", f(solution))
