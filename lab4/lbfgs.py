import numpy as np

class LBFGS:
    def __init__(self, m = 10):
        self.m = m
        self.s_history = []
        self.y_history = []
        
    def two_loop_recursion(self, grad):
        q = grad.copy()
        alpha = np.zeros(len(self.s_history))
        
        for i in range(len(self.s_history) - 1, -1, -1):
            s, y = self.s_history[i], self.y_history[i]
            rho = 1.0 / np.dot(y, s)
            alpha[i] = rho * np.dot(s, q)
            q = q - alpha[i] * y
        
        if len(self.s_history) > 0:
            s_last, y_last = self.s_history[-1], self.y_history[-1]
            gamma = np.dot(s_last, y_last) / np.dot(y_last, y_last)
            r = gamma * q
        else:
            r = q
        
        for i in range(len(self.s_history)):
            s, y = self.s_history[i], self.y_history[i]
            rho = 1.0 / np.dot(y, s)
            beta = rho * np.dot(y, r)
            r = r + s * (alpha[i] - beta)
        
        return -r
    
    def optimize(self, func, grad_func, x0, max_iter = 1000, tol = 1e-6):
        x = x0.copy()
        grad = grad_func(x)
        history = [func(x)]
        
        for k in range(max_iter):
            d = self.two_loop_recursion(grad)
            alpha = 1.0
            c1 = 1e-4
            c2 = 0.9

            for _ in range(20):
                x_new = x + alpha * d
                f_new = func(x_new)
                grad_new = grad_func(x_new)
                if f_new <= func(x) + c1 * alpha * np.dot(grad, d):
                    if np.dot(grad_new, d) >= c2 * np.dot(grad, d):
                        break
                alpha *= 0.5
            else:
                alpha = 1.0
            x_new = x + alpha * d
            grad_new = grad_func(x_new)
            s = x_new - x
            y = grad_new - grad
            self.s_history.append(s)
            self.y_history.append(y)
            if len(self.s_history) > self.m:
                self.s_history.pop(0)
                self.y_history.pop(0)
            x = x_new
            grad = grad_new
            history.append(func(x))
            if np.linalg.norm(grad) < tol:
                break
        
        return x, func(x), history
