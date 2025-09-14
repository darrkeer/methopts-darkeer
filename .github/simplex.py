from scipy.optimize import linprog
import numpy as np

class Simplex:
    def __init__(self, A, b, c, bounds=None):
        self.A = A
        self.b = b
        self.c = c
        self.bounds = bounds

    def solve(self):
        n = len(self.c)
        if self.bounds is None:
            bounds = [(0, 1)] * n
        else:
            bounds = self.bounds

        if not self.A:
            A_ub = None
            b_ub = None
        else:
            A_ub = self.A
            b_ub = self.b

        try:
            res = linprog(
                c=self.c,
                A_ub=A_ub,
                b_ub=b_ub,
                bounds=bounds,
                method="highs"
            )

            if res.success:
                return res.fun, res.x.tolist()
            else:
                return np.inf, None
        except:
            return np.inf, None
