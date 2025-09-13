import numpy as np

EPS = 1e-6
INF = np.inf

class Simplex:
    def __init__(self, A, b, c):
        self.m = len(A)
        self.n = len(A[0])
        
        self.setup_table(A, b, c)
        self.init_base()
    
    def setup_table(self, A, b, c):
        self.T = np.zeros((self.m + 1, self.n + self.m + 1))
        
        for i in range(self.m):
            for j in range(self.n):
                self.T[i, j] = A[i][j]
            self.T[i, self.n + i] = 1
            self.T[i, -1] = b[i]
        
        for j in range(self.n):
            self.T[self.m, j] = -c[j]
    
    def init_base(self):
        self.basis = [self.n + i for i in range(self.m)]
        self.non_basis = [j for j in range(self.n)]
    
    def upd_base(self, row, col):
        pivot_val = self.T[row, col]
        self.T[row] /= pivot_val
        for i in range(self.m + 1):
            if i == row:
                continue
            factor = self.T[i, col]
            if abs(factor) > EPS:
                self.T[i] -= factor * self.T[row]
        
        idx = self.non_basis.index(col)
        self.basis[row], self.non_basis[idx] = self.non_basis[idx], self.basis[row]
    
    def get_start(self):
        for col in self.non_basis:
            if self.T[self.m, col] < -EPS:
                return col
        return -1
    
    def get_end(self, enter_col):
        min_ratio = INF
        leave_row = -1
        
        for i in range(self.m):
            if self.T[i, enter_col] > EPS:
                ratio = self.T[i, -1] / self.T[i, enter_col]
                if ratio < min_ratio - EPS:
                    min_ratio = ratio
                    leave_row = i
        
        return leave_row
    
    def main_cycle(self):
        while True:
            enter_col = self.get_start()
            if enter_col < 0:
                break
            
            leave_row = self.get_end(enter_col)
            if leave_row < 0:
                return False
            
            self.upd_base(leave_row, enter_col)
        
        return True
    
    def get_solution(self):
        x = np.zeros(self.n)
        for i in range(self.m):
            if self.basis[i] < self.n:
                x[self.basis[i]] = self.T[i, -1]
        return x
    
    def solve(self):
        if not self.main_cycle():
            return INF, None
        
        opt = self.T[self.m, -1]
        sol = self.get_solution()
        
        return opt, sol