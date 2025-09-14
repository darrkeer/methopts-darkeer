from my_simplex import Simplex
import math
import numpy as np
from collections import deque
from scipy.optimize import linprog
from enum import Enum

class Constraint(Enum):
    Less = 0
    Eq = 1
    Greater = 2

class Graph:
    def __init__(self, N):
        self.N = N
        self.cost = [[np.inf] * N for _ in range(N)]
        for i in range(N):
            self.cost[i][i] = 0

    def add_edge(self, i, j, w):
        self.cost[i][j] = self.cost[j][i] = w

class Node:
    def __init__(self):
        self.fixed = []
        self.forbidden = []
        self.constraint_indices = []

def node_id(i, j, N):
    if i > j:
        i, j = j, i
    return (i * (2 * N - i - 1)) // 2 + (j - i - 1)

class BCState:
    def __init__(self, G, max_nodes=1500):
        self.G = G
        self.max_nodes = max_nodes
        self.best_res = np.inf
        self.best_path = []

        self.num_vars = self.G.N * (self.G.N - 1) // 2
        self.A = []
        self.b = []
        self.constraints = []
        self.c = [0] * self.num_vars
        
        idx = 0
        for i in range(self.G.N):
            for j in range(i + 1, self.G.N):
                self.c[idx] = self.G.cost[i][j]
                idx += 1

    def add_constraint(self, row, rhs, sense):
        self.A.append(row)
        self.b.append(rhs)
        self.constraints.append(sense)
        return len(self.A) - 1

    def get_constraints_for_node(self, node):
        A = []
        b = []
        constraints = []
        
        for i in range(len(self.A)):
            A.append(self.A[i][:])
            b.append(self.b[i])
            constraints.append(self.constraints[i])
        
        for i, j in node.fixed:
            row = [0] * self.num_vars
            k = node_id(i, j, self.G.N)
            row[k] = 1
            A.append(row)
            b.append(1)
            constraints.append(Constraint.Eq)
        
        for i, j in node.forbidden:
            row = [0] * self.num_vars
            k = node_id(i, j, self.G.N)
            row[k] = 1
            A.append(row)
            b.append(0)
            constraints.append(Constraint.Eq)
        
        return A, b, constraints

    def solve_lp(self, node):
        A, b, constraints = self.get_constraints_for_node(node)
        
        A_ub, b_ub = [], []
        for row, bi, sense in zip(A, b, constraints):
            if sense == Constraint.Eq:
                A_ub.append(row)
                b_ub.append(bi)
                A_ub.append([-v for v in row])
                b_ub.append(-bi)
            elif sense == Constraint.Less:
                A_ub.append(row)
                b_ub.append(bi)
            elif sense == Constraint.Greater:
                A_ub.append([-v for v in row])
                b_ub.append(-bi)

        solver = Simplex(A_ub, b_ub, self.c, bounds=[(0, 1)] * self.num_vars)
        return solver.solve()

    def carter_separation(self, x):
        N = self.G.N
        violations = []
        
        adj = [[] for _ in range(N)]
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                if x[idx] > 1e-6:
                    adj[i].append(j)
                    adj[j].append(i)
                idx += 1
        
        visited = [False] * N
        for i in range(N):
            if not visited[i]:
                component = []
                stack = [i]
                visited[i] = True
                while stack:
                    u = stack.pop()
                    component.append(u)
                    for v in adj[u]:
                        if not visited[v]:
                            visited[v] = True
                            stack.append(v)
                
                if 2 <= len(component) < N:
                    sum_edges = 0
                    idx = 0
                    for u in range(N):
                        for v in range(u + 1, N):
                            if u in component and v in component:
                                sum_edges += x[idx]
                            idx += 1
                    
                    if sum_edges > len(component) - 1 + 1e-6:
                        violations.append(component)
        
        return violations

    def solve_node(self, node):
        if self.max_nodes <= 0:
            return
        self.max_nodes -= 1

        lp, x = self.solve_lp(node)
        
        if x is None or lp >= self.best_res - 1e-9:
            return

        integral = all(abs(xi - round(xi)) <= 1e-6 for xi in x)
        
        if integral:
            if self.is_valid_tour(x):
                length, path = self.extract_tour(x)
                if length < self.best_res:
                    self.best_res = length
                    self.best_path = path
            return
        
        violations = self.carter_separation(x)
        if violations:
            new_constraint_indices = []
            for component in violations:
                row = [0] * self.num_vars
                for i in component:
                    for j in component:
                        if i < j:
                            k = node_id(i, j, self.G.N)
                            row[k] = 1
                constraint_idx = self.add_constraint(row, len(component) - 1, Constraint.Less)
                new_constraint_indices.append(constraint_idx)
            
            self.solve_node(node)
            for _ in range(len(new_constraint_indices)):
                self.A.pop()
                self.b.pop()
                self.constraints.pop()
            
            return
        
        branch_edge = None
        min_dist = 1
        
        idx = 0
        for i in range(self.G.N):
            for j in range(i + 1, self.G.N):
                if 1e-6 < x[idx] < 1 - 1e-6:
                    dist = abs(x[idx] - 0.5)
                    if dist < min_dist:
                        min_dist = dist
                        branch_edge = (i, j, idx)
                idx += 1
        
        if branch_edge:
            i, j, edge_idx = branch_edge
            
            left_node = Node()
            left_node.fixed = node.fixed[:]
            left_node.forbidden = node.forbidden + [(i, j)]
            self.solve_node(left_node)
            
            right_node = Node()
            right_node.fixed = node.fixed + [(i, j)]
            right_node.forbidden = node.forbidden[:]
            self.solve_node(right_node)

    def is_valid_tour(self, x):
        N = self.G.N
        degrees = [0] * N
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                if abs(x[idx] - 1) < 1e-6:
                    degrees[i] += 1
                    degrees[j] += 1
                idx += 1
        
        if not all(deg == 2 for deg in degrees):
            return False
        
        adj = [[] for _ in range(N)]
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                if abs(x[idx] - 1) < 1e-6:
                    adj[i].append(j)
                    adj[j].append(i)
                idx += 1
        
        visited = [False] * N
        stack = [0]
        visited[0] = True
        count = 1
        
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    count += 1
                    stack.append(v)
        
        return count == N

    def extract_tour(self, x):
        N = self.G.N
        adj = [[] for _ in range(N)]
        length = 0
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                if abs(x[idx] - 1) < 1e-6:
                    length += self.G.cost[i][j]
                    adj[i].append(j)
                    adj[j].append(i)
                idx += 1
        
        path = [0]
        visited = [False] * N
        visited[0] = True
        current = 0
        prev = -1
        
        for _ in range(N - 1):
            for neighbor in adj[current]:
                if neighbor != prev and not visited[neighbor]:
                    path.append(neighbor)
                    visited[neighbor] = True
                    prev, current = current, neighbor
                    break
        
        return length, path

    def solve(self):
        N = self.G.N
        for i in range(N):
            row = [0] * self.num_vars
            for j in range(N):
                if i != j:
                    k = node_id(min(i, j), max(i, j), N)
                    row[k] = 1
            self.add_constraint(row, 2, Constraint.Eq)
        
        root_node = Node()
        self.solve_node(root_node)
        
        if self.best_res == np.inf:
            return np.inf, []
        return self.best_res, self.best_path