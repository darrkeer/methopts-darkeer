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

class Graph:
    def __init__(self, n):
        self.n = n
        self.cost = [[0] * n for _ in range(n)]
    
    def set_cost(self, i, j, value):
        self.cost[i][j] = value
        self.cost[j][i] = value

class TSPSolution:
    def __init__(self, length, tour):
        self.length = length
        self.tour = tour

class BranchAndCutSolver:
    def __init__(self, graph, max_iter=1000):
        self.graph = graph
        self.n = graph.n
        self.max_iter = max_iter
        self.global_constraints = []  # Глобальный пул ограничений
        self.node_constraint_indices = []  # Индексы ограничений в узлах дерева
    
    def solve(self):
        best_solution = None
        best_length = INF
        
        stack = [self.create_initial_node()]
        
        for iteration in range(self.max_iter):
            if not stack:
                break
                
            node = stack.pop()
            lp_solution = self.solve_lp_relaxation(node)
            
            if lp_solution is None or lp_solution[0] >= best_length:
                continue
                
            if self.is_integer_solution(lp_solution[1]):
                tour = self.reconstruct_tour(lp_solution[1])
                length = self.calculate_tour_length(tour)
                if length < best_length:
                    best_length = length
                    best_solution = TSPSolution(length, tour)
                continue
            
            branch_edge = self.select_branching_edge(lp_solution[1])  # Ветвление по ребрам близким к 0.5
            if branch_edge is None:
                continue
                
            left_node, right_node = self.create_branch_nodes(node, branch_edge)
            stack.append(left_node)
            stack.append(right_node)
            
            violated_constraints = self.find_violated_constraints(lp_solution[1])  # Решение задачи сепарации
            for constraint in violated_constraints:
                self.global_constraints.append(constraint)
                self.node_constraint_indices.append(len(self.global_constraints) - 1)
        
        return best_solution if best_solution else TSPSolution(INF, [])
    
    def create_initial_node(self):
        return {'constraints': [], 'fixed_edges': []}
    
    def solve_lp_relaxation(self, node):
        A, b, c = self.build_lp_problem(node)
        simplex = Simplex(A, b, c)  # Использование симплекс-метода из первой задачи
        return simplex.solve()
    
    def build_lp_problem(self, node):
        num_vars = self.n * self.n
        num_constraints = len(self.global_constraints) + len(node['constraints'])
        
        A = []
        b = []
        
        for constr_idx in node['constraints']:
            constraint = self.global_constraints[constr_idx]
            A.append(constraint['coeffs'])
            b.append(constraint['rhs'])
        
        for fixed_edge in node['fixed_edges']:
            i, j, value = fixed_edge
            coeffs = [0] * num_vars
            coeffs[i * self.n + j] = 1
            A.append(coeffs)
            b.append(value)
        
        c = [self.graph.cost[i][j] for i in range(self.n) for j in range(self.n)]
        
        return A, b, c
    
    def is_integer_solution(self, x):
        return all(abs(val - round(val)) < EPS for val in x)
    
    def reconstruct_tour(self, x):
        return list(range(self.n))
    
    def calculate_tour_length(self, tour):
        length = 0
        for i in range(len(tour)):
            j = (i + 1) % len(tour)
            length += self.graph.cost[tour[i]][tour[j]]
        return length
    
    def select_branching_edge(self, x):
        closest_to_half = None
        min_diff = INF
        
        for i in range(self.n):
            for j in range(i + 1, self.n):
                idx = i * self.n + j
                if 0 < x[idx] < 1:
                    diff = abs(x[idx] - 0.5)
                    if diff < min_diff:
                        min_diff = diff
                        closest_to_half = (i, j)
        
        return closest_to_half
    
    def create_branch_nodes(self, parent_node, edge):
        i, j = edge
        left_node = {
            'constraints': parent_node['constraints'][:],
            'fixed_edges': parent_node['fixed_edges'] + [(i, j, 0)]
        }
        right_node = {
            'constraints': parent_node['constraints'][:],
            'fixed_edges': parent_node['fixed_edges'] + [(i, j, 1)]
        }
        return left_node, right_node
    
    def find_violated_constraints(self, x):
        violated = []
        x_matrix = np.array(x).reshape((self.n, self.n))
        
        for S in self.find_subtours(x_matrix):
            if len(S) < self.n:
                coeffs = [0] * (self.n * self.n)
                for i in S:
                    for j in range(self.n):
                        if j not in S:
                            coeffs[i * self.n + j] = 1
                violated.append({'coeffs': coeffs, 'rhs': 1})
        
        return violated
    
    def find_subtours(self, x_matrix):
        visited = [False] * self.n
        subtours = []
        
        for i in range(self.n):
            if not visited[i]:
                tour = []
                current = i
                while not visited[current]:
                    visited[current] = True
                    tour.append(current)
                    next_node = np.argmax(x_matrix[current])
                    current = next_node
                subtours.append(tour)
        
        return subtours