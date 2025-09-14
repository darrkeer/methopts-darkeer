import pytest
from bandc import *

def test_sample():
    g = Graph(4)
    g.add_edge(0,1,2)
    g.add_edge(1,2,2)
    g.add_edge(2,3,3)
    g.add_edge(3,0,2)
    g.add_edge(0,2,3)
    g.add_edge(1,3,3)
    solver = BCState(g)
    res, path = solver.solve()

    assert len(path) == len(set(path)) == 4

    assert almost_equal(res, 9.0)

def almost_equal(a, b, tol=1e-6):
    return abs(a - b) <= tol

def test_triangle():
    G = Graph(3)
    G.add_edge(0, 1, 6)
    G.add_edge(1, 2, 8)
    G.add_edge(0, 2, 10)

    solver = BCState(G)
    res, path = solver.solve()

    assert almost_equal(res, 6 + 8 + 10)
    assert len(path) == 3
    assert len(set(path)) == 3

def test_check_tour_order():
    g = Graph(4)
    g.cost = [
        [0, 2, 3, 2],
        [2, 0, 3, 4],
        [3, 3, 0, 3],
        [2, 4, 3, 0]
    ]

    solver = BCState(g)
    res, path = solver.solve()

    assert almost_equal(res, 10.0)
    assert len(path) == 4

    seen = [False] * 4
    for v in path:
        assert 0 <= v < 4
        seen[v] = True

    assert all(seen)

    for i in range(len(path)):
        u = path[i]
        v = path[(i + 1) % len(path)]
        assert g.cost[u][v] != 0

def test_tour_must_be_returned_with_length():
    g = Graph(4)
    g.cost = [
        [0, 3, 4, 3],
        [3, 0, 4, 5],
        [4, 4, 0, 4],
        [3, 5, 4, 0]
    ]

    solver = BCState(g)
    res, path = solver.solve()

    assert almost_equal(res, 14.0)
    assert len(path) > 0

    total = 0.0
    for i in range(len(path)):
        u = path[i]
        v = path[(i + 1) % len(path)]
        total += g.cost[u][v]

    assert almost_equal(total, res)

def test_tour_length_must_match_sum():
    g = Graph(4)
    g.cost = [
        [0, 4, 5, 4],
        [4, 0, 5, 6],
        [5, 5, 0, 5],
        [4, 6, 5, 0]
    ]

    solver = BCState(g)
    res, path = solver.solve()

    assert almost_equal(res, 18.0)

    total = 0.0
    for i in range(len(path)):
        u = path[i]
        v = path[(i + 1) % len(path)]
        total += g.cost[u][v]

    assert almost_equal(total, res)

def test_tour_size_is_n():
    g = Graph(5)
    g.cost = [
        [0, 3, 4, 5, 6],
        [3, 0, 7, 8, 9],
        [4, 7, 0, 10, 11],
        [5, 8, 10, 0, 12],
        [6, 9, 11, 12, 0]
    ]

    solver = BCState(g)
    res, path = solver.solve()

    assert len(path) == g.N

def test_tour_is_permutation():
    g = Graph(4)
    g.cost = [
        [0, 5, 6, 7],
        [5, 0, 8, 9],
        [6, 8, 0, 10],
        [7, 9, 10, 0]
    ]

    solver = BCState(g)
    res, path = solver.solve()

    seen = [False] * g.N
    for v in path:
        assert 0 <= v < g.N
        assert not seen[v]
        seen[v] = True

def test_tour_length_matches_cost():
    g = Graph(4)
    g.cost = [
        [0, 6, 7, 8],
        [6, 0, 9, 10],
        [7, 9, 0, 11],
        [8, 10, 11, 0]
    ]

    solver = BCState(g)
    res, path = solver.solve()

    total = 0.0
    for i in range(len(path)):
        u = path[i]
        v = path[(i + 1) % len(path)]
        total += g.cost[u][v]

    assert almost_equal(total, res)

def test_tour_must_be_cycle():
    g = Graph(3)
    g.cost = [
        [0, 4, 5],
        [4, 0, 3],
        [5, 3, 0]
    ]

    solver = BCState(g)
    res, path = solver.solve()

    assert len(path) > 0

    start = path[0]
    end = path[-1]
    last_edge = g.cost[end][start]

    assert last_edge < 1e9

def test_recovers_hamiltonian_cycle_correctly():
    N = 6
    G = Graph(N)
    G.cost = [[100.0] * N for _ in range(N)]
    for i in range(N):
        j = (i + 1) % N
        G.cost[i][j] = 2.0
        G.cost[j][i] = 2.0

    solver = BCState(G)
    res, path = solver.solve()

    assert almost_equal(res, N * 2.0)
    assert len(path) == N
    
    seen = [False] * N
    for v in path:
        assert 0 <= v < N
        assert not seen[v]
        seen[v] = True

def test_tour_is_built_when_integer_solution_found():
    N = 5
    G = Graph(N)
    G.cost = [[20.0] * N for _ in range(N)]
    
    for i in range(N):
        j = (i + 1) % N
        G.cost[i][j] = 3.0
        G.cost[j][i] = 3.0

    solver = BCState(G)
    res, path = solver.solve()

    assert almost_equal(res, N * 3.0)
    assert len(path) == N

    visited = [False] * N
    actual_length = 0.0

    for i in range(N):
        u = path[i]
        v = path[(i + 1) % N]
        assert not visited[u]
        visited[u] = True
        actual_length += G.cost[u][v]

    assert almost_equal(actual_length, N * 3.0)

def test_tour_is_built_correctly():
    N = 9
    G = Graph(N)
    G.cost = [[200.0] * N for _ in range(N)]
    
    for i in range(N):
        j = (i + 1) % N
        G.cost[i][j] = 4.0
        G.cost[j][i] = 4.0

    solver = BCState(G)
    res, path = solver.solve()

    assert almost_equal(res, N * 4.0)
    assert len(path) == N

    visited = [False] * N
    for i in range(N):
        v = path[i]
        assert not visited[v]
        visited[v] = True

    assert all(visited)

    actual_length = 0.0
    for i in range(N):
        u = path[i]
        v = path[(i + 1) % N]
        actual_length += G.cost[u][v]

    assert almost_equal(actual_length, N * 4.0)

def test_small_graph_with_known_optimal():
    g = Graph(3)
    g.cost = [
        [0, 10, 15],
        [10, 0, 20],
        [15, 20, 0]
    ]

    solver = BCState(g)
    res, path = solver.solve()


    assert almost_equal(res, 45.0)
    assert len(path) == 3
    assert len(set(path)) == 3

def test_asymmetric_costs():
    g = Graph(3)
    g.cost = [
        [0, 5, 8],
        [6, 0, 7],
        [9, 4, 0]
    ]

    solver = BCState(g)
    res, path = solver.solve()

    assert len(path) == 3
    assert len(set(path)) == 3
