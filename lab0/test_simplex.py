import pytest
from lab0.my_simplex import Simplex
from sys import stderr
import numpy as np
from lab0.my_simplex import EPS

def test_simple():
    A = [[1, 1], [1, 0], [0, 1]]
    b = [2, 1, 1]
    c = [1, 1]
    s = Simplex(A, b, c)
    value, solve = s.solve()
    assert value == 2
    assert len(solve) == 2 and solve[0] == 1 and solve[1] == 1

def test_non_zero():
    A = [[1, 2], [2, 1], [-1, 0], [0, -1]]
    b = [6, 6, 0, 0]
    c = [1, 1]
    s = Simplex(A, b, c)
    value, solve = s.solve()
    assert value == 4
    assert abs(solve[0] + solve[1] - 4) < EPS

def test_ones():
    A = [[-1, 1], [1, -1]]
    b = [-1, -1]
    c = [1, 1]
    s = Simplex(A, b, c)
    value, solve = s.solve()
    assert value == np.inf

def test_big():
    A = [
        [2, 1, 1, 0],
        [1, 3, 0, 1],
        [1, 1, 1, 1]
    ]
    b = [8, 9, 6]
    c = [3, 2, 4, 1]
    s = Simplex(A, b, c)
    value, solve = s.solve()
    assert value != np.inf
    assert len(solve) == 4
    assert 2*solve[0] + solve[1] + solve[2] <= 8 + EPS
    assert solve[0] + 3*solve[1] + solve[3] <= 9 + EPS
    assert solve[0] + solve[1] + solve[2] + solve[3] <= 6 + EPS