from bandc import *

g = Graph(4)
g.add_edge(0,1,1)
g.add_edge(1,2,1)
g.add_edge(2,3,1)
g.add_edge(3,0,1)
g.add_edge(0,2,2)
g.add_edge(1,3,2)

solver = BCState(g)
sol = solver.solve()

assert len(sol.tour) == len(set(sol.tour)) == 4

