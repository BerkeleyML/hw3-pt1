from otter.test_files import test_case

OK_FORMAT = False

name = "q2a"
points = 10

@test_case(points=None, hidden=False)
def test_q2a_public(topological_sort, BearTensor):
    import numpy as np
    a = BearTensor(np.array([1, 2, 3]), 'a')
    b = BearTensor(np.array([4, 5, 6]), 'b')
    c = BearTensor(np.array([2, 3, 3]), 'c')
    d = a - b
    e = d + c
    nodes = topological_sort(e)
    assert len(nodes) == 5
    assert nodes[-1] == e
    assert nodes.index(a) < nodes.index(d)
    assert nodes.index(b) < nodes.index(d)
    assert nodes.index(d) < nodes.index(e)
    assert nodes.index(c) < nodes.index(e)

