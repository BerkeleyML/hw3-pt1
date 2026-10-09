from otter.test_files import test_case

OK_FORMAT = False

name = "q3a"
points = 3

@test_case(points=None, hidden=False)
def test_q3a_step_single(SGD, BearTensor):
    import numpy as np
    p = BearTensor(np.array([10.0]), 'p')
    p.adjoint = np.array([2.0])
    optimizer = SGD([p], lr=0.1)
    optimizer.step()
    assert np.allclose(p.value, 9.8)

@test_case(points=None, hidden=False)
def test_q3a_step_multiple(SGD, BearTensor):
    import numpy as np
    p = BearTensor(np.array([2.0, 3.0]), 'p')
    p.adjoint = np.array([3.0, 3.0])
    optimizer = SGD([p], lr=0.2)
    optimizer.step()
    assert np.allclose(p.value, np.array([1.4, 2.4]))

