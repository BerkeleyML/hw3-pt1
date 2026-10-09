from otter.test_files import test_case

OK_FORMAT = False

name = "q3b"
points = 3

@test_case(points=None, hidden=False)
def test_q3b_public(Momentum, BearTensor):
    import numpy as np
    p = BearTensor(np.array([10.0]), 'p')
    optimizer = Momentum([p], lr=0.1, beta=0.9)
    p.adjoint = np.array([2.0])
    optimizer.step()
    assert np.allclose(optimizer.velocities[0], 2.0)
    assert np.allclose(p.value, 9.8)
    p.adjoint = np.array([4.0])
    optimizer.step()
    assert np.allclose(optimizer.velocities[0], 5.8)
    assert np.allclose(p.value, 9.22)

