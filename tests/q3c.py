from otter.test_files import test_case

OK_FORMAT = False

name = "q3c"
points = 5

@test_case(points=None, hidden=False)
def test_q3c_public(Adam, BearTensor):
    import numpy as np
    p = BearTensor(np.array([10.0]), 'p')
    optimizer = Adam([p], lr=0.1, beta1=0.9, beta2=0.999, eps=1e-08)
    p.adjoint = np.array([2.0])
    optimizer.step()
    assert np.allclose(p.value, 9.9)
    p.adjoint = np.array([-1.0])
    optimizer.step()
    assert np.allclose(p.value, 9.8733, atol=0.0001)

