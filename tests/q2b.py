from otter.test_files import test_case

OK_FORMAT = False

name = "q2b"
points = 10

@test_case(points=None, hidden=False)
def test_q2b_sub_grad(BearTensor):
    import numpy as np
    a = BearTensor(np.array([1, 2, 3]), 'a')
    b = BearTensor(np.array([4, 5, 6]), 'b')
    c = a - b
    s = c.sum()
    s.backward()
    expected_a_grad = np.ones_like(c.value)
    expected_b_grad = -np.ones_like(c.value)
    assert np.allclose(a.adjoint, expected_a_grad), f'Gradient for a failed: {a.adjoint} != {expected_a_grad}'
    assert np.allclose(b.adjoint, expected_b_grad), f'Gradient for b failed: {b.adjoint} != {expected_b_grad}'

@test_case(points=None, hidden=False)
def test_q2b_add_grad(BearTensor):
    import numpy as np
    a = BearTensor(np.array([1, 2, 3]), 'a')
    b = BearTensor(np.array([4, 5, 6]), 'b')
    c = a + b
    s = c.sum()
    s.backward()
    assert np.allclose(a.adjoint, np.ones_like(c.value))
    assert np.allclose(b.adjoint, np.ones_like(c.value))

@test_case(points=None, hidden=False)
def test_q2b_mul_grad(BearTensor):
    import numpy as np
    a = BearTensor(np.array([1.0, 2.0, 3.0]), 'a')
    b = BearTensor(np.array([4.0, 5.0, 6.0]), 'b')
    c = a * b
    s = c.sum()
    s.backward()
    assert np.allclose(a.adjoint, b.value)
    assert np.allclose(b.adjoint, a.value)

@test_case(points=None, hidden=False)
def test_q2b_pow_grad(BearTensor):
    import numpy as np
    a = BearTensor(np.array([-1, 2, 3]), 'a')
    c = a ** 3
    s = c.sum()
    s.backward()
    expected_value = np.array([3, 12, 27])
    assert np.allclose(a.adjoint, expected_value)

@test_case(points=None, hidden=False)
def test_q2b_matmul_grad(BearTensor):
    import numpy as np
    a = BearTensor(np.array([[-1, 2], [2, 3]]), 'a')
    b = BearTensor(np.array([[4, 2], [0, -3]]), 'a')
    c = a @ b
    s = c.sum()
    s.backward()
    a_value = np.array([[6, -3], [6, -3]])
    b_value = np.array([[1, 1], [5, 5]])
    assert np.allclose(a.adjoint, a_value)
    assert np.allclose(b.adjoint, b_value)

@test_case(points=None, hidden=False)
def test_q2b_matvec_grad(BearTensor):
    import numpy as np
    A = BearTensor(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), 'A')
    v = BearTensor(np.array([1.0, -1.0]), 'v')
    (A @ v).sum().backward()
    assert np.allclose(A.adjoint, [[1, -1], [1, -1], [1, -1]]), f'Gradient for A in A @ v failed: {A.adjoint}'
    assert np.allclose(v.adjoint, [9, 12]), f'Gradient for v in A @ v failed: {v.adjoint}'
    u = BearTensor(np.array([1.0, 0.0, -1.0]), 'u')
    B = BearTensor(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), 'B')
    (u @ B).sum().backward()
    assert np.allclose(u.adjoint, [3, 7, 11]), f'Gradient for u in u @ B failed: {u.adjoint}'
    assert np.allclose(B.adjoint, [[1, 1], [0, 0], [-1, -1]]), f'Gradient for B in u @ B failed: {B.adjoint}'

@test_case(points=None, hidden=False)
def test_q2b_dot_grad(BearTensor):
    import numpy as np
    a = BearTensor(np.array([-1, 2, 3]), 'a')
    b = BearTensor(np.array([4, 2, 2]), 'a')
    c = a.dot(b)
    s = c.sum()
    s.backward()
    assert np.allclose(a.adjoint, b.value)
    assert np.allclose(b.adjoint, a.value)

@test_case(points=None, hidden=False)
def test_q2b_sum_grad(BearTensor):
    import numpy as np
    a = BearTensor(np.array([-1, 2, 3]), 'a')
    c = a.sum()
    c.backward()
    assert np.allclose(a.adjoint, np.array([1, 1, 1]))

@test_case(points=None, hidden=False)
def test_q2b_mean_grad(BearTensor):
    import numpy as np
    a = BearTensor(np.array([-1, 2, 3]), 'a')
    c = a.mean()
    c.backward()
    assert np.allclose(a.adjoint, np.array([1 / 3, 1 / 3, 1 / 3]))

@test_case(points=None, hidden=False)
def test_q2b_relu_grad(BearTensor):
    import numpy as np
    a = BearTensor(np.array([-1, 2, 3]), 'a')
    c = a.relu()
    c.backward()
    assert np.allclose(a.adjoint, np.array([0, 1, 1]))

@test_case(points=None, hidden=False)
def test_q2b_sigmoid_grad(BearTensor):
    import numpy as np
    a = BearTensor(np.array([-1, 2, 3]), 'a')
    c = a.sigmoid()
    c.backward()
    assert np.allclose(a.adjoint, c.value * (1 - c.value))

