from otter.test_files import test_case

OK_FORMAT = False

name = "q1"
points = 10

@test_case(points=None, hidden=False)
def test_q1_sub(BearTensor):
    import numpy as np
    a = BearTensor(np.array([1, 2, 3]), 'a')
    b = BearTensor(np.array([4, 5, 6]), 'b')
    c = a - b
    expected_value = np.array([-3, -3, -3])
    assert np.array_equal(c.value, expected_value), f'Forward subtraction failed: {c.value} != {expected_value}'

@test_case(points=None, hidden=False)
def test_q1_add(BearTensor):
    import numpy as np
    a = BearTensor(np.array([1, 2, 3]), 'a')
    b = BearTensor(np.array([4, 5, 6]), 'b')
    c = a + b
    expected_value = np.array([5, 7, 9])
    assert np.array_equal(c.value, expected_value), f'Forward addition failed: {c.value} != {expected_value}'

@test_case(points=None, hidden=False)
def test_q1_mul(BearTensor):
    import numpy as np
    a = BearTensor(np.array([1.0, 2.0, 3.0]), 'a')
    b = BearTensor(np.array([4.0, 5.0, 6.0]), 'b')
    c = a * b
    expected_value = np.array([4.0, 10.0, 18.0])
    assert np.allclose(c.value, expected_value), f'Forward multiplication failed: {c.value} != {expected_value}'

@test_case(points=None, hidden=False)
def test_q1_pow(BearTensor):
    import numpy as np
    a = BearTensor(np.array([-1, 2, 3]), 'a')
    c = a ** 2
    expected_value = np.array([1, 4, 9])
    assert np.array_equal(c.value, expected_value), f'Forward exponentiation failed: {c.value} != {expected_value}'

@test_case(points=None, hidden=False)
def test_q1_matmul(BearTensor):
    import numpy as np
    a = BearTensor(np.array([[-1, 2], [2, 3]]), 'a')
    b = BearTensor(np.array([[4, 2], [0, -3]]), 'a')
    c = a @ b
    expected_value = np.array([[-4, -8], [8, -5]])
    assert np.array_equal(c.value, expected_value), f'Forward matmul failed: {c.value} != {expected_value}'

@test_case(points=None, hidden=False)
def test_q1_dot(BearTensor):
    import numpy as np
    a = BearTensor(np.array([-1, 2, 3]), 'a')
    b = BearTensor(np.array([4, 2, 2]), 'a')
    c = a.dot(b)
    expected_value = np.array(6)
    assert c.value.shape == (), f'dot should return a 0-d array, got shape {c.value.shape}'
    assert np.array_equal(c.value, expected_value), f'Forward dot product failed: {c.value} != {expected_value}'

@test_case(points=None, hidden=False)
def test_q1_sum(BearTensor):
    import numpy as np
    a = BearTensor(np.array([-1, 2, 3]), 'a')
    c = a.sum()
    expected_value = np.array(4)
    assert c.value.shape == (), f'sum should return a 0-d array, got shape {c.value.shape}'
    assert np.array_equal(c.value, expected_value), f'Forward sum failed: {c.value} != {expected_value}'

@test_case(points=None, hidden=False)
def test_q1_mean(BearTensor):
    import numpy as np
    a = BearTensor(np.array([-1, 2, 3]), 'a')
    c = a.mean()
    expected_value = np.array(4 / 3)
    assert c.value.shape == (), f'mean should return a 0-d array, got shape {c.value.shape}'
    assert np.allclose(c.value, expected_value, atol=1e-05), f'Forward sum failed: {c.value} != {expected_value}'

@test_case(points=None, hidden=False)
def test_q1_relu(BearTensor):
    import numpy as np
    a = BearTensor(np.array([-1, 2, 3]), 'a')
    c = a.relu()
    expected_value = np.array([0, 2, 3])
    assert np.allclose(c.value, expected_value, atol=1e-05), f'Forward relu failed: {c.value} != {expected_value}'

@test_case(points=None, hidden=False)
def test_q1_sigmoid(BearTensor):
    import numpy as np
    a = BearTensor(np.array([-1, 2, 3]), 'a')
    c = a.sigmoid()
    expected_value = np.array([0.26894142, 0.88079708, 0.95257413])
    assert np.allclose(c.value, expected_value, atol=1e-05), f'Forward relu failed: {c.value} != {expected_value}'

