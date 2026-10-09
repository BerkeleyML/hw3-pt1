from otter.test_files import test_case

OK_FORMAT = False

name = "q4"
points = 10

@test_case(points=None, hidden=False)
def test_q4_predict_single_example(predict):
    import numpy as np
    x = np.zeros(11)
    prediction = np.asarray(predict(x))
    assert prediction.size == 1, f'Expected one scalar prediction, got shape {prediction.shape}'
    assert np.isfinite(prediction).all(), 'Prediction must be finite'

