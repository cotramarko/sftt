import unittest
import numpy as np

from kalman_filter import KalmanFilter

class TestKalmanFilter(unittest.TestCase):
    def setUp(self):
        A = np.eye(2)
        H = np.eye(2)
        Q = np.eye(2)
        R = np.eye(2)

        x_0 = np.array([1, 1])
        P_0 = np.eye(2)
        self.kalman_filter = KalmanFilter(A, H, Q, R, x_0, P_0)

    def test_predict(self):
        self.kalman_filter.predict()
        (x, P) = self.kalman_filter.get_state()
        self.assertTrue((x == np.array([1, 1])).all())
        self.assertTrue((P == 2 * np.eye(2)).all())

    def test_update(self):
        self.kalman_filter.update(np.array([0, 0]))
        (x, P) = self.kalman_filter.get_state()
        self.assertTrue((x == np.array([0.5, 0.5])).all())
        self.assertTrue((P == 0.5 * np.eye(2)).all())


if __name__ == '__main__':
    unittest.main(verbosity=2)
