import numpy as np
import unittest

from ukf import UKF


class TestUKF(unittest.TestCase):
    def setUp(self):

        def fx(x, T): return x

        def hx(x): return x

        x0 = np.array([0, 0])
        P0 = np.array([[1, 0.5], [0.5, 1]])
        Q = np.eye(2)
        R = np.eye(2)

        self.ukf = UKF(fx, hx, Q, R, x0, P0)

    def test_predict(self):
        self.ukf.predict(T=1)
        (x, P) = self.ukf.get_state()
        x = np.round(x, decimals=5)
        P = np.round(P, decimals=5)

        self.assertTrue((x == np.array([0, 0])).all())
        self.assertTrue((P == np.array([[2, 0.5], [0.5, 2]])).all())

    def test_update(self):
        self.ukf.predict(T=1)
        self.ukf.update(np.array([1, 1]))

        (x, P) = self.ukf.get_state()
        x = np.round(x, decimals=5)
        P = np.round(P, decimals=5)

        self.assertTrue((x == np.array([0.71429,  0.71429])).all())
        self.assertTrue((P == np.array([[0.65714,  0.05714], [0.05714, 0.65714]])).all())


if __name__ == '__main__':
    unittest.main()
