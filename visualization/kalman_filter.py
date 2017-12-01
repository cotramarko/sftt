import numpy as np


class KF():
    def __init__(self, A, H, Q, R, x_0, P_0):
        # Model parameters
        self.A = np.asmatrix(A)
        self.H = np.asmatrix(H)
        self.Q = np.asmatrix(Q)
        self.R = np.asmatrix(R)

        # Initial state
        self._x = np.asmatrix(x_0)
        self._P = np.asmatrix(P_0)

        self.V = np.asmatrix(self._x.shape)
        self.S = np.asmatrix(self.R.shape)
        self.K = np.asmatrix(self.A.shape)

    def predict(self):
        self._x = self.A * self._x
        self._P = self.A * self._P * self.A.transpose() + self.Q

    def update(self, Z):
        self.S = self.H * self._P * self.H.transpose() + self.R
        self.V = Z - self.H * self._x
        self.K = self._P * self.H.transpose() * np.linalg.inv(self.S)

        self._x = self._x + self.K * self.V
        self._P = self._P - self.K * self.S * self.K.transpose()

    def get_state(self):
        return self._x, self._P
