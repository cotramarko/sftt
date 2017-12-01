import numpy as np
from numpy.random import multivariate_normal as mvn


class Target():
    def __init__(self, A, H, Q, R, x_0, P_0):
        self.A = np.asmatrix(A)
        self.H = np.asmatrix(H)
        self.Q = np.asmatrix(Q)
        self.R = np.asmatrix(R)

        self.xr = x_0.size
        (self.yr, _) = self.H.shape
        self.x_k = np.asmatrix(x_0.reshape((self.xr, 1)))

    def prop_state(self):
        self.x_k = self.A * self.x_k + \
            mvn(np.zeros((self.xr, )), self.Q).reshape(self.xr, 1)
        return self.x_k

    def prop_meas(self):
        z_k = self.H * self.x_k + \
            mvn(np.zeros((self.yr, )), self.R).reshape(self.yr, 1)

        return z_k
