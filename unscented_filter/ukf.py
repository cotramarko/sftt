import numpy as np


class UKF():
    def __init__(self, fx, hx, Q, R, x0, P0):
        self.fx = fx
        self.hx = hx
        self.Q = Q
        self.R = R
        self.x = x0.reshape(-1, 1)
        self.P = P0
        (self.n, _) = Q.shape

        w0 = 1 - self.n / 3
        wi = (1 - w0) / (2 * self.n)
        self.w = \
            np.concatenate((wi * np.ones(self.n), w0 * np.ones(1), wi * np.ones(self.n)), axis=0)

    def create_sp(self):
        sp_0 = self.x[None, ...].transpose()

        P_root = np.linalg.cholesky(self.P)
        sp_pos = self.x[None, ...] + np.sqrt(3) * P_root
        sp_neg = self.x[None, ...] - np.sqrt(3) * P_root
        sp_pos = sp_pos.transpose()
        sp_neg = sp_neg.transpose()

        sig_points = np.concatenate((sp_neg, sp_0, sp_pos), axis=0)
        return sig_points

    def predict(self, T):
        sig_points = self.create_sp()
        self.x = np.sum(self.fx(sig_points, T) * self.w[:, None, None], axis=0)

        delta = self.fx(sig_points, T) - self.x[None, ...]

        delta_sq = np.matmul(delta, delta.transpose(0, 2, 1))

        self.P = self.Q + np.sum(delta_sq * self.w[:, None, None], axis=0)

    def update(self, y):
        sig_points = self.create_sp()
        y_pred = np.sum(self.hx(sig_points) * self.w[:, None, None], axis=0)

        delta_spx = sig_points - self.x[None, ...]
        delta_spy = self.hx(sig_points) - y_pred[None, ...]
        delta_spxy = np.matmul(delta_spx, delta_spy.transpose(0, 2, 1))

        Pxy = np.sum(delta_spxy * self.w[:, None, None], axis=0)

        delta_spyy = np.matmul(delta_spy, delta_spy.transpose(0, 2, 1))
        Pyy = self.R + np.sum(delta_spyy * self.w[:, None, None], axis=0)

        K = np.matmul(Pxy, np.linalg.inv(Pyy))
        self.x = self.x + np.matmul(K, y[:, None] - y_pred)
        self.P = self.P - K @ Pyy @ K.transpose()

    def get_state(self):
        """
        return (self.x, self.P)
        self.x.shape = (k, 1)
        self.P.shape = (k, k)
        """
        return self.x, self.P
