import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn


def mvn_pdf(x, mu, sigma):
    d = (x - mu)[:, np.newaxis, :]  # Nx1x2 column vector
    dt = (x - mu)[:, :, np.newaxis]  # Nx2x1 row vector

    a = 1 / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(sigma))
    b = -0.5 * np.sum(dt * np.linalg.inv(sigma) * d, axis=(2, 1))

    pdf = a * np.exp(b)
    return pdf


class ParticleFilter():
    """docstring for ParticleFilter"""
    def __init__(self, prior_dist, motion_model, meas_model, N):
        self.prior_dist = prior_dist
        self.meas_model = meas_model
        self.motion_model = motion_model
        self.N = N

        self.init_state()

    def init_state(self):
        self.x = self.prior_dist.draw_samples(self.N)
        self.w = np.ones(shape=(self.N)) / self.N

    def predict(self):
        self.x = self.motion_model.propagate(self.x)

    def update(self, z):
        self.w = self.w * self.meas_model.likelihood(z, self.x)
        self.w = self.w / np.sum(self.w)

    def resample(self):
        idx = np.random.choice(np.arange(0, self.N), p=self.w, size=self.N)
        self.x = self.x[idx, :]
        self.w = np.ones(shape=(self.N)) / self.N

    def get_MMSE(self):
        raise NotImplementedError

    def get_MAP(self):
        raise NotImplementedError


# ============================================================================================
#
#   TESTS
#
# ============================================================================================

if __name__ == '__main__':

    class prior_dist():
        def __init__(self):
            pass

        def draw_samples(self, N):
            return np.random.multivariate_normal([0, 0], np.diag([.5, .5]), N)

    class motion_model():
        def __init__(self):
            self.A = np.diag([1, 1])
            self.q_cov = 0.1 ** 2
            self.Q = self.q_cov * np.diag([1, 1])

        def propagate(self, x):
            N = x.shape[0]
            x = np.sum(x[:, np.newaxis, :] * self.A.transpose(), axis=2) \
                + np.random.multivariate_normal([0, 0], self.Q, N)
            return x

    class meas_model():
        def __init__(self):
            self.H = np.diag([1, 1])
            self.r_cov = 0.5 ** 2
            self.R = self.r_cov * np.diag([1, 1])

        def likelihood(self, z, x):
            return mvn_pdf(x, z, self.R)

    x = np.array([[3, 4], [2, 1], [7, 8], [0, 0]])

    model = motion_model()
    y = model.propagate(x)

    N = 4
    z = np.array([0, 0])

    start = time.process_time()
    pf = \
        ParticleFilter(prior_dist(), motion_model(), meas_model(), N)

    pf.init_state()
    pf.predict()
    pf.update(z)
    pf.resample()
    end = time.process_time()

    print('Execution time: ', (end - start))
    print(pf.x)
    print(pf.w)
