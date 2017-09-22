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
    def __init__(self, prior_dist, motion_model, meas_model, N_particles):
        self.prior_dist = prior_dist
        self.meas_model = meas_model
        self.motion_model = motion_model
        self.N_particles = N_particles

        self.init_state()

    def init_state(self):
        self.x = self.prior_dist.draw_samples(self.N_particles)
        self.w = np.ones(shape=(self.N_particles)) / self.N_particles

    def predict(self):
        self.x = self.motion_model.propagate(self.x)

    def update(self, z):
        self.w = self.w * self.meas_model.likelihood(z, self.x)
        self.w = self.w / np.sum(self.w)

    def resample(self):
        self.x = np.random.choice(self.x, p=self.w)
        self.w = np.ones(shape=(self.N_particles)) / self.N_particles


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

        def likelihood(self, z, x):
            pass

    x = np.array([[3, 4], [2, 1], [7, 8], [0, 0]])

    mm = motion_model()
    y = mm.propagate(x)

    N_particles = 2
    pf = ParticleFilter(prior_dist(), motion_model(), None, N_particles)
    pf.init_state()

    pdf = mvn_pdf(x, mu=np.array([0, 0]), sigma=np.array([[1, 0.3], [0.2, 1]]))
    print('custom:', pdf)

    print('scipy: ', mvn.pdf(x, np.array([0, 0]), np.array([[1, 0.3], [0.2, 1]])))
