import numpy as np


def mvn_pdf(x, mu, sigma):
    d = (x - mu)[:, np.newaxis, :]  # Nx1x2 column vector
    dt = (x - mu)[:, :, np.newaxis]  # Nx2x1 row vector

    a = 1 / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(sigma))
    b = -0.5 * np.sum(dt * np.linalg.inv(sigma) * d, axis=(2, 1))

    pdf = a * np.exp(b)
    return pdf


def normal_pdf(x, mu, cov):
    a = 1 / np.sqrt(2 * np.pi * cov)
    b = np.exp(-np.square(x - mu) / (2 * cov))

    return a * b


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
        # TODO use kill_invalid only once
        (lik, self.x, idx) = self.meas_model.likelihood(z, self.x)
        self.w = self.w[idx]

        self.w = self.w * lik
        self.w = self.w / np.sum(self.w)

    def resample(self):
        idx = np.random.choice(np.arange(0, self.w.size), p=self.w, size=self.N)
        self.x = self.x[idx, :]
        self.w = np.ones(shape=(self.N)) / self.N

    def get_MMSE(self):
        raise NotImplementedError

    def get_MAP(self):
        raise NotImplementedError
