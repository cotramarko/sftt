import numpy as np
import matplotlib.pyplot as plt


def diag_block_mat_boolindex(L):
    shp = L[0].shape
    mask = np.kron(np.eye(len(L)), np.ones(shp)) == 1
    out = np.zeros(np.asarray(shp) * len(L), dtype=int)
    out[mask] = np.concatenate(L).ravel()
    return out


def cross_distance(x):
    (N, _) = x.shape
    delta = x[None, ...] - x[:, None, :]
    z = np.linalg.norm(delta.reshape(-1, 2), ord=2, axis=1)
    return z.reshape(N, N)


def in_range(meas_mat, sensor_range):
    in_range_mat = (meas_mat > 0) & (meas_mat < sensor_range)
    return in_range_mat


def draw_cov(x, cov, col='k'):
    phi = np.linspace(0, 2 * np.pi, 100)
    sig3 = x[:, None] + 3 * np.dot(np.linalg.cholesky(cov), np.array([np.cos(phi), np.sin(phi)]))
    plt.plot(sig3[0, :], sig3[1, :], col)
