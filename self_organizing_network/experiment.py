import numpy as np
import matplotlib.pyplot as plt
from sftt.unscented_filter.ukf import UKF


def draw_cov(x, cov, col='k'):
    phi = np.linspace(0, 2 * np.pi, 100)
    sig3 = x[:, None] + 3 * np.dot(np.linalg.cholesky(cov), np.array([np.cos(phi), np.sin(phi)]))
    plt.plot(sig3[0, :], sig3[1, :], col)


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


def in_range(z, range):
    z = z <= range
    return z


class nodeNetwork():
    def __init__(self, base, x_true):
        self.x = np.concatenate((base, x_true))

    def get_connections(self, range=5):
        z_matrix = cross_distance(self.x)
        (r, _) = z_matrix.shape
        (idx_rows, idx_cols) = np.triu_indices(r, k=1)

        z_triu = z_matrix[idx_rows, idx_cols]
        mask_range = in_range(z_matrix, range=range)
        mask_range[0, 1] = False
        mask_range[0, 2] = False
        keep_elements = mask_range[idx_rows, idx_cols]

        self.z = z_triu[keep_elements]

        return (idx_rows, idx_cols, keep_elements)

    def get_measurements(self, r_cov):
        m = self.z + \
            np.random.multivariate_normal(np.zeros_like(
                self.z), r_cov * np.diag(np.ones_like(self.z)))
        return m


def connect_positions(x, idx_rows, idx_cols, keep_elements):
    cval = 0.75
    gray = (cval, cval, cval, cval)
    for (r, c, b) in zip(idx_rows, idx_cols, keep_elements):
        if b:
            plt.plot(x[[r, c], 0], x[[r, c], 1], '-', color=gray)


class meas_func():
    def __init__(self, base, keep_elements):
        self.base = base
        self.keep_elements = keep_elements

    def __call__(self, sig_points):
        Z = []
        for x in sig_points:
            x = np.concatenate((base, x.reshape(-1, 2)))
            z = cross_distance(x)
            (r, _) = z.shape
            (idx_rows, idx_cols) = np.triu_indices(r, k=1)

            z_triu = z[idx_rows, idx_cols]
            z = z_triu[self.keep_elements]
            Z.append(z)

        Z = np.array(Z)
        return Z[..., None]


if __name__ == '__main__':
    np.random.seed(1337)

    N = 6
    sensor_range = 7
    base = np.array([[5, 5], [5, 7], [3, 4]])
    x_true = np.random.uniform(size=(N, 2)) * 14 - 2

    node_network = nodeNetwork(base, x_true)
    (idx_rows, idx_cols, keep_elements) = node_network.get_connections(range=7)

    z = node_network.z

    h = meas_func(base, keep_elements)

    # x0 = np.ones(shape=(N, 2)).flatten()
    x0 = np.random.uniform(size=(N, 2)).flatten() * 14 - 2
    P0 = diag_block_mat_boolindex([np.eye(2) * 100**2] * N)

    cov_mask = diag_block_mat_boolindex([np.ones((2, 2))] * N)

    Q = np.eye(len(x0))
    r_cov = 0.1 ** 2
    R = r_cov * np.eye(len(z))

    def fx(x, T):
        return x

    m = node_network.get_measurements(r_cov)
    print(m)

    ukf = UKF(fx, h, Q, R, x0, P0)

    xx = []
    PP = []

    K = 1
    for _ in range(K):
        m = node_network.get_measurements(r_cov)
        ukf.update(m)
        (x, P) = ukf.get_state()
        ukf.P = P + np.diag(np.ones(len(x0))) * 1
        xx.append(x.reshape(-1, 2))
        PP.append(P)

    xx = np.array(xx)
    PP = np.array(PP)

    plt.figure()
    for j in range(K):
        nodes = np.concatenate((base, x_true))
        connect_positions(nodes, idx_rows, idx_cols, keep_elements)
        plt.plot(nodes[:, 0], nodes[:, 1], 'ko', mfc=(1, 1, 1))
        plt.plot(base[:, 0], base[:, 1], 'co', mfc='c')

        for i in range(N):
            plt.plot(xx[j, i, 0], xx[j, i, 1], 'r.')
            draw_cov(xx[j, i, :], PP[j, (i * 2):((i + 1) * 2), (i * 2):((i + 1) * 2)], col='r')
        plt.axis([-5, 10, -2, 13])
        plt.pause(0.25)
        plt.show()
        plt.clf()
