import numpy as np


def mvn_pdf(x, mu, sigma):
    # custom version for 4D arrays
    if len(x.shape + mu.shape + sigma.shape) != 12:
        print(len(x.shape + mu.shape + sigma.shape))
        raise BaseException('Input must be 4D')

    d = (x - mu)
    dt = (x - mu).transpose(0, 1, 3, 2)

    a = 1 / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(sigma))
    dts = np.matmul(dt, np.linalg.inv(sigma))
    dtsd = np.matmul(dts, d)

    b = -0.5 * dtsd
    b = b.squeeze()

    pdf = a * np.exp(b)
    return pdf


class IdentityBank():
    """docstring for IdentityBank"""
    def __init__(self, max_nbr=10000):
        self.max_nbr = max_nbr
        self.all_ids = np.arange(0, self.max_nbr)

    def create_unique_ids(self, N):
        ids = np.random.choice(self.all_ids, N, replace=False)
        idx = np.where(self.all_ids == ids)
        self.all_ids = np.delete(self.all_ids, idx)
        return ids[:, None]

    def insert_ids(self, ids):
        self.all_ids = np.append(self.all_ids, ids)


class GaussComp():
    """docstring for GaussComp"""
    def __init__(self, A, H, R, Q):
        self.A = A  # 1,K,K
        self.H = H  # 1,L,K
        self.R = R  # 1,L,L
        self.Q = Q  # 1,K,K

    def predict(self, x, P):
        """(N, K) = x.shape
           (N, K, K) = P.shape """

        x_p = np.matmul(self.A[None, ...], x[..., None]).squeeze()

        AP = np.matmul(self.A[None, ...], P).squeeze()
        APAT = np.matmul(AP, self.A.transpose())
        P_p = APAT + self.Q[None, ...]

        return x_p, P_p

    def update(self, x, P, z):
        """(N, K) = x.shape
           (N, K, K) = P.shape
           (M, L) = z.shape """

        z_pred = np.matmul(self.H[None, ...], x[..., None]).squeeze()

        v = z[:, None, :] - z_pred[None, ...]  # Nbr meas - gauss comps - state size

        HP = np.matmul(self.H[None, ...], P).squeeze()

        HPHT = np.matmul(HP, self.H.transpose())
        S = HPHT + self.R[None, ...]

        PHT = np.dot(P, self.H.transpose())
        K = np.matmul(PHT, np.linalg.inv(S))

        Kv = np.matmul(K[None, ...], v[..., None]).squeeze()
        x_u = x[None, ...] + Kv

        KS = np.matmul(K, S)
        KSKT = np.matmul(KS, K.transpose(0, 2, 1))

        P_u = P - KSKT
        P_u = P_u[None, ...]  # Covariance same for all measurements, add dummy axis

        q = mvn_pdf(z[:, None, :, None], z_pred[None, ..., None], S[None, ...])

        return x_u, P_u, q[..., None]


class PhdTracker(object):
    """docstring for PhdTracker"""
    # TODO: Add a unique identifier as well as id-generator in this class
    def __init__(self, max_comps, Ps, Pd, x_birth, P_birth, K_rate, gauss_comp, rfs_birth_prob=0.1):
        self.max_comps = max_comps  # Max nbr of gaussian components in the mixture
        self.Ps = Ps  # Survival probability, replace with function down the line
        self.Pd = Pd  # Detection probability
        self.x_birth = x_birth  # gauss means of the birth RFS (N,K shape, N - nbr, K - vector dim)
        self.P_birth = P_birth  # gauss cov of the birth RFS (N,K shape, N - nbr, KxK - cov mat dim)
        self.K_rate = K_rate  # poisson clutter rate (false detections)

        (N, _) = self.x_birth.shape
        self.birth_nbr = N
        self.w_birth = rfs_birth_prob * np.ones((N, 1))  # init gauss comp weights

        self.gauss_comp = gauss_comp  # handle to a gauss comp class
        self.id_bank = IdentityBank()

        self.x = self.x_birth
        self.P = self.P_birth
        self.w = self.w_birth
        self.track_id = self.id_bank.create_unique_ids(N)

    def predict(self):
        (x_pred, P_pred) = \
            self.gauss_comp.predict(self.x, self.P)

        w_pred = self.Ps * self.w

        id_birth = self.id_bank.create_unique_ids(self.birth_nbr)

        self.x = np.vstack((x_pred, self.x_birth))
        self.P = np.vstack((P_pred, self.P_birth))
        self.w = np.vstack((w_pred, self.w_birth))
        self.track_id = np.vstack((self.track_id, id_birth))

    def update(self, z):
        # z - dimension (M, L)
        x_nd = self.x  # not detected targets
        P_nd = self.P
        w_nd = (1 - self.Pd) * self.w
        track_id_nd = self.track_id

        (x_up, P_up, q) = \
            self.gauss_comp.update(self.x, self.P, z)

        w_up = self.calculate_updated_weights(q)  # dimension (M, N, 1)

        (m, n, k) = x_up.shape

        # Concat the meas,comp dim into a meas+comp-dim
        w_up = w_up.reshape(-1, 1)
        x_up = x_up.reshape(-1, k)

        track_id_up = np.tile(self.track_id, (m, 1))

        # Covariance same for all measurements per target, must repmat
        P_up = np.repeat(P_up, m, axis=0).reshape(-1, k, k)

        self.x = np.vstack((x_nd, x_up))
        self.P = np.vstack((P_nd, P_up))
        self.w = np.vstack((w_nd, w_up))
        self.track_id = np.vstack((track_id_nd, track_id_up))

    def calculate_updated_weights(self, q):
        top = self.Pd * self.w[None, ...] * q  # (M, N, 1)
        ''' for each measurement we marginalize over the weights '''
        bot = self.K_rate + np.sum(top, axis=1)  # (M, 1)
        up_w = top / bot[:, None, :]  # (M, N, 1)

        return up_w

    def prune(self, prune_threshold=0.1):
        idx_valid = np.argwhere(self.w.flatten() > prune_threshold).flatten()

        sum_all_weights = np.sum(self.w.flatten())
        sum_valid_weights = np.sum(self.w[idx_valid, :].flatten())

        scale_ratio = sum_all_weights / sum_valid_weights

        self.w = self.w[idx_valid, :] * scale_ratio
        self.x = self.x[idx_valid, ...]
        self.P = self.P[idx_valid, ...]
        self.track_id = self.track_id[idx_valid, ...]

        # TODO: remove corresponding track id and put them back into the id bank

    def merge(self, merge_threshold=10):
        (w_new, x_new, P_new, track_id_new) = ([], [], [], [])

        while len(self.w) > 0:
            j = np.argmax(self.w.flatten())

            d = self.x[j, ...] - self.x
            d = d[..., None]
            dt = d.transpose(0, 2, 1)
            dtp = np.matmul(dt, np.linalg.inv(self.P))
            dtpd = np.matmul(dtp, d).squeeze()

            idx_merge = np.argwhere(dtpd < merge_threshold).flatten()
            wj = np.sum(self.w[idx_merge, ...].flatten(), axis=0)
            xj = (1 / wj) * \
                np.sum(self.w[idx_merge, ...] * self.x[idx_merge, ...], axis=0)

            d = xj - self.x[idx_merge, ...]
            d = d[..., None]
            dt = d.transpose(0, 2, 1)
            ddt = np.matmul(d, dt)
            Pj = (1 / wj) * \
                np.sum(self.w[idx_merge, :, None] * (self.P[idx_merge, ...] + ddt), axis=0)

            w_new.append(wj)
            x_new.append(xj)
            P_new.append(Pj)
            track_id_new.append(self.track_id[j, ...])

            self.w = np.delete(self.w, idx_merge, axis=0)
            self.x = np.delete(self.x, idx_merge, axis=0)
            self.P = np.delete(self.P, idx_merge, axis=0)

            self.id_bank.insert_ids(self.track_id[idx_merge, ...])
            self.track_id = np.delete(self.track_id, idx_merge, axis=0)

        self.w = np.array(w_new)
        self.w = self.w[:, None]
        self.x = np.array(x_new)
        self.P = np.array(P_new)
        self.track_id = np.array(track_id_new)

        # TODO: Improve this and break it out as a separate function
        unique_ids = np.unique(self.track_id)
        for u_id in unique_ids:
            # find all occurances of u_id
            match_idx, = np.where(self.track_id.flatten() == u_id)
            if len(match_idx) > 1:
                # find max weight
                j = np.argmax(self.track_id[match_idx, ...])
                # remove the max weight track id from the matching idx
                match_idx = np.delete(match_idx, j)
                # generate new track ids
                N = len(match_idx)
                new_ids = self.id_bank.create_unique_ids(N)
                self.track_id[match_idx, :] = new_ids

    def get_tracks(self, w_threshold=0.5):
        idx_tracks = np.argwhere(self.w.flatten() > w_threshold).flatten()

        return self.x[idx_tracks, :], self.P[idx_tracks, ...], self.track_id[idx_tracks, ...]


if __name__ == '__main__':
    A = np.array([[3, 1], [4, 2]])
    H = np.array([[1, 0], [0, 1]])
    R = np.diag([0.1, 0.2])
    Q = np.array([[1, 0], [0, 1]])

    x = np.array([[1, 2], [3, 4], [5, 6]])
    P = np.array([[[1, 0],
                   [0, 1]],
                  [[1, 0.25],
                   [0.25, 1]],
                  [[1, 0.25],
                   [0.25, 1]]])

    z = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])

    gauss_comp = GaussComp(A, H, R, Q)

#    gauss_comp.predict(x, P)
#    gauss_comp.update(x, P, z)
#    gauss_comp.likelihood(x, P, z)

    # max_comps, Ps, Pd, x_birth, P_birth, K_rate, gauss_comp
    tracker = PhdTracker(max_comps=4,
                         Ps=0.9,
                         Pd=0.99,
                         x_birth=x,
                         P_birth=P,
                         K_rate=0.001,
                         gauss_comp=gauss_comp)

    tracker.predict()
    tracker.update(z)
    tracker.prune()
    # tracker.merge_tracks()
    (x, P) = tracker.get_tracks()

    print(x)
