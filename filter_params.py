import numpy as np
from robot import distance_to_object
from bootstrap_PF import normal_pdf


class prior_dist():
    def __init__(self):
        pass

    def draw_samples(self, N):
        x = np.random.uniform(0, 10, (N, 1))
        y = np.random.uniform(0, 10, (N, 1))

#        x = np.random.normal(loc=4, scale=1, size=(N, 1))
#        y = np.random.normal(loc=2, scale=1, size=(N, 1))

#        v = np.zeros((N, 1))
        v = np.random.uniform(-0.5, 0.5, (N, 1))
        phi = np.random.uniform(0, 2 * np.pi, (N, 1))
        phi = np.ones_like(phi) * np.pi / 2
        dphi = np.random.uniform(-np.pi / 4, np.pi / 4, (N, 1))

        state = np.concatenate((x, y, v, phi, dphi), axis=1)
        return state


class motion_model():
    def __init__(self, dt, v_cov, dphi_cov):
        self.dt = dt
        self.v_cov = v_cov
        self.dphi_cov = dphi_cov

    def propagate(self, st):
        x = st[:, 0] + st[:, 2] * self.dt * np.cos(st[:, 3])
        y = st[:, 1] + st[:, 2] * self.dt * np.sin(st[:, 3])
        v = st[:, 2] + \
            np.random.normal(scale=np.sqrt(self.v_cov), size=st[:, 2].shape)

        print(np.min(v), np.max(v))
        phi = st[:, 3] + self.dt * st[:, 4]
        dphi = st[:, 4] + \
            np.random.normal(scale=np.sqrt(self.dphi_cov), size=st[:, 4].shape)

        st = np.stack((x, y, v, phi, dphi), axis=1)
        return st


class meas_model():
    def __init__(self, d_cov, v_cov, dphi_cov, map_object):
        self.d_cov = d_cov
        self.v_cov = v_cov
        self.dphi_cov = dphi_cov

        self.map_object = map_object

    def kill_invalid(self, st):
        idx = self.map_object.valid_point(st[:, 0:2])
        return st[idx, :], idx

    def likelihood(self, z, st):

        (valid_st, idx) = self.kill_invalid(st)
        (rays, hits, d_pred) = distance_to_object(valid_st[:, 0],
                                                  valid_st[:, 1],
                                                  valid_st[:, 3],
                                                  self.map_object)
        v_pred = valid_st[:, 2]
        dphi_pred = valid_st[:, 4]

        # normal_pdf(z[0], d_pred, self.d_cov)
        joint_lik = normal_pdf(z[0], d_pred, self.d_cov) * \
            normal_pdf(z[1], v_pred, self.v_cov) * \
            normal_pdf(z[2], dphi_pred, self.dphi_cov)

        return joint_lik, valid_st, idx
