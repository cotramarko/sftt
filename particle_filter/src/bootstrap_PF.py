import numpy as np
import utils_np

import torch
import utils_torch


class ParticleFilter():
    """Generic particle filter"""

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
        (lik, self.x, idx) = self.meas_model.likelihood(z, self.x)
        self.w = self.w[idx]

        self.w = self.w * lik
        self.w = self.w / np.sum(self.w)

    def resample(self):
        idx = np.random.choice(np.arange(0, self.w.size), p=self.w, size=self.N)
        self.x = self.x[idx, :]
        self.w = np.ones(shape=(self.N)) / self.N

    def get_MMSE(self):
        mmse = np.sum(self.w[:, None] * self.x, axis=0)
        return mmse

    def get_MAP(self):
        map_idx = np.argmax(self.w)
        map_ = self.x[map_idx, :]
        return map_


class prior_dist():
    """Represents a prior distribution over the robot's state"""

    def __init__(self):
        pass

    def draw_samples(self, N):
        x = np.random.uniform(0, 10, (N, 1))
        y = np.random.uniform(0, 10, (N, 1))
        v = np.random.uniform(-0.5, 0.5, (N, 1))

#        x = np.random.normal(loc=4, scale=1, size=(N, 1))
#        y = np.random.normal(loc=2, scale=1, size=(N, 1))
#        v = np.zeros((N, 1))

        phi = np.random.uniform(0, 2 * np.pi, (N, 1))
        phi = np.ones_like(phi) * np.pi / 2
        dphi = np.random.uniform(-np.pi / 4, np.pi / 4, (N, 1))

        state = np.concatenate((x, y, v, phi, dphi), axis=1)
        return state


class motion_model():
    """Represents how the robot's state evolves over time."""

    def __init__(self, dt, v_cov, dphi_cov):
        ''' dt - sampling time
            v_cov  - covariance for velocity
            dphi_cov - covariance for turningratio '''
        self.dt = dt
        self.v_cov = v_cov
        self.dphi_cov = dphi_cov

    def propagate(self, st):
        """ Propagates the robot's state for a single step """
        x = st[:, 0] + st[:, 2] * self.dt * np.cos(st[:, 3])
        y = st[:, 1] + st[:, 2] * self.dt * np.sin(st[:, 3])
        v = st[:, 2] + \
            np.random.normal(scale=np.sqrt(self.v_cov), size=st[:, 2].shape)

        phi = st[:, 3] + self.dt * st[:, 4]
        dphi = st[:, 4] + \
            np.random.normal(scale=np.sqrt(self.dphi_cov), size=st[:, 4].shape)

        st = np.stack((x, y, v, phi, dphi), axis=1)
        return st


class meas_model():
    def __init__(self, d_cov, v_cov, dphi_cov, room, boundry):
        self.d_cov = d_cov
        self.v_cov = v_cov
        self.dphi_cov = dphi_cov

        self.room = room
        self.boundry = boundry

    def kill_invalid(self, st):
        idx = utils_np.valid_point(self.room, self.boundry, st[:, 0:2])
        return st[idx, :], idx

    def likelihood(self, z, st):
        (valid_st, idx) = self.kill_invalid(st)
        (rays, hits, d_pred) = utils_np.distance_to_object(valid_st[:, 0],
                                                           valid_st[:, 1],
                                                           valid_st[:, 3],
                                                           self.room,
                                                           self.boundry)
        v_pred = valid_st[:, 2]
        dphi_pred = valid_st[:, 4]

        joint_lik = utils_np.normal_pdf(z[0], d_pred, self.d_cov) * \
            utils_np.normal_pdf(z[1], v_pred, self.v_cov) * \
            utils_np.normal_pdf(z[2], dphi_pred, self.dphi_cov)

        return joint_lik, valid_st, idx


class meas_model_torch():
    def __init__(self, d_cov, v_cov, dphi_cov, room, boundry):
        self.d_cov = torch.Tensor([d_cov]).type(torch.FloatTensor).cuda()
        self.v_cov = torch.Tensor([v_cov]).type(torch.FloatTensor).cuda()
        self.dphi_cov = torch.Tensor([dphi_cov]).type(torch.FloatTensor).cuda()

        self.room = room
        self.boundry = boundry

        self.rects = torch.Tensor([self.room] + self.boundry).type(torch.FloatTensor).cuda()
        self.dists = torch.arange(0, 15, 0.01).view(-1, 1, 1).type(torch.FloatTensor).cuda()

    def kill_invalid(self, st):
        idx = utils_torch.valid_point(st[:, 0:2], self.rects)
        return st[idx.type(torch.cuda.LongTensor), :], idx

    def likelihood(self, z, st):
        st = torch.from_numpy(st).type(torch.FloatTensor).cuda()
        z = torch.from_numpy(z).type(torch.FloatTensor).cuda()

        (valid_st, idx) = self.kill_invalid(st)
        (_, _, d_pred) = utils_torch.distance_to_object(valid_st[:, 0],
                                                        valid_st[:, 1],
                                                        valid_st[:, 3],
                                                        self.dists,
                                                        self.rects)
        v_pred = valid_st[:, 2]
        dphi_pred = valid_st[:, 4]

        joint_lik = utils_torch.normal_pdf(z[0], d_pred, self.d_cov) * \
            utils_torch.normal_pdf(z[1], v_pred, self.v_cov) * \
            utils_torch.normal_pdf(z[2], dphi_pred, self.dphi_cov)

        return joint_lik.cpu().numpy(), valid_st.cpu().numpy(), idx.cpu().numpy()
