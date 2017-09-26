import time
import numpy as np
import matplotlib.pyplot as plt

from bootstrap_PF import ParticleFilter, normal_pdf
from my_map import Map
from robot import Robot, RobotIllustrator


def distance_to_object(x, y, phi, map_object):
    ''' Computes the distance from (x,y) with heading phi towards the nearest
    object found in the map. The map is held by the map_object '''
    base_point = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1))).reshape(1, -1, 2)
    d_vec = np.hstack((np.cos(phi).reshape(-1, 1), np.sin(phi).reshape(-1, 1))).reshape(1, -1, 2)

    dists = np.arange(0, 15, 0.01).reshape(-1, 1, 1)

    ray = base_point + dists * d_vec  # DxNx2
    (D, N, _) = ray.shape

    all_points = ray.reshape(-1, 2)

    idx = np.bitwise_not(map_object.valid_point(all_points))
    idx = idx.reshape(D, N)

    invalid_dists = dists.reshape(-1, 1) * idx
    invalid_dists = invalid_dists.flatten()
    invalid_dists[invalid_dists == 0] = np.nan
    invalid_dists = invalid_dists.reshape(D, N)

    z = np.nanmin(invalid_dists, axis=0)  # Nx2
    idx_z = np.nanargmin(invalid_dists, axis=0)

    ray_hits = ray[idx_z, np.arange(N), :]  # Nx2
    ray = ray.transpose(1, 0, 2)  # NxDx2

    return ray, ray_hits, z


class prior_dist():
    def __init__(self):
        pass

    def draw_samples(self, N):
        x = np.random.uniform(0, 10, (N, 1))
        y = np.random.uniform(0, 10, (N, 1))
        v = np.random.uniform(-0.5, 0.5, (N, 1))
        phi = np.random.uniform(0, 2 * np.pi, (N, 1))
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
        phi = st[:, 3] + self.dt * st[:, 4]
        dphi = st[:, 4] + \
            np.random.normal(scale=np.sqrt(self.dphi_cov), size=st[:, 4].shape)

        st = np.stack((x, y, v, phi, dphi), axis=1)
        return st


class meas_model():
    def __init__(self, r_cov, map_object):
        self.r_cov = r_cov
        self.map_object = map_object

    def _kill_invalid(self, st):
        idx = self.map_object.valid_point(st[:, 0:2])
        return st[idx, :]

    def likelihood(self, z, st):

        valid_st = self._kill_invalid(st)

        (rays, hits, z_pred) = distance_to_object(valid_st[:, 0],
                                                  valid_st[:, 1],
                                                  valid_st[:, 3],
                                                  self.map_object)

        return normal_pdf(z_pred, z, self.r_cov), rays, hits


if __name__ == '__main__':
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 20)
    my_map = Map(ax)
    my_map.draw_map()

    p = prior_dist()
    state = p.draw_samples(10)

    # ax.plot(state[:, 0], state[:, 1], 'r.')

    model = motion_model(1, 1, 1)
    state = model.propagate(state)
    ax.plot(state[:, 0], state[:, 1], 'b.')

    meas = meas_model(0.1, my_map)
    t = time.time()
    liks, rays, hits = meas.likelihood(2, state)
    print(np.min(liks), np.max(liks), np.sum(liks))

    print(time.time() - t)
    for r in rays:
        ax.plot(r[:, 0], r[:, 1], ':r')
    for h in hits:
        ax.plot(h[0], h[1], 'xr')

plt.axis('equal')
plt.show()
