import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from bootstrap_PF import ParticleFilter, normal_pdf
from my_map import Map
from robot import Robot, RobotIllustrator, distance_to_object


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
    def __init__(self, r_cov, v_cov, map_object):
        self.r_cov = r_cov
        self.v_cov = v_cov
        self.map_object = map_object

    def kill_invalid(self, st):
        idx = self.map_object.valid_point(st[:, 0:2])
        return st[idx, :], idx

    def likelihood(self, z, st):

        (valid_st, idx) = self.kill_invalid(st)
        (rays, hits, r_pred) = distance_to_object(valid_st[:, 0],
                                                  valid_st[:, 1],
                                                  valid_st[:, 3],
                                                  self.map_object)
        v_pred = valid_st[:, 2]

        joint_lik = normal_pdf(z[0], r_pred, self.r_cov) * \
            normal_pdf(z[1], v_pred, self.v_cov)

        return joint_lik, valid_st, idx


def draw_particles(ax, st, w):
    x = st[:, 0]
    y = st[:, 1]
    phi = st[:, 3]
    r = 0.1
    lx = np.array([x, x + r * np.cos(phi)]).reshape(2, -1)
    ly = np.array([y, y + r * np.sin(phi)]).reshape(2, -1)

    sc_obj = ax.scatter(x, y, c=w, cmap='Reds', s=1)
    pl_obj = ax.plot(lx, ly, 'r', linewidth=0.3)

    return sc_obj, pl_obj


def run_filter():
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)
    my_map = Map(ax)
    my_map.draw_map()

    pd = prior_dist()
    motion = motion_model(0.1, 0.5 ** 2, 0.5 ** 2)  # dt, v_cov, d_phicov

    meas = meas_model(0.1 ** 2, 0.03 ** 2, my_map)  # r_cov, v_cov

    pf = ParticleFilter(pd, motion, meas, 5000)
    pf.init_state()

    st = pf.x
    w = pf.w

    lo1, lo2 = draw_particles(ax, st, w)

    plt.savefig('pic1_%.5d.png' % 0, dpi=300)
    with open('robot_log.txt') as file:
        reader = csv.reader(file)
        lr, = ax.plot(0, 0, 'r.')
        for i, r in enumerate(reader):
            if i > 0:
                d = np.array(r, dtype=np.float32)

                z = np.array([d[-1], d[3]])  # r, v

                pf.predict()
                pf.update(z)

                st = pf.x
                w = pf.w

                lr.remove()
                lr, = ax.plot(d[1], d[2], 'bo', mfc='none')
                lo1.remove()
                for l in lo2:
                    l.remove()
                lo1, lo2 = draw_particles(ax, st, w)

                plt.savefig('pic1_%.5d.png' % i, dpi=300)

                #plt.pause(0.1)

                pf.resample()
                print('done with %d' % i)


if __name__ == '__main__':
    run_filter()
