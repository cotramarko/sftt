import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

from bootstrap_PF import ParticleFilter, prior_dist, motion_model, meas_model, meas_model_torch
from my_map import Map


def draw_particles(ax, st, w, ms):
    x = st[:, 0]
    y = st[:, 1]

    sorted_idx = w.argsort()
    x = x[sorted_idx]
    y = y[sorted_idx]
    w = w[sorted_idx]
    sc_obj = ax.scatter(x, y, c=w, cmap='plasma', s=ms, edgecolors=None)

    return sc_obj


class Particles_vis():
    def __init__(self, ax):
        self.ax = ax
        self.handle = None

    def draw(self, st, w, ms=5):
        self.handle = draw_particles(self.ax, st, w, ms=ms)

    def remove(self):
        self.handle.remove()

    def re_draw(self, st, w, ms=5):
        self.remove()
        self.draw(st, w, ms=ms)


def update_zoomed_in_view(ax, x, y, delta):
    # changes axis
    ax.axis([x - delta, x + delta, y - delta, y + delta])
    pass


def connect_plots(ax0, ax1, x, y, delta):
    xy0 = (x + delta, y + delta)
    xy1 = (x - delta, y + delta)

    xy2 = (x + delta, y - delta)
    xy3 = (x - delta, y - delta)

    con0 = ConnectionPatch(xyA=xy0, xyB=xy1, coordsA="data", coordsB="data",
                           axesA=ax0, axesB=ax1, color="m")
    ax0.add_artist(con0)

    con1 = ConnectionPatch(xyA=xy2, xyB=xy3, coordsA="data", coordsB="data",
                           axesA=ax0, axesB=ax1, color="m")
    ax0.add_artist(con1)

    return con0, con1


class Frame():
    def __init__(self, ax):
        self.ax = ax
        self.handle = None

    def draw(self, x, y, delta):
        coords = np.array([[x - delta, y - delta],
                           [x - delta, y + delta],
                           [x + delta, y + delta],
                           [x + delta, y - delta]])

        closed_coords = \
            np.concatenate((coords, coords[0, :].reshape(1, 2)), axis=0)
        self.handle, = self.ax.plot(closed_coords[:, 0], closed_coords[:, 1], 'm')

    def remove(self):
        self.handle.remove()

    def re_draw(self, x, y, delta):
        self.remove()
        self.draw(x, y, delta)


class PlotState():
    def __init__(self, ax, opt):
        self.ax
        self.opt = opt
        self.handle = None

    def draw(self, x, y, ms=None):
        if ms is not None:
            self.handle, = self.ax.plot(x, y, self.opt, mfc='none')
        else:
            self.handle, = self.ax.plot(x, y, self.opt, mfc='none', ms=ms)

    def remove(self):
        self.handle.remove()

    def re_draw(self, x, y, ms=None):
        self.remove()
        self.draw(x, y, ms=ms)


def draw_box(ax, x, y, delta):

    coords = np.array([[x - delta, y - delta],
                       [x - delta, y + delta],
                       [x + delta, y + delta],
                       [x + delta, y - delta]])

    closed_coords = \
        np.concatenate((coords, coords[0, :].reshape(1, 2)), axis=0)
    handle, = ax.plot(closed_coords[:, 0], closed_coords[:, 1], 'r')

    return handle


def run_filter():

    my_map = Map()

    pd = prior_dist()
    motion = motion_model(0.1, 0.5 ** 2, 0.5 ** 2)  # dt, q_v_cov, q_d_phicov
#    meas = meas_model(0.9 ** 2, 0.03 ** 2, 0.1 ** 2, my_map.room, my_map.boundry)  # r_d_cov, r_v_cov, r_d_phicov

    meas = meas_model_torch(0.9 ** 2, 0.03 ** 2, 0.1 ** 2, my_map.room, my_map.boundry)  # r_d_cov, r_v_cov, r_d_phicov

    pf = ParticleFilter(pd, motion, meas, 10000)
    pf.init_state()

    st = pf.x
    w = pf.w

    x_k = []
    x_pf = []
    w_pf = []

    with open('data/robot_log.txt') as file:
        reader = csv.reader(file)

        for i, r in enumerate(reader):
            d = np.array(r, dtype=np.float32)
            if i == 0:
                x_k.append(d)
                x_pf.append(st)
                w_pf.append(w)
            else:
                z = np.array([d[-1], d[3], d[5]])  # r, v, dphi

                pf.predict()
                pf.update(z)

                st = pf.x
                w = pf.w

                x_k.append(d)
                x_pf.append(st)
                w_pf.append(w)

                pf.resample()
            print('done with %d' % i)

    return x_k, x_pf, w_pf


def run_filter_with_plots():
    fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]})

    (ax0, ax1) = ax
    fig.set_size_inches(20, 12)
    my_map = Map(ax0)
    my_map.draw_map()

    pd = prior_dist()
    motion = motion_model(0.1, 0.1 ** 2, 1 ** 2)  # dt, q_v_cov, q_d_phicov

    meas = meas_model(0.8 ** 2, 0.1 ** 2, 0.1 ** 2, my_map.room, my_map.boundry)  # r_d_cov, r_v_cov, r_d_phicov

    pf = ParticleFilter(pd, motion, meas, 1000)
    pf.init_state()

    st = pf.x
    w = pf.w

    fdelta = 0.5

    particles0 = Particles_vis(ax0)
    particles1 = Particles_vis(ax1)
    frame0 = Frame(ax0)
    frame1 = Frame(ax1)

    ax0.axis('equal')
#    ax0.axis('off')

    ax1.axis('square')
#    ax1.axis('off')

    plt.tight_layout(pad=0)
    with open('robot_log5.txt') as file:
        reader = csv.reader(file)
        for i, r in enumerate(reader):
            d = np.array(r, dtype=np.float32)

            if i == 0:
                lr0, = ax0.plot(d[1], d[2], 'mo', mfc='none')
                lr1, = ax1.plot(d[1], d[2], 'mo', mfc='none')
                frame0.draw(d[1], d[2], fdelta)
                frame1.draw(d[1], d[2], fdelta - 0.001)
                update_zoomed_in_view(ax1, d[1], d[2], fdelta)
                particles0.draw(st, w, ms=10)
                particles1.draw(st, w, ms=50)

                x_map = pf.get_MAP()
                x_mmse = pf.get_MMSE()

                lr_map0, = ax0.plot(x_map[0], x_map[1], 'rx', mfc='none')
                lr_mmse0, = ax0.plot(x_mmse[0], x_mmse[1], 'r+', mfc='none')

                lr_map1, = ax1.plot(x_map[0], x_map[1], 'rx', mfc='none', ms=30)
                lr_mmse1, = ax1.plot(x_mmse[0], x_mmse[1], 'r+', mfc='none', ms=30)

                con0, con1 = connect_plots(ax0, ax1, d[1], d[2], fdelta)

                ax1.legend([lr1, lr_map1, lr_mmse1],
                           ['True state', 'MAP estimate', 'MMSE estimate'],
                           loc='upper center', borderpad=2, labelspacing=2.5,
                           bbox_to_anchor=(0.5, 1.35))

#                ax1.axis('square')
                plt.pause(0.1)

                plt.savefig('pic7_%.5d.png' % i, dpi=300)

            else:
                z = np.array([d[-1], d[3], d[5]])  # r, v, dphi
                print(z)
                pf.predict()
                pf.update(z)

                st = pf.x
                w = pf.w

                x_map = pf.get_MAP()
                x_mmse = pf.get_MMSE()

                lr_map0.remove()
                lr_map0, = ax0.plot(x_map[0], x_map[1], 'rx', mfc='none')

                lr_map1.remove()
                lr_map1, = ax1.plot(x_map[0], x_map[1], 'rx', mfc='none', ms=30)

                lr_mmse0.remove()
                lr_mmse0, = ax0.plot(x_mmse[0], x_mmse[1], 'r+', mfc='none')

                lr_mmse1.remove()
                lr_mmse1, = ax1.plot(x_mmse[0], x_mmse[1], 'r+', mfc='none', ms=30)

                lr0.remove()
                lr0, = ax0.plot(d[1], d[2], 'mo', mfc='none')

                lr1.remove()
                lr1, = ax1.plot(d[1], d[2], 'mo', mfc='none')

                frame0.remove()
                frame0.draw(d[1], d[2], fdelta)

                frame1.remove()
                frame1.draw(d[1], d[2], fdelta)

                update_zoomed_in_view(ax1, d[1], d[2], fdelta)

                particles0.remove()
                particles0.draw(st, w, ms=10)

                particles1.remove()
                particles1.draw(st, w, ms=50)

                con0.remove()
                con1.remove()

                con0, con1 = connect_plots(ax0, ax1, d[1], d[2], fdelta)
                plt.savefig('pic7_%.5d.png' % i, dpi=300)

#                plt.pause(2)

                pf.resample()
                print('\n\tdone with %d\n' % i)


if __name__ == '__main__':
    run_filter()
