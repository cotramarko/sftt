import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from bootstrap_PF import ParticleFilter, normal_pdf
from my_map import Map
from robot import Robot, RobotIllustrator, distance_to_object
from filter_params import prior_dist, motion_model, meas_model


def draw_particles(ax, st, w):
    x = st[:, 0]
    y = st[:, 1]
    phi = st[:, 3]
    r = 0.03
    lx = np.array([x, x + r * np.cos(phi)]).reshape(2, -1)
    ly = np.array([y, y + r * np.sin(phi)]).reshape(2, -1)

    sorted_idx = w.argsort()
    x = x[sorted_idx]
    y = y[sorted_idx]
    w = w[sorted_idx]
    # TODO: the points going into scatter should be sorted according to w, 
    # since then the most likely points can be plotted last (i.e. more visible)
    sc_obj = ax.scatter(x, y, c=w, cmap='plasma', s=1, edgecolors=None)
    pl_obj = ax.plot(lx, ly, color='r', linewidth=0.3, alpha=0.2)

    return sc_obj, pl_obj


def run_filter():

    my_map = Map()
    my_map.draw_map()

    pd = prior_dist()
    motion = motion_model(0.1, 0.5 ** 2, 0.5 ** 2)  # dt, q_v_cov, q_d_phicov
    meas = meas_model(0.9 ** 2, 0.03 ** 2, 0.1 ** 2, my_map)  # r_d_cov, r_v_cov, r_d_phicov

    pf = ParticleFilter(pd, motion, meas, 1000)
    pf.init_state()

    st = pf.x
    w = pf.w

    x_k = []
    x_pf = []
    w_pf = []

    with open('robot_log.txt') as file:
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
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)
    my_map = Map(ax)
    my_map.draw_map()

    pd = prior_dist()
    motion = motion_model(0.1, 0.1 ** 2, 1 ** 2)  # dt, q_v_cov, q_d_phicov

    meas = meas_model(1.2 ** 2, 0.1 ** 2, 0.1 ** 2, my_map)  # r_d_cov, r_v_cov, r_d_phicov

    pf = ParticleFilter(pd, motion, meas, 5000)
    pf.init_state()

    st = pf.x
    w = pf.w

    lo1, lo2 = draw_particles(ax, st, w)
    for l in lo2:
        l.remove()

    plt.savefig('pic1_%.5d.png' % 0, dpi=300)
    with open('robot_log.txt') as file:
        reader = csv.reader(file)
        lr, = ax.plot(0, 0, 'r.')
        for i, r in enumerate(reader):
            if i > 1:
                d = np.array(r, dtype=np.float32)

                z = np.array([d[-1], d[3], d[5]])  # r, v, dphi
                print(z)
                pf.predict()
                pf.update(z)

                st = pf.x
                w = pf.w

                lr.remove()
                lr, = ax.plot(d[1], d[2], 'bo', mfc='none')
                lo1.remove()
                lo1, lo2 = draw_particles(ax, st, w)
                for l in lo2:
                    l.remove()

                plt.savefig('pic1_%.5d.png' % i, dpi=300)

                #plt.pause(0.1)

                pf.resample()
                print('done with %d' % i)


if __name__ == '__main__':
    run_filter_with_plots()
