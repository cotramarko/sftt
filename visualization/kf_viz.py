import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def draw_stcov(ax, st, cov, col='k', zz=0):
    phi = np.linspace(0, 2 * np.pi, 100)
    sig3 = st[:, None] + np.dot(np.linalg.cholesky(cov), np.array([np.cos(phi), np.sin(phi)]))
    ph0, = ax.plot(sig3[0, :], sig3[1, :], zz * np.ones(sig3[0, :].shape), col)
    ph1, = ax.plot(st[0, None], st[1, None], zz, col + '.')
    return ph0, ph1


x0 = np.array([1, -2])
P0 = np.array([[1, 0.0], [0.0, 1]])

wx, ly = (5, 8)
zl, zu = (0.0, 1.0)

x_range = np.linspace(-0.5 * wx, 0.5 * wx, 10)
y_range = np.linspace(-0.5 * ly, 0.5 * ly, 10)

xx, yy = np.meshgrid(x_range, y_range)

fig = plt.figure(figsize=(8, 8))
alpha = 0.25
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx, yy, zu * np.ones(xx.shape), color=(1, 0, 0, alpha))
ax.plot_surface(xx, yy, zl * np.ones(xx.shape), color=(0, 0, 1, alpha))

ax.view_init(azim=-25, elev=20)
bds = 3.00
ax.set_xlim3d(-bds, bds)
ax.set_ylim3d(-bds, bds)
ax.set_zlim3d(0, 3)

ax.text(2.5, 3.5, zl, r'$\mathcal{X}$', fontsize=20)
ax.text(2.5, 3.5, zu, r'$\mathcal{Y}$', fontsize=20)

plt.axis('off')
plt.tight_layout(pad=0)
###
handles_0 = []

(ph0, ph1) = draw_stcov(ax, x0, P0)
th0 = ax.text(1, -2, zl - 0.1, r'$\hat{x}_{k-1|k-1}$', fontsize=10, ha='center')
th1 = ax.text(2, -3, zl, r'$P_{k-1|k-1}$', fontsize=10, ha='center')

handles_0.extend([ph0, ph1, th0, th1])

fig_nbr = 0

for _ in range(82):
    plt.savefig('pic_%06d.png' % fig_nbr, dpi=300)
    fig_nbr += 1
# plt.pause(2)
###
handles_1 = []

x01 = np.array([0, 2])
P01 = np.array([[1.5, 0.0], [0.0, 1.5]])

ph, = ax.plot([1, 0], [-2, 2], 'k--', linewidth=0.5)
(ph0, ph1) = draw_stcov(ax, x01, P01, col='b')
th0 = ax.text(0, 2, zl - 0.1, r'$\hat{x}_{k|k-1}$', fontsize=10, ha='center', color='b')
th1 = ax.text(1.5, 1, zl, r'$P_{k|k-1}$', fontsize=10, ha='center', color='b')
th2 = ax.text(0.5, 0, zl + 0.05, r'$A$', fontsize=16, ha='center', color='k')

handles_1.extend([ph, ph0, ph1, th0, th1, th2])

for _ in range(82):
    plt.savefig('pic_%06d.png' % fig_nbr, dpi=300)
    fig_nbr += 1
# plt.pause(2)
###
handles_2 = []

S = np.array([[2.5, 0.0], [0.0, 2.5]])
(ph0, ph1) = draw_stcov(ax, x01, S, col='r', zz=zu)
ph2, = ax.plot([0, 0], [2, 2], [zl, zu], 'k--', linewidth=0.5)

th0 = ax.text(0, 2, zu + 0.1, r'$\hat{y}_{k|k-1}$', fontsize=10, ha='center', color='r')
th1 = ax.text(1, 0.25, zu, r'$S_{k|k-1}$', fontsize=10, ha='center', color='r')
th2 = ax.text(0, 2.2, zl + zu / 2, r'$H$', fontsize=16, ha='center', color='k')

ph3, = ax.plot([1], [2.5], [zu], 'gv', ms=3)
ph4, = ax.plot([1], [2.5], [zu], 'gx')
ph5, = ax.plot([1, 1], [2.5, 2.5], [zu, zu + 1], 'g-', linewidth=0.5)
th3 = ax.text(1, 2.75, zu + 0.75, r'$y_k$', fontsize=10, ha='center', color='g')

handles_2.extend([ph0, ph1, ph2, th0, th1, th2, ph3, ph4, ph5, th3])

for _ in range(82):
    plt.savefig('pic_%06d.png' % fig_nbr, dpi=300)
    fig_nbr += 1
# plt.pause(2)
###
x1 = np.array([0.6, 2.37])
P1 = np.array([[1, 0.0], [0.0, 1]])

handles_3 = []
(ph0, ph1) = draw_stcov(ax, x1, P1)
th0 = ax.text(0.6, 2.37, zl - 0.1, r'$\hat{x}_{k|k}$', fontsize=10, ha='center', color='k')
th1 = ax.text(2, 1.5, zl, r'$P_{k|k}$', fontsize=10, ha='center', color='k')
handles_3.extend([ph0, ph1, th0, th1])

# plt.axis('square')
plt.axis('off')
plt.tight_layout(pad=0)

res = 0.025
alpha_phase = np.arange(0, 1 + res, res)
freq = alpha_phase.size
print(freq)
for (alpha_inc, alpha_dec) in zip(alpha_phase, reversed(alpha_phase)):
    #    for h in handles_0:
    #        h.set_alpha(alpha_dec)
    for h in handles_1:
        h.set_alpha(alpha_dec)
    for h in handles_2:
        h.set_alpha(alpha_dec)
    for h in handles_3:
        h.set_alpha(alpha_inc)

    plt.savefig('pic_%06d.png' % fig_nbr, dpi=300)
    fig_nbr += 1

for _ in range(82):
    plt.savefig('pic_%06d.png' % fig_nbr, dpi=300)
    fig_nbr += 1
# plt.pause(2)
