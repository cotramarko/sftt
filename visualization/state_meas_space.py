import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Target():
    def __init__(self, A, H, Q, R, x_0, P_0):
        self.A = A
        self.H = H
        self.Q = Q
        self.R = R

        self.x_k = x_0

    def prop_state(self):
        self.x_k = np.mat(self.A) * np.mat(self.x_k) + \
            np.random.multivariate_normal(mean=np.array([0, 0]),
                                          cov=self.Q).reshape(2, 1)
        return self.x_k

    def prop_meas(self):
        z_k = np.mat(self.H) * np.mat(self.x_k) + \
            np.random.multivariate_normal(mean=np.array([0, 0]),
                                          cov=self.R).reshape(2, 1)
        return z_k


def circle(res=100, r=1, perturb=False):
    if perturb:
        R = r * (1 + np.cumsum(np.random.normal(scale=0.25, size=res)))
    else:
        R = r

    phi = np.linspace(0, 2 * np.pi, res)
    x = R * np.cos(phi)
    y = R * np.sin(phi)

    x = np.append(x, x[0])
    y = np.append(y, y[0])
    return x, y


def generate_tracks_meas(A, H, Q, R, x_0, P_0, K=100):
    target = Target(A, H, Q, R, x_0, P_0)

    X = np.zeros(shape=(2, K))
    Z = np.zeros(shape=(2, K))

    for time_step in range(K):
        x = target.prop_state()
        z = target.prop_meas()

        X[:, time_step] = x.reshape((2,))
        Z[:, time_step] = z.reshape((2,))

    return X, Z


def draw_planes():

    wx = 5
    ly = 12
    zl = 0.1
    zu = 0.4

    xx, yy = np.meshgrid(np.arange(0, wx, wx - 1), np.arange(0, ly, ly - 1))

    fig = plt.figure()
    alpha = 0.4
    q = 0.0
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, zu, color=(1, 0, 0, alpha))
    ax.plot_surface(xx, yy, zl, color=(0, 0, 1, alpha))

    ax.view_init(azim=-23, elev=30)
    ax.set_xlim3d(-3, 3)
    ax.set_ylim3d(3, 11)
    ax.set_zlim3d(0, 0.5)

    X = [2.0, 2.5, 1.8, 2.9]
    Y = [1.0, 5.0, 7.0, 10.0]
    cords = np.array([X, Y])

    ax.plot(cords[0, :], cords[1, :], zl, '-b.', linewidth=0.4)
    ax.plot(cords[0, :], cords[1, :], zu, '.', c=(q, q, q, 0.5), linewidth=0.4)
    mvn = np.random.multivariate_normal
#    r = mvn(mean=[0, 0], cov=0.1 * np.eye(2), size=(4)).transpose()
#    ax.plot(cords[0, :] + r[0, :], cords[1, :] + r[1, :], zu, 'r.', ms=4)

    xr, yr = circle(res=20, r=0.5)

    for j in range(4):
        x = cords[0, j]
        y = cords[1, j]
        ax.plot([x, x], [y, y], [zl, zu], ':', c=(q, q, q), linewidth=0.4)
        ax.plot(xr + x, yr + y, zu, 'r')

    ax.text(4.5, 11, zl, r'$\mathcal{X}$', fontsize=20)
    ax.text(4.5, 11, zu, r'$\mathcal{Z}$', fontsize=20)

    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig('/Users/markocotra/Desktop/imgs.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    x_0 = np.array([[1],
                    [1]])
    P_0 = np.eye(2) * 0.2
    A = np.eye(2)
    H = np.eye(2)

    Q = np.eye(2) * 2
    R = np.eye(2) * 0.02
    K = 6
    (X, Z) = generate_tracks_meas(A, H, Q, R, x_0, P_0, K)

    draw_planes()
