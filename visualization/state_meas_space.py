import numpy as np
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
        return x_k

    def prop_meas(self):
        z = np.mat(H) * np.mat(self.x_k) + \
            

x_0 = np.array([[1],
                [1]])
P_0 = np.eye(2) * 0.2
A = np.eye(2)
H = np.eye(2)

Q = np.eye(2) * 2
R = np.eye(2) * 0.2

target = Target(A, H, Q, R, x_0, P_0)
target.prop_state()

raise


def generate_tracks_meas(A, H, Q, R, K=100):

    pass


def draw_planes():

    wx = 5
    ly = 12

    xx, yy = np.meshgrid(np.arange(0, wx, wx - 1), np.arange(0, ly, ly - 1))

    fig = plt.figure()
    alpha = 0.3
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, 0.5, color=(1, 0, 0, alpha))
    ax.plot_surface(xx, yy, 0.1, color=(0, 0, 1, alpha))

    ax.view_init(azim=-27, elev=30)
    ax.set_xlim3d(-3, 3)
    ax.set_ylim3d(3, 10)
    ax.set_zlim3d(0, 0.5)
    ax.plot([0, 1], [1, 5])

#    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()


if __name__ == '__main__':
    draw_planes()
