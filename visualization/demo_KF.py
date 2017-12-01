import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from target import Target
from kalman_filter import KF


def generate_tracks_meas(A, H, Q, R, x_0, P_0, K=100):
    target = Target(A, H, Q, R, x_0, P_0)

    xr = x_0.size
    (yr, _) = H.shape
    X = np.zeros((xr, K))
    Z = np.zeros((yr, K))

    for k in range(K):
        x = target.prop_state()
        z = target.prop_meas()

        X[:, k] = x.flatten()
        Z[:, k] = z.flatten()

    return X, Z


if __name__ == '__main__':
    np.random.seed(1337)
    T = 1.0
    A = np.array([[1, 0, T, 0],
                  [0, 1, 0, T],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])

    sq1 = 0.8**2
    sq2 = 0.1**2

    cov_xx = sq1 * 0.33 * (T**3)
    cov_xdx = sq1 * 0.5 * (T**2)
    cov_yy = sq2 * 0.33 * (T**3)
    cov_ydy = sq2 * 0.5 * (T**2)
    cov_dxdx = sq1 * T
    cov_dydy = sq2 * T

    Q = np.array([[cov_xx,    0.0, cov_xdx,     0.0],
                  [0.0, cov_yy,     0.0, cov_ydy],
                  [cov_xdx,   0.0, cov_dxdx,    0.0],
                  [0.0, cov_ydy,      0.0, cov_dydy]])
    rq = 0.2**2
    R = rq * np.eye(2)
    #               x   x' y  y'
    x_0 = np.array([0, 3, 0, 3]).reshape(4, 1)
    #                     x    y    x'   y'
    P_0 = np.array([[1.0, 0.0, 0.5, 0.0],  # x
                    [0.0, 1.0, 0.0, 0.5],  # y
                    [0.5, 0.0, 0.9, 0.0],  # x'
                    [0.0, 0.5, 0.0, 0.9]])  # y'

    K = 4
    X, Z = generate_tracks_meas(A, H, Q, R, x_0, P_0, K=K)

    X_pred = np.zeros((4, K))
    P_pred = np.zeros((4, 4, K))

    X_post = np.zeros((4, K))
    P_post = np.zeros((4, 4, K))

    kf = KF(A, H, Q, R, x_0, P_0)

    for k in range(K):
        kf.predict()
        (x, P) = kf.get_state()
        X_pred[:, k] = x.flatten()
        P_pred[:, :, k] = P

        kf.update(Z[:, k].reshape(2, 1))
        (x, P) = kf.get_state()
        X_post[:, k] = x.flatten()
        P_post[:, :, k] = P

    pos_real = np.vstack((X[0, :], X[2, :]))
    diff_1 = pos_real - Z
    pos_pred = np.vstack((X_pred[0, :], X_pred[2, :]))
    diff_2 = pos_real - pos_pred
    pos_post = np.vstack((X_post[0, :], X_post[2, :]))
    diff_3 = pos_real - pos_post

    print(np.sqrt(np.mean(np.square(diff_1[0, :]))))
    print(np.sqrt(np.mean(np.square(diff_2[0, :]))))
    print(np.sqrt(np.mean(np.square(diff_3[0, :]))))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.plot([x_0[0], X[0, 0]], [x_0[2], X[2, 0]], '--bo', markerfacecolor=(1, 1, 1))
    plt.plot(X[0, :], X[2, :], '--bo', label='True state')
    plt.plot(Z[0, :], Z[1, :], 'rx', label='Observed measurement')
    plt.plot(X_post[0, :], X_post[2, :], 'ko', label='KF inferred state')
    plt.plot([-0.25], [1.75], '.', color=(1, 1, 1))
    plt.plot([3.5], [-0.25], '.', color=(1, 1, 1))
    plt.axis([-.25, 5, -.25, 2])
    plt.axis('scaled')
    plt.xlabel('X')
    plt.ylabel('Y')
#    plt.show()

    x_min = -2.5
    x_max = 8
    y_min = -0.55
    y_max = 2.00
    dx = 0.2
    dy = 0.2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, dx),
                         np.arange(y_min, y_max, dy))
    ax = plt.gca()
    ax.plot_surface(xx, yy, -0.1, color=(0, 0, 1, 0.1), shade=False)
    ax.scatter(0, 0, 1)
    plt.axis('off')
    plt.show()
