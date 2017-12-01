import numpy as np
import matplotlib.pyplot as plt
from phd_tracker import GaussComp, PhdTracker


def create_track(A, H, Q, R, x0, N):
    (d, _) = Q.shape
    (g, _) = R.shape

    state = np.zeros((1, d))
    meas = np.zeros((1, g))

    x = x0
    for _ in range(N):
        x = np.matmul(A, x) + np.random.multivariate_normal(mean=np.zeros(d), cov=Q)
        y = np.matmul(H, x) + np.random.multivariate_normal(mean=np.zeros(g), cov=R)

        state = np.vstack((state, x[None, :]))
        meas = np.vstack((meas, y[None, :]))

    return state[1:], meas[1:]


if __name__ == '__main__':
    T = 1
    A = np.array([[1, 0, T, 0],  # px
                  [0, 1, 0, T],  # py
                  [0, 0, 1, 0],  # vx
                  [0, 0, 0, 1]])  # vy

    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])

    q_cov = 0.5 ** 2
    q00 = np.diag((0.33 * T**3) * np.ones(2))
    q01 = np.diag((0.50 * T**2) * np.ones(2))
    q11 = np.diag(T * np.ones(2))

    q0 = np.hstack((q00, q01))
    q1 = np.hstack((q01, q11))
    Q = q_cov * np.vstack((q0, q1))

    r_cov = 0.125 ** 2
    R = r_cov * np.diag([1, 1])

    x0 = np.array([0, 0, 1, 1])
    (state0, meas0) = create_track(A, H, Q, R, x0, 10)
    (state1, meas1) = create_track(A, H, Q, R, x0, 10)
    (state2, meas2) = create_track(A, H, Q, R, x0, 10)

    plt.figure()
    plt.plot(state0[:, 0], state0[:, 1], '-o')
    plt.plot(meas0[:, 0], meas0[:, 1], 'rx')

    plt.plot(state1[:, 0], state1[:, 1], '-o')
    plt.plot(meas1[:, 0], meas1[:, 1], 'rx')

    plt.plot(state2[:, 0], state2[:, 1], '-o')
    plt.plot(meas2[:, 0], meas2[:, 1], 'rx')
#    plt.show()

    meas_scan = np.stack((meas0, meas1, meas2)).transpose(1, 0, 2)

    #
    # Test the phd_filter
    #

    gauss_comp = GaussComp(A, H, R, Q)

    tracker = PhdTracker(max_comps=10,
                         Ps=1,
                         Pd=0.8,
                         x_birth=np.vstack((x0[None, ...], x0[None, ...])),
                         P_birth=np.vstack((Q[None, ...], Q[None, ...])),
                         K_rate=0,
                         gauss_comp=gauss_comp)

    for j in range(10):
        z = meas_scan[j, ...]
        tracker.predict()
        if j == 6:
            tracker.update(z[1:, ...])
        else:
            tracker.update(z)
        tracker.prune()
        tracker.merge()
        (x, P, track_ids) = tracker.get_tracks(0.5)
        print(np.sort(track_ids.flatten()))
        print('time step: %d, nbr of targets: %d' % (j, x.shape[0]))
        plt.plot(x[:, 0], x[:, 1], 'mo')

        plt.pause(0.5)

plt.show()
