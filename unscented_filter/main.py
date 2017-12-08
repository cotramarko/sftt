import csv
import numpy as np
import matplotlib.pyplot as plt

from ukf import UKF


def create_filter(x, P):

    def fx(x, T):
        # x.shape = (N, k, 1)
        # x = [xp, yp, v, phi, dphi]
        xp = x[:, 0, :] + T * x[:, 2, :] * np.cos(x[:, 3, :])
        yp = x[:, 1, :] + T * x[:, 2, :] * np.sin(x[:, 3, :])
        v = x[:, 2, :]

        phi = x[:, 3, :] + T * x[:, 4, :]
        dphi = x[:, 4, :]

        x = np.concatenate((xp, yp, v, phi, dphi), axis=1)
        return x[..., None]

    q_v = 1 ** 2
    q_dphi = 1 ** 2
    gamma = np.array([[0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1]])
    Q = gamma.transpose() @ np.diag([q_v, q_dphi]) @ gamma

    def hx(x):
        # x.shape = (N, k, 1)
        return x[:, 0:-1, :]

    r_x = 3 ** 2
    r_y = 3 ** 2
    r_v = 3 ** 2
    r_phi = 2 ** 2
    R = np.diag([r_x, r_y, r_v, r_phi])

    ukf = UKF(fx, hx, Q, R, x, P)
    return ukf


def csv2si(row):
    row = row * np.array([1 / 1000, 1 / 100, 1 / 100, 1 / 100, 1, np.pi / 180])
    return row


plt.figure()

data = []

with open('car.log') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=';')
    for i, row in enumerate(spamreader):
        r = csv2si(np.array(row[0:-1], dtype=float))
        data.append(r)
        if i > 10000:
            break


data = np.array(data)

match_idx = [False, True, True, False, True, True]
x0 = data[0, match_idx]
x0 = np.concatenate((x0, [0]))

ukf = create_filter(x0, np.eye(5))
time_diff = np.diff(data[:, 0], axis=0)
# time_diff = np.concatenate((time_diff, [0]))

st = []
for j in range(1000):
    ukf.predict(time_diff[j])
    ukf.update(data[j + 1, match_idx])
    x, _ = ukf.get_state()
    st.append(x)


st = np.array(st)
print(st.shape)
plt.plot(st[:, 0, :], st[:, 1, :], '-bo')
plt.plot(data[1:1000, 1], data[1:1000, 2], 'rx')
plt.show()
