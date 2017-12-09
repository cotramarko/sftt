import numpy as np
import matplotlib.pyplot as plt

from kalman_filter import KalmanFilter
from simulate_model import simulate_system, create_model_parameters

np.random.seed(21)
(A, H, Q, R) = create_model_parameters()
K = 100000
# initial state
x = np.array([0, 0.1, 0, 0.1])
P = 0 * np.eye(4)

(state, meas) = simulate_system(K, x)
kalman_filter = KalmanFilter(A, H, Q, R, x, P)

est_state = np.zeros((K, 4))
est_cov = np.zeros((K, 4, 4))

for k in range(K):
    kalman_filter.predict()
    kalman_filter.update(meas[k, :])
    (x, P) = kalman_filter.get_state()

    est_state[k, :] = x
    est_cov[k, ...] = P

    if (k % 1000 == 0):
        print(k)

'''
plt.figure(figsize=(7, 5))
plt.plot(state[:, 0], state[:, 2], '-bo')
plt.plot(est_state[:, 0], est_state[:, 2], '-ko')
plt.plot(meas[:, 0], meas[:, 1], ':rx')

plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend(['true state', 'observed measurement'])
plt.legend(['true state', 'inferred state', 'observed measurement'])
plt.axis('square')
plt.tight_layout(pad=0)
plt.show()
'''
idx = [True, False, True, False]
kf_pos_delta = state[:, idx] - est_state[:, idx]
meas_pos_delta = state[:, idx] - meas

kf_pos_dist = np.sqrt(np.sum(np.square(kf_pos_delta), axis=1))
meas_pos_dist = np.sqrt(np.sum(np.square(meas_pos_delta), axis=1))

rms_kf_dist = np.sqrt(np.mean(np.square(kf_pos_dist)))
rms_meas_dist = np.sqrt(np.mean(np.square(meas_pos_dist)))

print('KF RMS')
print(rms_kf_dist)
print('Measurement RMS')
print(rms_meas_dist)
