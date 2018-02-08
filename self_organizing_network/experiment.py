import numpy as np
import matplotlib.pyplot as plt


def cross_distance(x):
    (N, _) = x.shape
    delta = x[None, ...] - x[:, None, :]
    z = np.linalg.norm(delta.reshape(-1, 2), ord=2, axis=1)
    return z.reshape(N, N)


def in_range(z, range=5):
    z = z <= range
    return z


r0 = np.array([0, 0])
r1 = np.array([2, 5])

N = 5

x = np.random.uniform(size=(N, 2)) * 10
x = np.array([[0, 0], [0, 1], [1, 1]])

cd = cross_distance(x)
print(cd)
print(in_range(cd))
