import numpy as np
import matplotlib.pyplot as plt
from utils_np import distance_to_object


def distance_to_wall(x, y, phi, rects):
    q = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
    s = q + 15 * np.hstack((np.cos(phi.reshape(-1, 1)), np.sin(phi.reshape(-1, 1))))

    rects = np.array(rects)  # Mx4
    L = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])  # 4x2

    # pr holds all cornerpoints of the walls in the map, size Mx4x2
    pr = rects[:, None, :2] + L[None, ...] * rects[:, None, 2:]

    p = pr[:, :2, :]  # Mx2x2
    r = pr[:, 2:, :]  # Mx2x2

    diff = q[:, None, None, :] - p[None, ...]

    # diff - NxMx2x2 --> NxMx2x1x2
    # r - Mx2x2 --> 1xMx1x2x2
    # corresponds to (q-p)xr
    top = np.cross(diff[:, :, :, None, :], r[None, :, None, :, :])
#    top = np.sum(diff[:, :, :, None, :] * r[None, :, None, :, :] * [1, -1], axis=-1)

    bot = np.cross(r[None, ...], s[:, None, None, :])
#    bot = np.sum(r[None, ...] * s[:, None, None, :] * [1, -1], axis=-1)

    u = top / bot[:, :, None, :]  # NxMx1x2
    (N, M, _, _) = u.shape
    u = u.reshape(N, M, -1)

    print('top:')
    print(top)
    print('bot:')
    print(bot)

    print(u)

    #  q - Nx2 ---> Nx1x1x2
    #  u - NxMx4 ---> NxMx4x1
    #  s - Nx2 ---> Nx1x1x2
    # hits - NxMx4x2
    hits = q[:, None, None, :] + u[..., None] * s[:, None, None, :]
    print(hits.shape)
    print(hits)


if __name__ == '__main__':
    y = np.array([1])
    x = np.array([1])
    phi = np.ones(1) * np.pi / 2
    #           x  y  dx dy
#    boundry = [(3, 0, 1, 4),
#               (7, 0, 3, 3),
#               (5, 5, 5, 1),
#               (2, 8, 5, 2),
#               (1, 5, 2, 2)]

    boundry = [(9, 9, 1, 1)]
    room = (0, 0, 10, 10)

    distance_to_wall(x, y, phi, [room] + boundry)

    ray, ray_hits, z = distance_to_object(x, y, phi, room, boundry)
    print('ray hits:')
    print(ray_hits)

