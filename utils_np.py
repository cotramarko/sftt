import numpy as np


def mvn_pdf(x, mu, sigma):
    d = (x - mu)[:, None, :]  # Nx1x2 column vector
    dt = (x - mu)[:, :, None]  # Nx2x1 row vector

    a = 1 / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(sigma))
    b = -0.5 * np.sum(dt * np.linalg.inv(sigma) * d, axis=(2, 1))

    pdf = a * np.exp(b)
    return pdf


def normal_pdf(x, mu, cov):
    a = 1 / np.sqrt(2 * np.pi * cov)
    b = np.exp(-np.square(x - mu) / (2 * cov))

    return a * b


def is_in_rect(p, rect):
    '''Given a point (or set of points) returns True if in rectangle rect'''
    A = np.array([rect[0], rect[1]])
    B = np.array([rect[0], rect[1] + rect[3]])
    D = np.array([rect[0] + rect[2], rect[1]])

    M = p
    AM = M - A
    AB = B - A
    AD = D - A
    AM_AB = np.dot(AM, AB)
    AB_AB = np.dot(AB, AB)

    AM_AD = np.dot(AM, AD)
    AD_AD = np.dot(AD, AD)

    cond1 = np.bitwise_and(0 < AM_AB, AM_AB < AB_AB)
    cond2 = np.bitwise_and(0 < AM_AD, AM_AD < AD_AD)

    return np.bitwise_and(cond1, cond2)


def valid_point(room, boundry, p):
    ''' Returns bool for a set of points whether they are valid or not in the map,
        i.e. not outside of the room or inside of a solid object. '''
    res = is_in_rect(p, room)
    for b in boundry:
        res = np.bitwise_and(res, np.bitwise_not(is_in_rect(p, b)))

    return res


def valid_point_broad(p, rects):
    # rects - Mx4, p - Nx2
    rects = np.array(rects)

    # preallocate some parts of this method
    A = rects[:, :2]  # Mx2
    B = np.hstack((rects[:, 0, None], rects[:, 1, None] + rects[:, 3, None]))  # Mx2
    D = np.hstack((rects[:, 0, None] + rects[:, 2, None], rects[:, 1, None]))  # Mx2

    M = p  # Nx2

    AM = M[:, None, :] - A[None, ...]  # NxMx2 points-by-boundries-by-xy
    AB = B - A  # Mx2 boundries-by-xy
    AD = D - A  # Mx2 boundries-by-xy

    AM_AB = np.sum(AM * AB, axis=-1)  # NxM
    AM_AD = np.sum(AM * AD, axis=-1)  # NxM

    AB_AB = np.sum(AB * AB, axis=-1)  # NxM
    AD_AD = np.sum(AD * AD, axis=-1)  # NxM

    cond1 = (0 < AM_AB) & (AM_AB < AB_AB)  # NxM
    cond2 = (0 < AM_AD) & (AM_AD < AD_AD)  # NxM

    # valid point logic:
    final_cond = cond1 & cond2
    # this slice represents the room coordiantes, which is why negation is used,
    # beacuse we actually want to be in the room
    final_cond[:, 0] = (~final_cond[:, 0])
    res = np.logical_not(np.sum(final_cond, axis=1))

    return res


def distance_to_object(x, y, phi, room, boundry):
    ''' Computes the distance from (x,y) with heading phi towards the nearest
    object found in the map. The map is defined by the room and the boundry (list of squares) '''
    base_point = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1))).reshape(1, -1, 2)
    d_vec = np.hstack((np.cos(phi).reshape(-1, 1), np.sin(phi).reshape(-1, 1))).reshape(1, -1, 2)

    dists = np.arange(0, 15, 0.01).reshape(-1, 1, 1)

    ray = base_point + dists * d_vec  # DxNx2
    (D, N, _) = ray.shape
    all_points = ray.reshape(-1, 2)

    idx = np.bitwise_not(valid_point(room, boundry, all_points))
    idx = idx.reshape(D, N)

    invalid_dists = dists.reshape(-1, 1) * idx
    invalid_dists = invalid_dists.flatten()
    invalid_dists[invalid_dists == 0] = np.nan
    invalid_dists = invalid_dists.reshape(D, N)

    z = np.nanmin(invalid_dists, axis=0)  # Nx2
    idx_z = np.nanargmin(invalid_dists, axis=0)

    ray_hits = ray[idx_z, np.arange(N), :]  # Nx2
    ray = ray.transpose(1, 0, 2)  # NxDx2

    return ray, ray_hits, z
