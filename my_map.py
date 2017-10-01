import time
import numpy as np
import matplotlib.pyplot as plt

# ==================================================================================================
# Helper functions
# ==================================================================================================


def draw_rectangle(spec, axes, fill_block=True):
    '''Draws a rectangle according to the tuple spec - (x, y, dx, dy)'''
    (x, y, dx, dy) = spec
    coords = np.array([[x, y],
                       [x + dx, y],
                       [x + dx, y + dy],
                       [x, y + dy]])
    if fill_block:
        axes.fill(coords[:, 0], coords[:, 1], c='k')
    else:
        closed_coords = \
            np.concatenate((coords, coords[0, :].reshape(1, 2)), axis=0)
        axes.plot(closed_coords[:, 0], closed_coords[:, 1], 'k')


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


# Move this method into Map class or seperate it out
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

    cond1 = np.bitwise_and(0 < AM_AB, AM_AB < AB_AB)  # NxM
    cond2 = np.bitwise_and(0 < AM_AD, AM_AD < AD_AD)  # NxM

    # valid point logic:
    final_cond = np.bitwise_and(cond1, cond2)
    # this slice represents the room coordiantes, which is why negation is used,
    # beacuse we actually want to be in the room
    final_cond[:, 0] = np.bitwise_not(final_cond[:, 0])
    res = np.logical_not(np.sum(final_cond, axis=1))

    return res


# ==================================================================================================
# Map class
# ==================================================================================================


class Map():
    '''
    Represents the map, holds all rectangles representing the boundry and
    has methods for checking whether a point is in the valid region of the map
    '''
    def __init__(self, axes=None):
        #                x  y  dx dy
        self.boundry = [(3, 0, 1, 4),
                        (7, 0, 3, 3),
                        (5, 5, 5, 1),
                        (2, 8, 5, 2),
                        (1, 5, 2, 2)]

        self.room = (0, 0, 10, 10)

        self.ax = axes

    def draw_map(self):
        '''Draws a map containing all rectangles'''
        if self.ax is None:
            (_, self.ax) = plt.subplots()

        draw_rectangle(self.room, self.ax, fill_block=False)
        for b in self.boundry:
            draw_rectangle(b, self.ax)

    def valid_point(self, p):
        ''' Returns bool for a set of points whether they are valid or not in the map,
            i.e. not outside of the room or inside of a solid object. '''
        res = is_in_rect(p, self.room)
        for b in self.boundry:
            res = np.bitwise_and(res, np.bitwise_not(is_in_rect(p, b)))

        return res

    def valid_point_broad(self, p):
        # rects - Mx4, p - Nx2
        rects = np.array([self.room] + self.boundry)

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

        cond1 = np.bitwise_and(0 < AM_AB, AM_AB < AB_AB)  # NxM
        cond2 = np.bitwise_and(0 < AM_AD, AM_AD < AD_AD)  # NxM

        # logic related to seeing if a point is valid
        final_cond = np.bitwise_and(cond1, cond2)
        # this slice represents the room coordiantes, which is why negation is used,
        # beacuse we actually want to be in the room
        final_cond[:, 0] = np.bitwise_not(final_cond[:, 0])
        res = np.logical_not(np.sum(final_cond, axis=1))

        return res


if __name__ == '__main__':
    my_map = Map(None)
    points = np.array([[1, 1], [2, 2], [3, 3], [3.5, 3]])
    points = np.random.normal(size=(1000000, 2))

    print('normal version')
    start = time.time()
    r = my_map.valid_point(points)
#    print(r)
    print('time: ', time.time() - start)

    print('')

    print('broad version')
    start = time.time()
    x = my_map.valid_point_broad(points)
    print('time: ', time.time() - start)
#    print(x)
