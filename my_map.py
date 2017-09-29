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


def is_in_rect_brad(p, rects):
    # rects - Mx4
    rects = np.array(rects)

    A = rects[:, :2]  # Mx2
    B = np.hstack((rects[:, 0, None], rects[:, 1, None] + rects[:, 3, None]))
    D = np.hstack((rects[:, 0, None] + rects[:, 2, None], rects[:, 1, None]))

    M = p  # Nx2

    AM = M[:, None, :] - A[None, ...]  # NxMx2 points-by-boundries-by-xy
    AB = B - A  # Mx2 boundries-by-xy
    AD = D - A  # Mx2 boundries-by-xy

    AM_AB = np.sum(AM * AB, axis=-1) # NxM
    AM_AD = np.sum(AM * AD, axis=-1) # NxM

    AB_AB = np.sum(AB * AB, axis=-1) # NxM
    AD_AD = np.sum(AD * AD, axis=-1) # NxM

    cond1 = np.bitwise_and(0 < AM_AB, AM_AB < AB_AB)
    cond2 = np.bitwise_and(0 < AM_AD, AM_AD < AD_AD)

    print(cond1)
    print(cond2)

    return AM


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


if __name__ == '__main__':
    my_map = Map(None)
    points = np.array([[1, 1], [2, 2], [3, 3]])

    for b in my_map.boundry:
        r = is_in_rect(points, b)

    print('broad version')
    AM, AB, AD = is_in_rect_brad(points, my_map.boundry)