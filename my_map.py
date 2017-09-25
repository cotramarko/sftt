import numpy as np
import matplotlib.pyplot as plt
import time


def draw_rectangle(spec, fill_block=True):
    '''Draws a rectangle according to the tuple spec - (x, y, dx, dy)'''
    (x, y, dx, dy) = spec
    coords = np.array([[x, y],
                       [x + dx, y],
                       [x + dx, y + dy],
                       [x, y + dy]])
    if fill_block:
        plt.fill(coords[:, 0], coords[:, 1], c='k')
    else:
        closed_coords = np.concatenate((coords, coords[0, :].reshape(1, 2)), axis=0)
        plt.plot(closed_coords[:, 0], closed_coords[:, 1], 'k')


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


class Map():
    '''Represents the map'''
    def __init__(self):
        self.boundry = [(3, 0, 1, 4),
                        (7, 0, 3, 3),
                        (5, 5, 5, 1),
                        (2, 8, 5, 2),
                        (1, 5, 2, 2)]

    def draw_map(self):
        '''Draws a map containing all rectangles'''
        draw_rectangle((0, 0, 10, 10), fill_block=False)
        for b in self.boundry:
            draw_rectangle(b)

    def valid_point(self, p):
        ''' Returns bool for a set of points whether they are valid or not in the map,
            i.e. not outside of the room or inside of a solid object. '''
        res = is_in_rect(p, (0, 0, 10, 10))
        for b in self.boundry:
            res = np.bitwise_and(res, np.bitwise_not(is_in_rect(p, b)))

        return res


# ============================================================================================
#
#   TESTS
#
# ============================================================================================

if __name__ == '__main__':
    my_map = Map()
    p = np.array([[0.1, 0.1], [5.1, 5.1]])

    my_map.draw_map()
    plt.scatter(p[:, 0], p[:, 1], c='r')
    plt.show()
