import time
import numpy as np
import matplotlib.pyplot as plt
from utils_np import valid_point, valid_point_broad

# ==================================================================================================
# Helper functions
# ==================================================================================================



# ==================================================================================================
# Map class
# ==================================================================================================


class Map():
    '''
    Represents the map, holds all rectangles representing the boundry and
    the room. Can also draw the room.
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
    
    def draw_rectangle(self, rect, fill_block=True):
        '''Draws a rectangle according to the tuple rect - (x, y, dx, dy)'''
        (x, y, dx, dy) = rect
        coords = np.array([[x, y],
                           [x + dx, y],
                           [x + dx, y + dy],
                           [x, y + dy]])
        if fill_block:
            self.ax.fill(coords[:, 0], coords[:, 1], c='k')
        else:
            closed_coords = \
                np.concatenate((coords, coords[0, :].reshape(1, 2)), axis=0)
            self.ax.plot(closed_coords[:, 0], closed_coords[:, 1], 'k')

    def draw_map(self):
        '''Draws a map containing all rectangles'''
        if self.ax is None:
            (_, self.ax) = plt.subplots()

        self.draw_rectangle(self.room, fill_block=False)
        for b in self.boundry:
            self.draw_rectangle(b)


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
