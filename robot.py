import numpy as np
import matplotlib.pyplot as plt
from my_map import Map
import pygame

def distance_to_object(x, y, phi, map_object):
        base_point = np.array([x, y]).reshape(1, 2)
        d_vec = np.array([np.cos(phi), np.sin(phi)]).reshape(1, 2)
        dists = np.arange(0, 15, 0.01).reshape(-1, 1)

        ray = base_point + dists * d_vec

        idx = map_object.valid_point(ray)
        ray_invalid = ray[np.bitwise_not(idx)]

        d = dists[np.bitwise_not(idx)]
        ray = ray[dists.flatten() < d[0].flatten(), :]

        return ray, ray_invalid[0, :], d[0]


class Robot():
    def __init__(self, init_state, map_object):
        (x0, y0, v0, phi0, dphi0) = init_state
        self.x = x0
        self.y = y0
        self.v = v0
        self.phi = phi0
        self.dphi = dphi0

        self.map_object = map_object

    def update(self, dv, dphi):
        self.dphi = dphi
        self.phi += dphi
        self.v += dv

        self.x += self.v * np.cos(self.phi)
        self.y += self.v * np.sin(self.phi)

    def distance_to_object(self):
        return distance_to_object(self.x, self.y, self.phi, self.map_object)

# ============================================================================================
#
#   TESTS
#
# ============================================================================================


if __name__ == '__main__':
    my_map = Map()

    x, y = (5, 2.5)
    robot = Robot((x, y, 0, np.pi / 3.3, 0), my_map)
    ray, point, dist = robot.distance_to_object()

    p0, = plt.plot(robot.x, robot.y, 'bo')
    p1, = plt.plot(ray[:, 0], ray[:, 1])
    p2, = plt.plot(point[0], point[1], 'rx')

    my_map.draw_map()

    pygame.init()
    pygame.display.set_mode((100, 100))
    for _ in range(100):
        p0.remove()
        p1.remove()
        p2.remove()

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                print(event.key)

        dv = 0
        dphi = 0
        robot.update(dv, dphi)
        ray, point, dist = robot.distance_to_object()

        p0, = plt.plot(robot.x, robot.y, 'bo')
        p1, = plt.plot(ray[:, 0], ray[:, 1], ':r')
        p2, = plt.plot(point[0], point[1], 'rx')

        plt.pause(0.1)
