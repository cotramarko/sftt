import numpy as np
import matplotlib.pyplot as plt
from my_map import Map


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
    def __init__(self, init_state, map_object, dt):
        (x0, y0, v0, phi0, dphi0) = init_state
        self.x = x0
        self.y = y0
        self.v = v0
        self.phi = phi0
        self.dphi = dphi0
        self.dt = dt
        self.T = 0

        self.map_object = map_object

    def update(self, dv, dphi, f=None):
        '''f is optional filewriter'''
        self.T += self.dt

        self.dphi = dphi
        self.phi += dphi * self.dt
        self.v += dv * self.dt

        self.x += self.v * np.cos(self.phi)
        self.y += self.v * np.sin(self.phi)

        if f is not None:
            f.write(self.get_state_as_string())

    def get_state_as_string(self):
        state_arr = '%.5f, %.5f, %.5f, %.5f, %.5f, %.5f \n' % \
            (self.T, self.x, self.y, self.v, self.phi, self.dphi)
        return state_arr

    def distance_to_object(self):
        return distance_to_object(self.x, self.y, self.phi, self.map_object)


class Robot_Illustrator():
    def __init__(self):
        pass

# ============================================================================================
#
#   TESTS
#
# ============================================================================================


if __name__ == '__main__':
    my_map = Map()

    x, y = (4.5, 1.0)
    dt = 0.1
    robot = Robot((x, y, 0, np.pi / 2, 0), my_map, dt)

    stop = False

    # Define robot movements
    def press(event):
        global f, stop
        if event.key == 'up':
            robot.update(0.1, 0, f)
        if event.key == 'down':
            robot.update(-0.1, 0, f)
        if event.key == 'left':
            robot.update(0, np.pi / 10, f)
        if event.key == 'right':
            robot.update(0, -np.pi / 10, f)
        if event.key == 'enter':
            stop = True

    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('key_press_event', press)

    # This below should be a function
    ray, point, dist = robot.distance_to_object()
    t = plt.text(10, 10, str(dist))
    p0, = plt.plot(robot.x, robot.y, 'bo')
    p1, = plt.plot(ray[:, 0], ray[:, 1])
    p2, = plt.plot(point[0], point[1], 'rx')

    my_map.draw_map()

    with open('robot_log.txt', 'w') as f:
        # Write initial state
        f.write(robot.get_state_as_string())
        while True:
            # This below should be a function
            t.remove()
            p0.remove()
            p1.remove()
            p2.remove()

            robot.update(0, 0, f)

            # TODO: We need to save the measurements as well!

            # This below should be a function
            ray, point, dist = robot.distance_to_object()
            t = plt.text(7, 11, 'r: %.3f, v: %.3f' % (dist[0], robot.v))
            p0, = plt.plot(robot.x, robot.y, 'bo')
            p1, = plt.plot(ray[:, 0], ray[:, 1], ':r')
            p2, = plt.plot(point[0], point[1], 'rx')

            plt.pause(0.1)

            if stop:
                break
