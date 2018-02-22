"""The methods in here take in a node_network instance, and attaches scenarios to it."""
import numpy as np
from utils import in_range, cross_distance


def get_large_baseline(self):
    """Creates the large random baseline."""
    base_nodes = np.array([[5, 5], [5, 7], [3, 4]])
    num_unknown_nodes = 100
    sensor_range = 25,
    size_of_square = 100
    seed = 1337

    np.random.seed(1337)
    self.N = num_unknown_nodes
    self.base = base_nodes
    (self.B, _) = base_nodes.shape
    self.x_true = np.random.uniform(size=(self.N, 2)) * size_of_square
    self.nodes = np.concatenate((self.base, self.x_true))

    self._base_map = np.concatenate((np.ones((self.B)), np.zeros(self.N)))
    self.node_true_dist_mat = cross_distance(self.nodes)
    self.in_range_mat = in_range(self.node_true_dist_mat, sensor_range)

    return self


def get_small_symmetry_1(self):
    """Creates a 4 node problem with two node symmetry problem."""
    base_nodes = np.array([
        [-2, 0],
        [2, 0]
    ])
    num_unknown_nodes = 2
    in_range_mat = np.array([
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 0, 0]
    ])
    self.N = num_unknown_nodes
    self.base = base_nodes
    (self.B, _) = base_nodes.shape
    self.x_true = np.array([
        [0, 1],
        [-2, -1]
    ])
    self.nodes = np.concatenate((self.base, self.x_true))
    self.node_true_dist_mat = cross_distance(self.nodes)
    self.in_range_mat = in_range_mat

    return self
