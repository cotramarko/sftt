import numpy as np
import matplotlib.pyplot as plt
from sftt.self_organizing_network.utils import in_range, cross_distance

BASE_NODES = np.array([[5, 5], [5, 7], [3, 4]])


class NodeNetwork():
    """
    Randomises a network of interconnect nodes, based on certain constraints such as size of 
    the area the network is in, sensor range and the number of nodes
    (with known and unknown position)

    Public methods
    ----------

    get_network_properties()

    get_network_measurement()

    plot_node_network()
    """

    def __init__(self, base_nodes=BASE_NODES, num_unknown_nodes=100, sensor_range=25,
                 size_of_square=100, seed=1337):
        """
        Parameters
        ----------
        base_nodes: np.array
            (B, 2) array containing the known xy position of B number of nodes
        num_unknown_nodes: int
            The number of nodes with unknown xy position to be generated in the area
        sensor_range: int
            The range that each node is able to communicate in
        size_of_square: int
            Length of one side of the square area that the nodes are placed in
        seed: int
            Seed to be used for generating random numbers

        """
        np.random.seed(seed)
        self.N = num_unknown_nodes
        self.base = base_nodes
        (self.B, _) = base_nodes.shape
        self.x_true = np.random.uniform(size=(self.N, 2)) * size_of_square
        self.nodes = np.concatenate((self.base, self.x_true))

        self._base_map = np.concatenate((np.ones((self.B)), np.zeros(self.N)))
        self.node_true_dist_mat = cross_distance(self.nodes)
        self.in_range_mat = in_range(self.node_true_dist_mat, sensor_range)

    def get_network_properties(self):
        """
        Returns
        ----------
        self._base_map: np.array
            (B + N, ) array containing 1 & 0, where 1 indicates that the corresponding node
            numbered by this index is a base node with known position.
            E.g. self._base_map = [0, 0, 1, 0] indicates that node 2 is a base node.

        self.in_range_mat: np.array
            (B + N, B + N) symmetric matrix representing which nodes are in range of eachother.
            Each row and column represent a node in the network. An element equal to 1 indicates
            the corresponding (row, column) node pair are in range of eachother. An element equal
            to 0 indicates the opposite.

        self.node_true_dist_mat: np.array
            (B + N, B + N) symmetric matrix representing the true distance between nodes.
            Each row and column represent a node in the network. An element indicates euqlidian
            distance between the corresponding (row, column) node pair.

        Dimensions
        ----------
        B: scalar
            The number of base nodes in the network (i.e. known position)
        N: scalar
            The number of nodes in the network with unknown position

        """
        return self._base_map, self.in_range_mat, self.node_true_dist_mat

    def get_network_measurement(self, r_cov=0.1 ** 2):
        """ Returns a single scan of all measurements inbetween the nodes the network.

        Parameters
        ----------
        r_cov: scalar
            Covariance of the (zero-mean) measurement noise that the measurements are subject to

        Returns
        ----------
        z: np.array
            (B + N, B + N) symmetric matrix representing the measured distance between nodes.
            Each row and column represent a node in the network. An element indicates the measured
            euqlidian distance between the corresponding (row, column) node pair.
        """
        M = self.N + self.B
        meas_noise = np.random.normal(scale=np.sqrt(r_cov), size=(M, M))
        m = np.tril(meas_noise) + np.tril(meas_noise, -1).T
        m = m * np.logical_not(np.eye(M))

        z = self.node_true_dist_mat + m
        return z

    def _get_connections(self):
        """ Returns a mapping of how to (graphically) connect the nodes that are in range of
        eachother, by isolating the upper triangular matrix of the self.

        Returns
        ----------
        (idx_rows, idx_cols, connected_nodes): (np.array, np.array, np.array)

        """
        (r, _) = self.node_true_dist_mat.shape
        (idx_rows, idx_cols) = np.triu_indices(r, k=1)
        connected_nodes = self.in_range_mat[idx_rows, idx_cols]

        return (idx_rows, idx_cols, connected_nodes)

    def _plot_node_connections(self, ax):
        (idx_rows, idx_cols, connected_nodes) = self._get_connections()
        cval = 0.75
        gray = (cval, ) * 4
        for (idx_r, idx_c, connected_node) in zip(idx_rows, idx_cols, connected_nodes):
            if connected_node:
                pair_idx = (idx_r, idx_c)
                ax.plot(self.nodes[pair_idx, 0], self.nodes[pair_idx, 1], '-', color=gray,
                        linewidth=0.5)

    def plot_node_network(self, ax=None):
        """ Plots the node network, i.e. where all of the nodes are located, which other nodes
        are in range and where base nodes are located.

        Parameters
        ----------
        ax: axis handle (object)
            Optional parameter, specifies the axis to use for plotting the node network. If not
            specified the current axis is used instead.

        """

        if ax is None:
            ax = plt.gca()

        self._plot_node_connections(ax)
        (ph0, ) = ax.plot(self.base[:, 0], self.base[:, 1], 'co', mfc='c')
        (ph1, ) = ax.plot(self.x_true[:, 0], self.x_true[:, 1], 'ko', mfc=(1, 1, 1))
        plt.legend([ph0, ph1], ['base nodes \n(known position)', 'nodes \n(unknown position)'],
                   loc=1, fontsize=8)


if __name__ == '__main__':
    node_network = NodeNetwork(num_unknown_nodes=100)
    print(node_network.get_network_measurement())
    print(node_network.get_network_measurement())
    node_network.plot_node_network()
    plt.show()
