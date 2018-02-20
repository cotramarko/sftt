import numpy as np
import matplotlib.pyplot as plt
from sftt.self_organizing_network.node_network import NodeNetwork
# import sftt.self_organizing_network.utils


class NodeSelector():
    def __init__(self, nbr_base, nbr_nodes, in_range_mat, base_map):
        self.list_of_nodes = np.arange(nbr_base, nbr_base + nbr_nodes)
        self.in_range_mat = in_range_mat
        self.base_map = base_map

    def select_next_node(self):
        if len(self.list_of_nodes) > 0:
            overlap = self.in_range_mat * self.base_map[:, None]
            overlap_idx = (np.sum(overlap, axis=0) > 1)  # all nodes present

            # just keep the nodes that are still present
            nodes_left_idx = overlap_idx[self.list_of_nodes]
            nodes_left = self.list_of_nodes[nodes_left_idx]
            selected_node = np.random.choice(nodes_left)

            # Update internal representation of where base nodes are
            self.base_map[selected_node] = 1

            # Remove selected node from list of nodes to be aligned
            idx, = np.where(self.list_of_nodes == selected_node)
            self.list_of_nodes = np.delete(self.list_of_nodes, idx)

            return selected_node, nodes_left
        else:
            return None


def optimizer(base_nodes_in_range, recorded_measurements, x0, cov0):
    """
    Parameters
    ----------
    base_nodes_in_range: np.array
        (B, 2) array containing the known xy position of B number of nodes
    recorded_measurements: np.array
        (M, 1 + B) array containing M measurements observed between the B known nodes and
        the unknown node
    x0: np.array
        (2, ) array representing the mean of the prior for the unknown node
    cov0: np.array
        (2, 2) array representing the covariance of the prior for the unknown node
    """
    n = 2
    sigN = 1 + 2 * n
    w0 = 1 - n / 3 * np.ones(shape=(1))
    wi = (1 - w0) / (2 * n) * np.ones(shape=(2 * n))

    W = np.concat((w0, wi))

    pass


if __name__ == '__main__':
    # Experiment
    node_network = NodeNetwork()
    (base_map, in_range_mat, _) = node_network.get_network_properties()
    nodes = node_network.nodes

    B = node_network.B
    N = node_network.N

    node_selector = NodeSelector(B, N, in_range_mat, base_map)

    plt.figure()
    node_network.plot_node_network()
    res = node_selector.select_next_node()
    while res is not None:
        (selected_node, nodes_left) = res
        for nl in nodes_left:
            plt.plot(nodes[nl, 0], nodes[nl, 1], '.r', mfc='r')
        plt.pause(0.25)
        plt.plot(nodes[selected_node, 0], nodes[selected_node, 1], '.b', mfc='b')
        plt.pause(0.25)
        res = node_selector.select_next_node()
