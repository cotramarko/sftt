import numpy as np
import matplotlib.pyplot as plt
from sftt.self_organizing_network.node_network import NodeNetwork
# import sftt.self_organizing_network.utils


if __name__ == '__main__':
    # Experiment
    node_network = NodeNetwork()
    (base_map, in_range_mat, _) = node_network.get_network_properties()
    nodes = node_network.nodes

    B = node_network.B
    N = node_network.N

    list_of_nodes = np.arange(B, N + B)

    plt.figure()
    node_network.plot_node_network()

    i = 0
    while len(list_of_nodes) > 0:
        overlap = in_range_mat * base_map[:, None]
        overlap_idx = (np.sum(overlap, axis=0) > 1)  # all nodes present

        nodes_left_idx = overlap_idx[list_of_nodes]  # just keep the nodes that are still present
        print('nodes_left_idx', nodes_left_idx.shape)
        nodes_left = list_of_nodes[nodes_left_idx]
        print('nodes_left', nodes_left)

        for nl in nodes_left:
            plt.plot(nodes[nl, 0], nodes[nl, 1], '.r', mfc='r')
        plt.pause(0.25)
        i += 1
        selected_node = np.random.choice(nodes_left)

        base_map[selected_node] = 1
#        base_map[:, selected_node] = 1

        idx, = np.where(list_of_nodes == selected_node)
        list_of_nodes = np.delete(list_of_nodes, idx)
        plt.plot(nodes[selected_node, 0], nodes[selected_node, 1], '.b', mfc='b')
        plt.pause(0.25)
        i += 1
