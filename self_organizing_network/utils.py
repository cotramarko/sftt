import numpy as np
import matplotlib.pyplot as plt
from sftt.self_organizing_network.experiment import nodeNetwork, connect_positions, cross_distance
import csv


def create_pairs(in_range_matrix):
    (j, _) = in_range_matrix.shape
    nbr_of_neighbors = np.sum(in_range_matrix, axis=0)
    sort_idx = np.argsort(nbr_of_neighbors)
    nodes = np.arange(j)

    ordered_nodes = nodes[sort_idx]

    pairs = []
    while len(ordered_nodes) > 1:
        r = ordered_nodes[0]
        probs = in_range_matrix[r, ordered_nodes] / np.sum(in_range_matrix[r, ordered_nodes])
        nn = np.random.choice(ordered_nodes, p=probs)
        in_range_matrix[:, nn] = 0
        in_range_matrix[nn, :] = 0

        idx_0, = np.where(ordered_nodes == r)
        idx_1, = np.where(ordered_nodes == nn)
        idx = np.concatenate((idx_0, idx_1))
        ordered_nodes = np.delete(ordered_nodes, idx)
        print('pair:', r, nn)
        print('remaining nodes:', ordered_nodes)
        pairs.append((r, nn))

    if len(ordered_nodes) == 1:
        pairs.append((ordered_nodes[0], ))
    return pairs


if __name__ == '__main__':
    np.random.seed(1337)
    N = 100
    sensor_range = 25
    base = np.array([[5, 5], [5, 7], [3, 4]])
    x_true = np.random.uniform(size=(N, 2)) * 100

    nodes = np.concatenate((base, x_true))

    node_network = nodeNetwork(base, x_true)
    (idx_rows, idx_cols, keep_elements) = node_network.get_connections(range=sensor_range)

    ms = cross_distance(nodes)
    in_range = (ms > 0) & (ms < sensor_range)
    in_range = in_range.astype(np.int16)

    base_mapping = in_range[0:3, :].astype(np.bool)

    plt.figure()
    nodes = np.concatenate((base, x_true))
    connect_positions(nodes, idx_rows, idx_cols, keep_elements)
    plt.plot(nodes[:, 0], nodes[:, 1], 'ko', mfc=(1, 1, 1))
    plt.plot(base[:, 0], base[:, 1], 'co', mfc='c')
    plt.show()

'''
    nodes = np.concatenate((base, x_true))

    node_network = nodeNetwork(base, x_true)
    (idx_rows, idx_cols, keep_elements) = node_network.get_connections(range=sensor_range)

    z = node_network.z

    ms = cross_distance(x_true)
    in_range = (ms > 0) & (ms < sensor_range)
    in_range = in_range.astype(np.int16)

    pairs = create_pairs(in_range)
    print(pairs)

    plt.figure()
    nodes = np.concatenate((base, x_true))
    connect_positions(nodes, idx_rows, idx_cols, keep_elements)
    plt.plot(nodes[:, 0], nodes[:, 1], 'ko', mfc=(1, 1, 1))
    plt.plot(base[:, 0], base[:, 1], 'co', mfc='c')

    x0 = np.random.uniform(size=(N, 2)) * 14 - 2
    for pair in pairs:
        plt.plot(x_true[pair, 0], x_true[pair, 1], '--r')
        plt.plot(x0[pair, 0], x0[pair, 1], ':og')

    plt.axis([-5, 10, -2, 13])
    plt.show()
'''
