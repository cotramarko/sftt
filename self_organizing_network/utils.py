import numpy as np
import matplotlib.pyplot as plt


def diag_block_mat_boolindex(L):
    shp = L[0].shape
    mask = np.kron(np.eye(len(L)), np.ones(shp)) == 1
    out = np.zeros(np.asarray(shp) * len(L), dtype=int)
    out[mask] = np.concatenate(L).ravel()
    return out


def cross_distance(x):
    (N, _) = x.shape
    delta = x[None, ...] - x[:, None, :]
    z = np.linalg.norm(delta.reshape(-1, 2), ord=2, axis=1)
    return z.reshape(N, N)


def in_range(meas_mat, sensor_range):
    in_range_mat = (meas_mat > 0) & (meas_mat < sensor_range)
    return in_range_mat


def draw_cov(x, cov, col='k'):
    phi = np.linspace(0, 2 * np.pi, 100)
    sig3 = x[:, None] + 3 * np.dot(np.linalg.cholesky(cov), np.array([np.cos(phi), np.sin(phi)]))
    plt.plot(sig3[0, :], sig3[1, :], col)


if __name__ == '__main__':
    # Temp kode, needs to be borken out into a function/class in different module
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

    print(in_range)
    base_map = np.zeros(shape=(N + 3))
    base_map[0:3] = 1
#    base_map[:, 0:3] = 1

    list_of_nodes = np.arange(N + 3)

    # plt.figure()
    nodes = np.concatenate((base, x_true))
    connect_positions(nodes, idx_rows, idx_cols, keep_elements)
    plt.plot(nodes[:, 0], nodes[:, 1], 'ko', mfc=(1, 1, 1))
    plt.plot(base[:, 0], base[:, 1], 'co', mfc='c')
    i = 0
    while len(list_of_nodes) > 0:

        overlap = in_range * base_map[:, None]
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
