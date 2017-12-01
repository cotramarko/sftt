import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.stats import norm


np.random.seed(1311)  # 1319
#  Generate a couple of 1D gausian mixture models


def random_gaussian_mixture(nbr_components, mu_limits, std_limits, x_limits, N=100):
    weights = np.random.rand(nbr_components)
    weights = weights / np.sum(weights)
    weights = weights.reshape(-1, 1)

    means = mu_limits[0] + np.random.rand(nbr_components) * mu_limits[1]
    stds = std_limits[0] + np.random.rand(nbr_components) * std_limits[1]

    normals = norm(loc=means.reshape(-1, 1), scale=stds.reshape(-1, 1))

#    x = np.linspace(x_limits[0], x_limits[1], num=N).reshape(1, -1)
    x = np.arange(x_limits[0], x_limits[1], 1 / N).reshape(1, -1)
    mm_pdf = np.sum(weights * normals.pdf(x), axis=0)

    # append zeros
    x = np.hstack((x[0, 0], x.flatten(), x[0, -1]))
    mm_pdf = np.hstack((0, mm_pdf, 0))
    return mm_pdf, x.flatten()


def draw_post(ax, y, pdf):
    x = 16 * np.ones(pdf.shape)
    vert = np.vstack((x, y, pdf)).transpose(1, 0)

    tri = Poly3DCollection([vert])
    tri.set_color((0.5, 0.4, 0.7, 0.6))
    tri.set_edgecolor('k')
    ax.add_collection3d(tri)
    return tri


def draw_pred(ax, vert):
    tri = Poly3DCollection([vert])
    tri.set_color((1, 1, 1, 0))
    tri.set_edgecolor((0.4, 0.4, 0.4))
    ax.add_collection3d(tri)
    return tri


def connect_motion_post(ax, cond, post_pdf):
    elems = post_pdf.size
    ax.plot(np.linspace(10, 16, elems), cond * np.ones(elems), zs=np.zeros(elems), c=(0, 0, 0))
    idx = np.where(xd == cond)
    ax.plot([16, 16], [cond, cond], zs=[0, post_pdf[idx]], c=(0, 0, 0))
    ax.plot([16], [cond], zs=post_pdf[idx], c=(0, 0, 0), marker='.')

    return post_pdf[idx]


def draw_motion_model(ax, verts):
    colors = [(1, 0, 0, 0.5),
              (1, 1, 0, 0.5),
              (1, 0, 1, 0.5),
              (0, 0.5, 1, 0.5),
              (0.3, 0.6, 1, 0.5),
              (0, 1, 1, 0.5),
              (0.6, 1, 0.3, 0.5),
              (1, 0, 0, 0.5)]

    for i, vert in enumerate(verts):
        tri = Poly3DCollection([vert])
        tri.set_color(colors[i])
        tri.set_edgecolor('k')
        ax.add_collection3d(tri)


def draw_labels(ax):
    (d, e) = (12, 6)
    plt.axis([-e, d, -e, d])

    ax.plot(np.linspace(0, 10, 2), np.zeros(2), 0, c='k')
    ax.plot(10 * np.ones(2), np.linspace(0, 10, 2), 0, c='k')
    ax.plot(16 * np.ones(2), np.linspace(0, 10, 2), 0, c='k')

    ax.view_init(elev=56, azim=-47)
    ax.text(5, -7.5, 0, r'$x_{k}$')
    ax.text(1, -14.5, 0.3, r'$p(x_{k}|y_{k:k-1})$')

    ax.text(17, c, 0, r'$x_{k-1}$')
    ax.text(16, 7, 0.2, r'$p(x_{k-1}|y_{k:k-1})$')

    ax.text(5, -3, 0.4, r'$p(x_{k}|x_{k-1})$')


if __name__ == '__main__':
    nbr_components = 4
    mu_limits = (0, 10)
    std_limits = (1, 4)
    x_limits = (0, 10)

    pdfs = []
    conditionals = [1, 2, 3, 4, 5, 6, 7, 8]
    nbr_conds = len(conditionals)

    cond_vals = []
    verts = []

    scaling = 3
    img_count = 0

    for c in conditionals:
        (pdf, xd) = random_gaussian_mixture(nbr_components, mu_limits, std_limits, x_limits)
        pdfs.append(pdf)
        arr = c * np.ones(pdf.shape)

        cond_vals.append(arr)
        verts.append(np.vstack((xd, arr, pdf)).transpose(1, 0))

    (post_pdf, y) = random_gaussian_mixture(nbr_components, [1, 8], [0.6, 1.8], x_limits)

    #  Draw without anything
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    draw_motion_model(ax, verts)  # add p(x_k | x_{k-1})
    ax.set_zlim(0, 0.3)

    draw_post(ax, y, post_pdf)  # add p(x_k | y_{k:k-1})
    x = -6 * np.ones(pdfs[0].shape)
    ax.plot(xd, x, np.zeros(post_pdf.shape), 'k')

    draw_labels(ax)
    plt.axis('off')
    plt.savefig('ani_%0.3d.png' % img_count, dpi=300)
    img_count += 1

    lik = []
    for i in range(nbr_conds):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        draw_motion_model(ax, verts)  # add p(x_k | x_{k-1})
        ax.set_zlim(0, 0.3)

        draw_post(ax, y, post_pdf)  # add p(x_k | y_{k:k-1})

        le = connect_motion_post(ax, conditionals[i], post_pdf)
        lik.append(le)

        # draw pred model
        x = -6 * np.ones(pdfs[0].shape)
        ax.plot(xd, x, np.zeros(post_pdf.shape), 'k')

        for j in range(i + 1):
            vert = np.vstack((xd, x, scaling * lik[j] * pdfs[j])).transpose(1, 0)
            draw_pred(ax, vert)

    #    tots = np.sum(np.vstack(np.array(pdfs)), axis=0)
    #    ax.plot(xd, x, np.zeros(tots.shape), 'r')

        draw_labels(ax)
        plt.axis('off')
        plt.savefig('ani_%0.3d.png' % img_count, dpi=300)
        img_count += 1

    # Show everything
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    draw_motion_model(ax, verts)  # add p(x_k | x_{k-1})
    ax.set_zlim(0, 0.3)

    draw_post(ax, y, post_pdf)  # add p(x_k | y_{k:k-1})

    # draw pred model
    x = -6 * np.ones(pdfs[0].shape)
    ax.plot(xd, x, np.zeros(post_pdf.shape), 'k')

    for j in range(nbr_conds):
        vert = np.vstack((xd, x, scaling * lik[j] * pdfs[j])).transpose(1, 0)
        draw_pred(ax, vert)

    draw_labels(ax)
    plt.axis('off')
    plt.savefig('ani_%0.3d.png' % img_count, dpi=300)
    img_count += 1

    # Sum over everything in pred
    for i in range(nbr_conds):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        draw_motion_model(ax, verts)  # add p(x_k | x_{k-1})
        ax.set_zlim(0, 0.3)

        draw_post(ax, y, post_pdf)  # add p(x_k | y_{k:k-1})

        # draw pred model
        x = -6 * np.ones(pdfs[0].shape)
        ax.plot(xd, x, np.zeros(post_pdf.shape), 'k')

        # Summation
        res = 0
        for j in range(i + 1):
            res = res + scaling * lik[j] * pdfs[j]

        for j in range(i+1, nbr_conds):
            vert = np.vstack((xd, x, scaling * lik[j] * pdfs[j])).transpose(1, 0)
            draw_pred(ax, vert)

        ax.plot(xd, x, np.zeros(x.shape), 'r')
        vert = np.vstack((xd, x, np.array(res))).transpose(1, 0)
        tri = Poly3DCollection([vert])
        tri.set_color((1, 0, 0, 0.5))
        tri.set_edgecolor((1, 0, 0))
        ax.add_collection3d(tri)

        draw_labels(ax)
        plt.axis('off')
        plt.savefig('ani_%0.3d.png' % img_count, dpi=300)
        img_count += 1

    # Show result
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    draw_motion_model(ax, verts)  # add p(x_k | x_{k-1})
    ax.set_zlim(0, 0.3)

    draw_post(ax, y, post_pdf)  # add p(x_k | y_{k:k-1})

    # draw pred model
    x = -6 * np.ones(pdfs[0].shape)
    ax.plot(xd, x, np.zeros(post_pdf.shape), 'k')

    # Summation
    res = 0
    for j in range(nbr_conds):
        res = res + scaling * lik[j] * pdfs[j]

#    for j in range(i, nbr_conds):
#        vert = np.vstack((xd, x, scaling * lik[j] * pdfs[j])).transpose(1, 0)
#        draw_pred(ax, vert)

    ax.plot(xd, x, np.zeros(x.shape), 'r')
    vert = np.vstack((xd, x, np.array(res))).transpose(1, 0)
    tri = Poly3DCollection([vert])
    tri.set_color((1, 0, 0, 0.5))
    tri.set_edgecolor((1, 0, 0))
    ax.add_collection3d(tri)

    draw_labels(ax)
    plt.axis('off')
    plt.savefig('ani_%0.3d.png' % img_count, dpi=300)
    img_count += 1
