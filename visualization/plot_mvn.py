import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal as mvn


def mvn_mesh(mean, cov, conf_bound=3.0, N=25):
    sig3x = conf_bound * np.sqrt(cov[0, 0])
    sig3y = conf_bound * np.sqrt(cov[1, 1])
    print(sig3x)
    mu_x = mean[0]
    mu_y = mean[1]
    x = np.linspace(mu_x - sig3x, mu_x + sig3x, N)
    y = np.linspace(mu_y - sig3y, mu_y + sig3y, N)
    X, Y = np.meshgrid(x, y)

    xy = np.column_stack((X.flatten(), Y.flatten()))
    m = mvn(mean, cov)
    Z = m.pdf(xy).reshape(X.shape)

    return X, Y, Z


if __name__ == '__main__':
    mean = np.array([1, 1])
    cov = np.array([[1, 0], [0, 1]])
    (X, Y, Z) = mvn_mesh(mean, cov, N=20, conf_bound=4.0)

    xx, yy = np.meshgrid(np.linspace(-10, 10, 20), np.linspace(-10, 10, 20))

    cmap = matplotlib.cm.get_cmap('Blues_r')
    print(cmap(0.1))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap=cmap, shade=False)  # cmap='Reds_r'

    ax.plot_surface(xx, yy, np.zeros(xx.shape), color=cmap(0.0), shade=False)

    plt.axis('off')
    plt.show()
