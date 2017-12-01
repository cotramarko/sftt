import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.stats as stats

normal_var = stats.norm(loc=0, scale=1)

dx = 10
x = np.linspace(-dx, dx, 100)
p = normal_var.pdf(x)

samples = 2.5 * normal_var.rvs(100000) + 2

xb = 9

plt.figure(figsize=(4, 8))
plt.subplot(2, 1, 1)
plt.plot(x, p)
plt.xlabel(r'$x$')
plt.title(r'pdf of $\mathcal{N}(\mu=0, \sigma=1)$')
ax = plt.gca()
ax.set_xlim([-xb, xb])
ax.set_ylim([-0.005, 0.42])

ax.set_yticks([])

plt.subplot(2, 1, 2)
ax = plt.gca()
plt.hist(samples, bins=100, normed=True)
plt.title(r'Histogram of $s_x$')
plt.xlabel(r'$s_x$')
ax.set_xlim([-xb, xb])
ax.set_yticks([])

plt.tight_layout(pad=0)

plt.savefig('comp_2.png', dpi=75)
plt.show()
