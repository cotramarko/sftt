import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.stats import norm


fig = plt.figure()
ax = fig.add_subplot(111)
col = (1, 0.5, 0, 0.4)
ax.bar([0, 1], [0.7, 0.3], color=col, edgecolor=(col, col))

ax.text(0, 0.71, r'$0.7$', ha='center')
ax.text(1, 0.31, r'$0.3$', ha='center')

ax.text(0, -0.05, r'$p(x_{k-1}=0|y_{1:k-1})$', ha='center')
ax.text(1, -0.05, r'$p(x_{k-1}=1|y_{1:k-1})$', ha='center')

plt.axis('off')
plt.savefig('bar.png', dpi=300)
#plt.show()
