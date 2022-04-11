import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


sns.set_theme()

q_grid = np.load('q_values.npy')
values = q_grid.max(axis=4)
values_map = np.mean(values, axis = (1,3))


ax = sns.heatmap(values_map)
ax.set(xlabel = 'theta', ylabel = 'x')
plt.show()
