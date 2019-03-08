import numpy as np
import seaborn
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # used for 3D plot


def ranked_plot(x, y, z, t):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    min_x = min(x)
    max_x = max(x)
    min_x -= (max_x - min_x) / 10
    max_x += (max_x - min_x) / 10

    min_y = min(y)
    max_y = max(y)
    min_y -= (max_y - min_y) / 10
    max_y += (max_y - min_y) / 10

    min_z = min(z)
    max_z = max(z)
    min_z -= (max_z - min_z) / 10
    max_z += (max_z - min_z) / 10

    ax.scatter(x, z, marker='+', color='darkturquoise', zdir='y', zs=max_y)
    ax.scatter(y, z, marker='+', color='cadetblue', zdir='x', zs=min_x)
    # ax.plot(x, y, 'k+', zdir='z', zs=min_z)

    # points with higher number of clicks
    oredered_indeces = np.argsort(t)
    cmap = matplotlib.cm.get_cmap('coolwarm')
    normalize = matplotlib.colors.Normalize(vmin=min(z[oredered_indeces[-10:]]),
                                            vmax=max(z[oredered_indeces[-10:]]))
    colors = [cmap(normalize(value)) for value in z[oredered_indeces[-10:]]]

    ax.scatter(x[oredered_indeces[-10:]],
               y[oredered_indeces[-10:]],
               z[oredered_indeces[-10:]], color=colors, s=100, edgecolors='none')

    ax.scatter(x, y, z, color='midnightblue')

    ax.set_xlim([min_x, max_x])
    ax.set_ylim([min_y, max_y])
    ax.set_zlim([min_z, max_z])

    ax.set_xlabel("lower bound")
    ax.set_ylabel("upper bound")
    ax.set_zlabel("CTR", rotation=270, labelpad=5)

    plt.draw()
    plt.savefig("../Plots/multiple_rand2.svg", transparent=True, format='svg', frameon=False)


def plot_distribution(vector, x_distribution, y_distribution, name):
    distribution_plot = seaborn.distplot(vector, color='cadetblue')
    distribution_plot.plot(x_distribution, y_distribution, color='darkorange')
    figure = distribution_plot.get_figure()
    path = "../Plots/" + name + ".svg"
    figure.savefig(path, transparent=True, format='svg', frameon=False)


def plot_multiple_functions_and_distributions(x_range, functions, distributions, name):
    figure = plt.figure()
    for dist in distributions:
        seaborn.distplot(dist)
    for func in functions:
        plt.plot(x_range, func)
    path = "../Plots/" + name + ".svg"
    figure.savefig(path, transparent=True, format='svg', frameon=False)
