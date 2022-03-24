# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt


def plot_curve_2d(x, y, xlabel, ylabel, title, save_path=None):
    """Plot curve x-y in 2d."""
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()
