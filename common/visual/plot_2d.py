# -*- coding: utf-8 -*-

from . import get_colors_from_cm
import matplotlib.pyplot as plt
import numpy as np


def draw_points(ax, points, point_size, min_values, max_values):
    """Draw points. Each line in list is [x(n_pts), y(n_pts)]"""
    n_point = len(points)
    colors = get_colors_from_cm(n_point)
    for idx, point in enumerate(points):
        ax.scatter(point[0], point[1], s=point_size, color=colors[idx])
        min_values[0] = np.minimum(min_values[0], min(point[0]))
        max_values[0] = np.maximum(max_values[0], max(point[0]))
        min_values[1] = np.minimum(min_values[1], min(point[1]))
        max_values[1] = np.maximum(max_values[1], max(point[1]))


def draw_lines(ax, lines, line_width, legends, min_values, max_values):
    """Draw lines. Each line in list is [x(n_pts), y(n_pts)]"""
    n_line = len(lines)
    colors = get_colors_from_cm(n_line)
    if legends is not None:
        assert len(legends) == n_line, 'Legend size not matched...should be {}...get {}'.format(n_line, len(legends))
    for idx, line in enumerate(lines):
        if legends is not None:
            ax.plot(line[0], line[1], linewidth=line_width, color=colors[idx], label=legends[idx])
        else:
            ax.plot(line[0], line[1], linewidth=line_width, color=colors[idx])
        min_values[0] = np.minimum(min_values[0], min(line[0]))
        max_values[0] = np.maximum(max_values[0], max(line[0]))
        min_values[1] = np.minimum(min_values[1], min(line[1]))
        max_values[1] = np.maximum(max_values[1], max(line[1]))


def draw_2d_components(
    points=None,
    point_size=2,
    lines=None,
    line_width=1,
    legends=None,
    xlabel='x',
    ylabel='y',
    title='',
    save_path=None,
):
    """draw 2d component, including lines, points, etc

    Args:
        points: list of point, each in [x(n_pts), y(n_pts)], len is N_point
        point_size: size of point, by default 2
        lines: list of line each in, [x(n_pts), y(n_pts)], len is N_line
        line_width: size of point, by default 2
        legends: list of str for the lines, len is N_line
        xlabel: x-axis label, by default x
        ylabel: y-axis label, by dafault y
        title: a string of figure title
        save_path: path to save the fig. None will only show fig
    """
    fig = plt.figure()
    ax = fig.add_subplot()

    min_values = np.array([[np.inf], [np.inf]])
    max_values = np.array([[-np.inf], [-np.inf]])

    if points is not None:
        draw_points(ax, points, point_size, min_values, max_values)

    if lines is not None:
        draw_lines(ax, lines, line_width, legends, min_values, max_values)

    # set axis limit
    axis_scale_factor = 0.2  # extent scale by such factor
    min_values -= (max_values - min_values) * axis_scale_factor
    max_values += (max_values - min_values) * axis_scale_factor

    X_min, Y_min = min_values[0], min_values[1]
    X_max, Y_max = max_values[0], max_values[1]
    max_range = np.array([X_max - X_min, Y_max - Y_min]) / 2.0

    mid_x = (X_max + X_min) * 0.5
    mid_y = (Y_max + Y_min) * 0.5
    ax.set_xlim(mid_x - max_range[0], mid_x + max_range[0])
    ax.set_ylim(mid_y - max_range[1], mid_y + max_range[1])

    if legends is not None:
        ax.legend(loc='upper right')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()
