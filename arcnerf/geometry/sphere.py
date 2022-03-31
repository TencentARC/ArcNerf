# -*- coding: utf-8 -*-

import math

import numpy as np


def uv_to_sphere_point(u, v, radius, origin=(0, 0, 0)):
    """Get sphere point from uv

    Args:
        u: np.array(n_pt,), in (0, 2pi)
        v: np.array(n_pt,), in (0, pi), or a fix value
        radius: radius fo sphere
        origin: origin of sphere, tuple of 3

    Returns:
        xyz: np.array(n_pt, 3), xyz position
    """
    if isinstance(v, float) or isinstance(v, int):
        v = np.repeat(np.array([v], dtype=u.dtype), u.shape[0], axis=0)
    assert u.shape == v.shape, 'Unmatched shape for u{} and v{}'.format(u.shape, v.shape)
    x = radius * (np.cos(u) * np.sin(v)) + origin[0]
    y = radius * (np.ones(np.size(u), dtype=u.dtype) * np.cos(v)) + origin[1]
    z = radius * (np.sin(u) * np.sin(v)) + origin[2]
    xyz = np.concatenate([x[:, None], y[:, None], z[:, None]], axis=-1)

    return xyz


def get_sphere_surface(radius, origin=(0, 0, 0), n_pts=100):
    """Get sphere surface position. y is up-down axis

    Args:
        radius: radius fo sphere
        origin: origin of sphere, tuple of 3
        n_pts: num of point on each dim, by default 100.

    Returns:
        x, y, z: 2-dim location.
    """
    u = np.linspace(0, 2 * np.pi, n_pts)  # horizontal
    v = np.linspace(0, np.pi, n_pts)  # vertical
    x = radius * np.outer(np.cos(u), np.sin(v)) + origin[0]
    y = radius * np.outer(np.ones(np.size(u), dtype=u.dtype), np.cos(v)) + origin[1]
    z = radius * np.outer(np.sin(u), np.sin(v)) + origin[2]

    return x, y, z


def get_regular_sphere_line(
    radius, u_start=0, origin=(0, 0, 0), n_rot=3, n_pts=100, upper=None, close=True, concat=True
):
    """Get several sphere surface line(circle) with regular vertical distance. from top to down.
     The circle is face up-down, rotate in counter-clockwise, y is up-down axis

    Args:
        radius: radius fo sphere
        u_start: start u in (0, 1), counter-clockwise direction, 0 is x-> direction
        origin: origin of sphere, tuple of 3
        n_rot: num of circle needed
        n_pts: num of point on line, by default 100.
        upper: if None, in both sphere
               if True, in upper sphere,
               if False, in lower sphere.
        close: if true, first one will be the same as last
        concat: if concat, return (n_pts, 3) array. else return a list of each level, by default is True

    Returns:
        line: list of line, each line is np.array(n_pts_per_line, 3) for a circle at different level
             or a np.array(n_pts, 3) if concat
    """
    assert 0 <= u_start <= 1, 'Invalid u_start, (0, 1) only'
    n_pts_per_rot = math.ceil(float(n_pts) / float(n_rot))
    if close:
        u = np.linspace(0, 1, n_pts_per_rot) + u_start
    else:
        u = np.linspace(0, 1, n_pts_per_rot + 1)[:n_pts_per_rot] + u_start
    u[u > 1.0] -= 1.0
    u *= (2 * np.pi)
    u = np.concatenate([u] * n_rot)[:n_pts]

    if upper is None:
        v_levels = np.linspace(1, -1, n_rot + 2)[1:-1]  # (n_rot,)
    elif upper is True:
        v_levels = np.linspace(1, 0, n_rot + 1)[1:]  # (n_rot,)
    elif upper is False:
        v_levels = np.linspace(-1, 0, n_rot + 1)[1:]  # (n_rot,)

    v_levels = (1 - v_levels) * np.pi / 2.0
    lines = []
    count = 0
    for i in range(n_rot - 1):
        lines.append(uv_to_sphere_point(u[count:count + n_pts_per_rot], v_levels[i], radius, origin))
        count += n_pts_per_rot
    lines.append(uv_to_sphere_point(u[count:], v_levels[-1], radius, origin))

    if concat:
        lines = np.concatenate(lines, axis=0)  # (n_pts, 3)

    return lines


def get_sphere_line(radius, u_start=0, v_ratio=0, origin=(0, 0, 0), n_pts=100, close=True):
    """Get sphere surface line(circle) different by angle. The circle is face up-down, rotate in counter-clockwise
     y is up-down axis

    Args:
        radius: radius fo sphere
        u_start: start u in (0, 1), counter-clockwise direction, 0 is x-> direction
        v_ratio: vertical lift ratio, in (-1, 1). 0 is largest, pos is on above.
        origin: origin of sphere, tuple of 3
        n_pts: num of point on line, by default 100.
        close: if true, first one will be the same as last

    Returns:
        line: np.array(n_pts, 3)
    """
    assert 0 <= u_start <= 1, 'Invalid u_start, (0, 1) only'
    assert -1 <= v_ratio <= 1, 'Invalid v ratio, (-1, 1) only'
    if close:
        u = np.linspace(0, 1, n_pts) + 1
    else:
        u = np.linspace(0, 1, n_pts + 1)[:n_pts] + 1
    u[u > 1.0] -= 1.0
    u *= (2 * np.pi)
    v = (1 - v_ratio) * np.pi / 2.0

    line = uv_to_sphere_point(u, v, radius, origin)

    return line


def get_spiral_line(radius, u_start=0, v_range=(1, 0), origin=(0, 0, 0), n_rot=3, n_pts=100):
    """Get spiral surface line, rotate in counter-clockwise

    Args:
        radius: radius fo sphere
        u_start: start u in (0, 1), counter-clockwise direction, 0 is x-> direction
        v_range: a tuple of v (v_start, v_end), start and end v ratio of spiral line
                    vertical lift angle, in (-1, 0). 0 is largest circle level, pos is on above.
        origin: origin of sphere, tuple of 3
        n_rot: num of full rotation, by default 3
        n_pts: num of point on line, by default 100.

    Returns:
        line: np.array(n_pts, 3)
    """
    assert 0 <= u_start <= 1, 'Invalid u_start, (0, 1) only'
    assert -1 <= v_range[0] <= 1 and -1 <= v_range[0] <= 1,\
        'Invalid v range, start and end all in (-1, 1) only'
    n_pts_per_rot = math.ceil(float(n_pts) / float(n_rot))
    u = np.linspace(0, 1, n_pts_per_rot) + u_start
    u[u > 1.0] -= 1.0
    u *= (2 * np.pi)
    u = np.concatenate([u] * n_rot)[:n_pts]
    v = np.linspace((1 - v_range[0]), (1 - v_range[1]), n_pts) * np.pi / 2.0

    line = uv_to_sphere_point(u, v, radius, origin)

    return line


def get_u_start_from_pos():
    pass


def get_v_from_pos():
    pass


def sphere_line_from_pos():
    pass


# calculate sphere circle with them, test u/v calculation