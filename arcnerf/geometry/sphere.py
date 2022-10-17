# -*- coding: utf-8 -*-

import math

import numpy as np
import torch
import torch.nn as nn

from common.utils.torch_utils import torch_to_np
from .transformation import normalize
from .ray import sphere_ray_intersection


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


def get_uv_from_pos(pos, origin=(0.0, 0.0, 0.0), radius=None):
    """Get the u, v in scaled range, and radius from pos

    Args:
        pos: np(3, ) position of point in xyz
        origin: the origin of sphere
        radius: radius of sphere, If None, assume the point is on the sphere

    Returns:
        u: (0, 1) representing (0, 2pi) xz-direction
        v: (-1, 1) representation (0, pi) y-direction
    """
    if radius is None:
        radius = np.linalg.norm(pos - np.array(origin, dtype=pos.dtype))
    v = np.arccos((pos[1] - origin[1]) / radius)  # in (0, pi)
    u = np.arctan((pos[2] - origin[2]) / (pos[0] - origin[0]))
    if u < 0:
        u += (2 * np.pi)  # in (0, 2pi)
    u = u / (2 * np.pi)  # in (0, 1)
    v = 1 - (v * 2.0 / np.pi)  # in (-1, 1)

    return u, v, radius


def get_circle(origin, radius, normal, n_pts=100, close=True):
    """Get circle representation in 3D

    Args:
        origin: np(3,), origin of the circle
        normal: np(3,), norm of the triangle
        radius: radius of circle
        n_pts: num of sampled points
        close: if true, first one will be the same as last

    Returns:
        line: np.array(n_pts, 3)
    """
    if close:
        u = np.linspace(0, 2 * np.pi, n_pts)
    else:
        u = np.linspace(0, 2 * np.pi, n_pts + 1)[:n_pts]

    a = np.cross(normal, np.array([1.0, 0.0, 0.0]))
    if not np.any(a):  # zeros
        a = np.cross(normal, np.array([0.0, 1.0, 0.0]))
    b = np.cross(normal, a)
    a = normalize(a)
    b = normalize(b)

    x = radius * (a[0] * np.cos(u) + b[0] * np.sin(u)) + origin[0]
    y = radius * (a[1] * np.cos(u) + b[1] * np.sin(u)) + origin[1]
    z = radius * (a[2] * np.cos(u) + b[2] * np.sin(u)) + origin[2]

    line = np.concatenate([x[:, None], y[:, None], z[:, None]], axis=-1)

    return line


def get_sphere_surface(radius, origin=(0, 0, 0), n_pts=100):
    """Get sphere surface position. y is up-down axis, y+ is down.

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
     The circle is face up-down, rotate in counter-clockwise, y is up-down axis, y+ is down.

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
        v_levels = np.linspace(-1, 1, n_rot + 2)[1:-1]  # (n_rot,)
    elif upper is True:
        v_levels = np.linspace(-1, 0, n_rot + 1)[1:]  # (n_rot,)
    elif upper is False:
        v_levels = np.linspace(1, 0, n_rot + 1)[1:]  # (n_rot,)

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
     y is up-down axis, y+ is down.

    Args:
        radius: radius fo sphere
        u_start: start u in (0, 1), counter-clockwise direction, 0 is x-> direction
        v_ratio: vertical lift ratio, in (-1, 1). 0 is largest, pos is on below part.
        origin: origin of sphere, tuple of 3
        n_pts: num of point on line, by default 100.
        close: if true, first one will be the same as last

    Returns:
        line: np.array(n_pts, 3)
    """
    assert 0 <= u_start <= 1, 'Invalid u_start, (0, 1) only'
    assert -1 <= v_ratio <= 1, 'Invalid v ratio, (-1, 1) only'
    if close:
        u = np.linspace(0, 1, n_pts) + u_start
    else:
        u = np.linspace(0, 1, n_pts + 1)[:n_pts] + u_start
    u[u > 1.0] -= 1.0
    u *= (2 * np.pi)
    v = (1 - v_ratio) * np.pi / 2.0

    line = uv_to_sphere_point(u, v, radius, origin)

    return line


def get_spiral_line(radius, u_start=0, v_range=(-1, 0), origin=(0, 0, 0), n_rot=3, n_pts=100):
    """Get spiral surface line, rotate in counter-clockwise

    Args:
        radius: radius fo sphere
        u_start: start u in (0, 1), counter-clockwise direction, 0 is x-> direction
        v_range: a tuple of v (v_start, v_end), start and end v ratio of spiral line
                    vertical lift angle, in (-1, 1). 0 is largest circle level, pos is on below part.
        origin: origin of sphere, tuple of 3
        n_rot: num of full rotation, by default 3
        n_pts: num of point on line, by default 100.

    Returns:
        line: np.array(n_pts, 3)
    """
    assert 0 <= u_start <= 1, 'Invalid u_start, (0, 1) only'
    assert -1 <= v_range[0] <= 1 and -1 <= v_range[0] <= 1, 'Invalid v range, start and end all in (-1, 1) only'
    n_pts_per_rot = math.ceil(float(n_pts) / float(n_rot))
    u = np.linspace(0, 1, n_pts_per_rot + 1)[:n_pts_per_rot] + u_start  # do not close
    u[u > 1.0] -= 1.0
    u *= (2 * np.pi)
    u = np.concatenate([u] * n_rot)[:n_pts]
    v = np.linspace((1 - v_range[0]), (1 - v_range[1]), n_pts) * np.pi / 2.0

    line = uv_to_sphere_point(u, v, radius, origin)

    return line


def get_swing_line(radius, u_range=(0, 0.5), v_range=(-1, 0), origin=(0, 0, 0), n_rot=3, n_pts=100, reverse=False):
    """Get swing surface line, always in counter-clockwise then clockwise order (top-down look)

    Args:
        radius: radius fo sphere
        u_range: a tuple of u (u_start, u_end), start and end u pos of line
                    horizontal pos, in (0, 1), counter-clockwise direction, 0 is x-> direction
        v_range: a tuple of v (v_start, v_end), start and end v ratio of spiral line
                    vertical lift angle, in (-1, 1). 0 is largest circle level, pos is on below part.
        origin: origin of sphere, tuple of 3
        n_rot: num of swing repeat, by default 3
        n_pts: num of point on line, by default 100.
        reverse: If False, from u_start -> u_end -> u_start.
                  If True, u is swing in clockwise, (u 0-1 is counter-clockwise in fact). Will be
                        u_end -> 1 -> u_start -> 1 -> u_end


    Returns:
        line: np.array(n_pts, 3)
    """
    assert 0 <= u_range[0] <= u_range[1] <= 1, 'Invalid u_range, in (0, 1) in order only'
    assert -1 <= v_range[0] <= 1 and -1 <= v_range[0] <= 1, 'Invalid v range, start and end all in (-1, 1) only'
    n_pts_per_rot_half = math.floor(float(n_pts) / float(n_rot) / 2.0 + 1)

    if reverse:
        u = np.linspace(u_range[1], 1 + u_range[0], n_pts_per_rot_half)
        u[u > 1.0] -= 1.0
        u = np.concatenate([u, np.flip(u)[1:-1]])
    else:
        u = np.linspace(u_range[0], u_range[1], n_pts_per_rot_half)
        u = np.concatenate([u, np.flip(u)[1:-1]])

    u *= (2 * np.pi)
    u = np.concatenate([u] * (n_rot + 1))[:n_pts]
    v = np.linspace((1 - v_range[0]), (1 - v_range[1]), n_pts) * np.pi / 2.0

    line = uv_to_sphere_point(u, v, radius, origin)

    return line


class Sphere(nn.Module):
    """A simple sphere class"""

    def __init__(self, origin=(0, 0, 0), radius=1.0, dtype=torch.float32, requires_grad=False):
        """
        Args:
            origin: origin point(centroid of sphere), a tuple of 3

            dtype: dtype of params. By default is torch.float32
            requires_grad: whether the parameters requires grad. If True, waste memory for graphic.
        """
        super(Sphere, self).__init__()
        self.requires_grad = requires_grad
        self.dtype = dtype

        self.origin = nn.Parameter(torch.tensor([0.0, 0.0, 0.0], dtype=dtype), requires_grad=self.requires_grad)
        self.radius = nn.Parameter(torch.tensor([0.0], dtype=dtype), requires_grad=self.requires_grad)

        self.set_params(origin, radius)

    def set_params(self, origin, radius):
        """you can call outside to reset the params"""
        self.set_origin(origin)
        self.set_radius(radius)

    @torch.no_grad()
    def set_origin(self, origin=(0.0, 0.0, 0.0)):
        """Set the origin """
        self.origin[0] = origin[0]
        self.origin[1] = origin[1]
        self.origin[2] = origin[2]

    def get_origin(self, in_tuple=False):
        """Gets origin in tensor(3, )"""
        if in_tuple:
            return tuple(torch_to_np(self.origin).tolist())
        return self.origin

    @torch.no_grad()
    def set_radius(self, radius):
        """Set the radius """
        self.radius[0] = radius

    def get_radius(self, in_float=False):
        """Gets radius in tensor(1, )"""
        if in_float:
            return float(self.radius[0])
        return self.radius

    def ray_sphere_intersection(self, rays_o: torch.Tensor, rays_d: torch.Tensor):
        """Calculate the rays intersection with the sphere
        Args:
            rays_o: ray origin, (N_rays, 3)
            rays_d: ray direction, (N_rays, 3)

        Returns:
            near: near intersection zvals. (N_rays, 1)
                  If only 1 intersection: if not tangent, same as far; else 0. clip by 0.
            far:  far intersection zvals. (N_rays, 1)
                  If only 1 intersection: if not tangent, same as far; else 0.
            pts: (N_rays, 2, 3), each ray has near/far two points with each sphere.
                                      if nan, means no intersection at this ray
            mask: (N_rays, 1), show whether each ray has intersection with the sphere, BoolTensor
        """
        near, far, pts, mask = sphere_ray_intersection(
            rays_o, rays_d, self.get_radius(in_float=True), self.get_origin(in_tuple=True)
        )

        return near, far, pts, mask
