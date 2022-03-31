# -*- coding: utf-8 -*-

import numpy as np

from .sphere import get_circle
from .transformation import normalize


def circumcircle_from_triangle(verts, n_pts=100, close=True):
    """Get the origin, radius, normal, and circle representation for circumcircle of triangle
    Returns None if all three point on same line
    Ref: https://blog.csdn.net/webzhuce/article/details/88371649

    Args:
        verts: np(3, 3), triangle verts, second dim is xyz
        n_pts: num of sampled points on circle
        close: if true, first one will be the same as last in circle

    Returns:
        centroid: np(3,), origin of the circle in 3d space
        radius: radius of circle
        normal: np(3,), norm of the triangle
        circle: circle xyz representation in 3d
    """
    assert verts.shape == (3, 3), 'Invalid triangle shape, should be (3, 3)'
    A, B, C = verts[0], verts[1], verts[2]
    AB_mid = 0.5 * (A + B)
    AC_mid = 0.5 * (A + C)
    normal = tri_normal(verts)
    AB = B - A
    AC = C - A

    r1 = np.dot(AB, AB_mid)
    r2 = np.dot(AC, AC_mid)
    r3 = np.dot(A, normal)
    r = np.array([r1, r2, r3])

    rep1 = np.array([AB[0], AC[0], normal[0]])  # (3,)
    rep2 = np.array([AB[1], AC[1], normal[1]])  # (3,)
    rep3 = np.array([AB[2], AC[2], normal[2]])  # (3,)

    mat = np.concatenate([AB[None, :], AC[None, :], normal[None, :]], axis=0)
    det = np.linalg.det(mat)
    if abs(det) < 1e-8:  # on same line
        return None, None, None, None

    m1 = np.concatenate([r[None, :], rep2[None, :], rep3[None, :]], axis=0)
    c1 = np.linalg.det(m1) / det
    m2 = np.concatenate([rep1[None, :], r[None, :], rep3[None, :]], axis=0)
    c2 = np.linalg.det(m2) / det
    m3 = np.concatenate([rep1[None, :], rep2[None, :], r[None, :]], axis=0)
    c3 = np.linalg.det(m3) / det

    centroid = np.array([c1, c2, c3])
    radius = np.linalg.norm([centroid - A])
    assert abs(radius - np.linalg.norm([centroid - B])) < 1e-8 \
           and abs(radius - np.linalg.norm([centroid - C]) < 1e-8), 'Wrong, check it...'

    circle = get_circle(centroid, radius, normal, n_pts, close)

    return centroid, radius, normal, circle


def tri_normal(verts):
    """Get the norm of triangle, y is always positive

    Args:
        verts: np(3, 3), triangle verts, second dim is xyz

    Returns:
        norm: np(3, ) normalize vec
    """
    assert verts.shape == (3, 3), 'Invalid triangle shape, should be (3, 3)'
    ab = verts[1] - verts[0]
    ac = verts[2] - verts[0]
    normal = normalize(np.cross(ab, ac))
    if normal[1] < 0:
        normal *= -1

    return normal


def creat_random_tri(max_range=1.0):
    """Create a random triangle verts in world space

    Args:
        max_range: max verts value range, by default 1

    Returns:
        pts: np(3, 3), triangle verts, second dim is xyz
    """
    return np.random.rand(3, 3) * max_range


def line_from_tri(verts):
    """Create bounding lines from triangle verts by adding first vert to last

        Args:
            verts: np(3, 3), triangle verts, second dim is xyz

        Returns:
            lines: np(4, 3), triangle lines, adding first vert to last
        """
    lines = np.concatenate([verts, verts[:1, :]], axis=0)

    return lines
