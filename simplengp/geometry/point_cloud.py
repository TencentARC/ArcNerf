# -*- coding: utf-8 -*-

import numpy as np
import trimesh


def save_point_cloud(pc_file, pts, color=None):
    """Save point cloud

    Args:
        pc_file: file name of pc, in .ply
        pts: point cloud xyz, (n_pts, 3)
        color: rgb color in (0-1), optional
    """
    colors_uint8 = (255.0 * color.copy()).astype(np.uint8) if color is not None else None
    pc = trimesh.points.PointCloud(pts, colors_uint8)
    pc.export(pc_file)


def load_point_cloud(pc_file):
    """Load point cloud

    Args:
        pc_file: file name of pc, in .ply

    Returns:
        pts: point cloud xyz, (n_pts, 3), numpy array
    """
    pts = trimesh.load(pc_file)
    pts = np.array(pts.vertices)

    return pts
