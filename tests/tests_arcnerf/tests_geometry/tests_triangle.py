# -*- coding: utf-8 -*-

import os
import os.path as osp
import unittest

import numpy as np

from arcnerf.geometry.triangle import tri_normal, creat_random_tri, circumcircle_from_triangle
from arcnerf.visual.plot_3d import draw_3d_components
from common.visual import get_colors_from_cm
from tests.tests_arcnerf.tests_geometry import TestGeomDict

RESULT_DIR = osp.abspath(osp.join(__file__, '..', 'results', 'triangle'))
os.makedirs(RESULT_DIR, exist_ok=True)


class TestDict(TestGeomDict):

    def setUp(self):
        super().setUp()

    def tests_tri_mesh(self):
        file_path = osp.join(RESULT_DIR, 'tri_mesh.png')
        verts = creat_random_tri()  # (3, 3)
        verts = np.concatenate([verts, np.array([0, 0, 0])[None, :]])  # (4, 3)

        meshes_sep = [verts[0:3, :][None, :], verts[1:4, :][None, :]]
        mesh_colors = get_colors_from_cm(2, to_np=True)

        draw_3d_components(
            lines=[verts[1:3, :]],
            meshes=meshes_sep,
            mesh_colors=mesh_colors,
            title='triangle meshes',
            save_path=file_path
        )

    def tests_tri_normal(self):
        file_path = osp.join(RESULT_DIR, 'tri_norm.png')
        verts = creat_random_tri()  # (3, 3)
        mesh = verts[0:3, :][None, :]  # (1, 3, 3)
        norm = tri_normal(verts)[None, :] / 3.0  # (1, 3)

        draw_3d_components(rays=(verts[:1, :], norm), meshes=[mesh], title='get norm for triangle', save_path=file_path)

    def tests_circumcircle_from_triangle(self):
        file_path = osp.join(RESULT_DIR, 'tri_circumcircle.png')
        verts = creat_random_tri()  # (3, 3)
        mesh = verts[0:3, :][None, :]  # (1, 3, 3)
        centroid, radius, normal, circle = circumcircle_from_triangle(verts)

        draw_3d_components(
            points=centroid[None, :],
            rays=(centroid[None, :], normal[None, :] / 3.0),
            lines=[circle],
            meshes=[mesh],
            title='circumcircle of triangle',
            save_path=file_path
        )


if __name__ == '__main__':
    unittest.main()
