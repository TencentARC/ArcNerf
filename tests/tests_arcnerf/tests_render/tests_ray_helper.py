#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import unittest

import numpy as np

from arcnerf.geometry import np_wrapper
from arcnerf.render.ray_helper import get_zvals_from_near_far
from common.visual.plot_2d import draw_2d_components

RESULT_DIR = osp.abspath(osp.join(__file__, '..', 'results', 'ray_helper'))
os.makedirs(RESULT_DIR, exist_ok=True)


class TestDict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.n_rays = 1
        cls.n_pts = 64
        cls.near = 2.0
        cls.far = 6.0

    def tests_get_zvals_from_near_far(self):
        near = np.ones((self.n_rays, 1)) * self.near
        far = np.ones((self.n_rays, 1)) * self.far
        # choices
        inclusive = [True, False]
        inverse_linear = [True, False]
        perturb = [True, False]
        # show lines
        legends = []
        points = []
        lines = []
        count = 1
        for i1 in inclusive:
            for i2 in inverse_linear:
                for p in perturb:
                    zvals = np_wrapper(get_zvals_from_near_far, near, far, self.n_pts, i1, i2, p)[0].tolist()
                    y = [count] * len(zvals)
                    points.append([zvals, y])
                    lines.append([zvals, y])
                    legends.append('Inclusive: {} - Inverse: {} - Perturb{}'.format(i1, i2, p))
                    count += 1

        file_path = osp.join(RESULT_DIR, 'get_zvals_from_near_far.png')
        draw_2d_components(
            points=points,
            lines=lines,
            legends=legends,
            xlabel='zvals',
            ylabel='',
            title='zvals by different method',
            save_path=file_path
        )
