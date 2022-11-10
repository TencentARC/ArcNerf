# -*- coding: utf-8 -*-

import importlib
import os.path as osp
from copy import deepcopy

from common.utils.cfgs_utils import valid_key_in_cfgs, get_value_from_cfgs_field
from common.utils.file_utils import scan_dir
from common.utils.registry import BOUND_REGISTRY

from .basic_bound import BasicBound
from .bitfield_bound import BitfieldBound
from .sphere_bound import SphereBound
from .volume_bound import VolumeBound

__all__ = ['BasicBound', 'BitfieldBound', 'SphereBound', 'VolumeBound']

bound_folder = osp.dirname(osp.abspath(__file__))
bound_filenames = [osp.splitext(osp.basename(v))[0] for v in scan_dir(bound_folder) if v.endswith('_bound.py')]
_bound_modules = [
    importlib.import_module(f'arcnerf.models.base_modules.obj_bound.{file_name}') for file_name in bound_filenames
]


def build_obj_bound(cfgs):
    """Select object bound from cfgs.
    For all the bound here, constrain the point sampling position

    Args:
        cfgs: select from cfgs.obj_bound field
    """
    if not valid_key_in_cfgs(cfgs, 'obj_bound'):
        key = 'basic'
        obj_bound = BOUND_REGISTRY.get(key_to_bound_type(key))(None)
    else:
        keys = get_value_from_cfgs_field(cfgs, 'obj_bound').__dict__.keys()
        if 'volume' in keys:
            key = 'volume'
        elif 'sphere' in keys:
            key = 'sphere'
        elif 'bitfield' in keys:
            key = 'bitfield'
        else:
            raise NotImplementedError('Not such bounding class {}...'.format(keys))

        bound_cfgs = deepcopy(cfgs.obj_bound)
        obj_bound = BOUND_REGISTRY.get(key_to_bound_type(key))(bound_cfgs)

    obj_bound_type = key if key != 'basic' else key

    return obj_bound, obj_bound_type


def key_to_bound_type(key):
    bound_dict = {
        'basic': 'BasicBound',
        'bitfield': 'BitfieldBound',
        'volume': 'VolumeBound',
        'sphere': 'SphereBound',
    }

    return bound_dict[key]
