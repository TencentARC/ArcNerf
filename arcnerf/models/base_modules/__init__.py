# -*- coding: utf-8 -*-

import importlib
import os.path as osp
from copy import deepcopy

from common.utils.cfgs_utils import valid_key_in_cfgs
from common.utils.file_utils import scan_dir
from common.utils.registry import MODULE_REGISTRY

from .activation import get_activation, Sine
from .embed import Embedder
from .linear_network_module import GeoNet, RadianceNet
from .linear import DenseLayer, SirenLayer
from .dense_volume_module import VolGeoNet, VolRadianceNet

__all__ = [
    'get_activation', 'build_geo_model', 'build_radiance_model', 'Embedder', 'DenseLayer', 'SirenLayer', 'Sine',
    'GeoNet', 'RadianceNet', 'VolGeoNet', 'VolRadianceNet'
]

module_folder = osp.dirname(osp.abspath(__file__))
module_filenames = [osp.splitext(osp.basename(v))[0] for v in scan_dir(module_folder) if v.endswith('_module.py')]
_modules_modules = [
    importlib.import_module(f'arcnerf.models.base_modules.{file_name}') for file_name in module_filenames
]


def build_geo_model(cfgs):
    """Build the geo module from configs.

    Args:
        cfgs (dict): Configuration. It must contain:
            type (str): Module type.
    """
    cfgs = deepcopy(cfgs)

    # default as linear module
    if not valid_key_in_cfgs(cfgs, 'type'):
        geo_model = MODULE_REGISTRY.get('GeoNet')(**cfgs.__dict__)
    else:
        geo_model = MODULE_REGISTRY.get(cfgs.type)(**cfgs.__dict__)

    return geo_model


def build_radiance_model(cfgs):
    """Build the raiance module from configs.

    Args:
        cfgs (dict): Configuration. It must contain:
            type (str): Model type.
    """
    cfgs = deepcopy(cfgs)

    # default as linear module
    if not valid_key_in_cfgs(cfgs, 'type'):
        radiance_model = MODULE_REGISTRY.get('RadianceNet')(**cfgs.__dict__)
    else:
        radiance_model = MODULE_REGISTRY.get(cfgs.type)(**cfgs.__dict__)

    return radiance_model
