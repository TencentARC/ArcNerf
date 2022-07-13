# -*- coding: utf-8 -*-

import importlib
import os.path as osp
from copy import deepcopy

from common.utils.cfgs_utils import dict_to_obj, valid_key_in_cfgs
from common.utils.file_utils import scan_dir
from common.utils.registry import ENCODER_REGISTRY

from .densegrid_encoder import DenseGridEmbedder
from .freq_encoder import FreqEmbedder
from .gaussian_encoder import GaussianEmbedder, Gaussian
from .hashgrid_encoder import HashGridEmbedder
from .sh_encoder import SHEmbedder

__all__ = ['DenseGridEmbedder', 'FreqEmbedder', 'Gaussian', 'GaussianEmbedder', 'HashGridEmbedder', 'SHEmbedder']

encoder_folder = osp.dirname(osp.abspath(__file__))
encoder_filenames = [osp.splitext(osp.basename(v))[0] for v in scan_dir(encoder_folder) if v.endswith('_encoder.py')]
_encoder_modules = [
    importlib.import_module(f'arcnerf.models.base_modules.encoding.{file_name}') for file_name in encoder_filenames
]


def build_encoder(cfgs):
    """Select encoder from cfgs.
    For all the encoder here, it should support to embed any input in (B, input_dim) into higher dimension (B, out)

    Args:
        cfgs: a obj with following required fields:
            input_dim: for (B, input_dim) tensor to get embedding
            n_freqs: embedding freq,
    """
    # default encoder
    if cfgs is None:
        cfgs = dict_to_obj({
            'type': 'FreqEmbedder',
            'input_dim': 3,
            'n_freqs': 0  # by default not encode
        })

    cfgs = deepcopy(cfgs)

    # default as FreqEmbedder
    if not valid_key_in_cfgs(cfgs, 'type'):
        encoder = ENCODER_REGISTRY.get('FreqEmbedder')(**cfgs.__dict__)
    else:
        encoder = ENCODER_REGISTRY.get(cfgs.type)(**cfgs.__dict__)

    return encoder, cfgs.input_dim, cfgs.n_freqs
