# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from common.utils.cfgs_utils import obj_to_dict
from . import ENCODER_REGISTRY


@ENCODER_REGISTRY.register()
class CompositeEmbedder(nn.Module):
    """A type of encoder that used multiple kinds of embedder
    """

    def __init__(self, sub_encoder_types, *args, **kwargs):
        super(CompositeEmbedder, self).__init__()

        self.encoders = []
        self.out_dim = 0
        for enc_idx, _ in enumerate(sub_encoder_types):
            enc_str = 'sub_encoder{}'.format(enc_idx + 1)
            assert enc_str in kwargs.keys(), 'You must have {} in cfgs...'.format(enc_str)
            enc_cfgs = obj_to_dict(kwargs[enc_str])

            # next input will be last output
            if enc_idx == 0:
                enc_cfgs['input_dim'] = kwargs['input_dim']
            else:
                enc_cfgs['input_dim'] = self.out_dim

            enc = ENCODER_REGISTRY.get(enc_cfgs['type'])(**enc_cfgs)
            self.encoders.append(enc)
            self.out_dim = enc.get_output_dim()

    def get_output_dim(self):
        """Get output dim"""
        return self.out_dim

    def forward(self, xyz: torch.Tensor):
        out = xyz
        for enc in self.encoders:
            out = enc(out)

        return out
