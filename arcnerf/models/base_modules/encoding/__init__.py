# -*- coding: utf-8 -*-

from .freq_encoder import FreqEmbedder
from .sh_encoder import SHEmbedder

from common.utils.cfgs_utils import get_value_from_cfgs_field, dict_to_obj


def get_encoder(cfgs):
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

    encoder_type = get_value_from_cfgs_field(cfgs, 'type', 'FreqEmbedder')

    if encoder_type == 'FreqEmbedder':
        encoder = FreqEmbedder(**cfgs.__dict__)
    elif encoder_type == 'SHEmbedder':
        encoder = SHEmbedder(**cfgs.__dict__)
    else:
        raise NotImplementedError('Invalid embeder {}'.format(encoder_type))

    return encoder, cfgs.input_dim, cfgs.n_freqs
