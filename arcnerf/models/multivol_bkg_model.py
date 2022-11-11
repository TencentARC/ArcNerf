# -*- coding: utf-8 -*-

from arcnerf.geometry.volume import Volume
from common.utils.cfgs_utils import get_value_from_cfgs_field
from common.utils.registry import MODEL_REGISTRY
from .bkg_model import BkgModel


@MODEL_REGISTRY.register()
class MultiVol(BkgModel):
    """ Multi-Volume model with several resolution. It is the one used in instant-ngp,
        but inner volume is removed
    """

    def __init__(self, cfgs):
        super(MultiVol, self).__init__(cfgs)

        self.cfgs = cfgs
        self.read_optim_cfgs()

        # volume setting
        vol_cfgs = self.cfgs.basic_volume
        if get_value_from_cfgs_field(vol_cfgs, 'n_grid') is None:
            vol_cfgs.n_grid = 128
        self.n_cascade = self.vol_cfgs.n_cascade
        self.basic_volume = Volume(**self.vol_cfgs.__dict__)

    def read_optim_cfgs(self):
        """Read optim params under model.obj_bound. Prams controls optimization"""
        params = {
            'near_distance': get_value_from_cfgs_field(self.cfgs.optim, 'near_distance', 0.0),
            'epoch_optim': get_value_from_cfgs_field(self.cfgs, 'epoch_optim', 16),  # You must optimize the volume
            'ema_optim_decay': get_value_from_cfgs_field(self.cfgs, 'ema_optim_decay', 0.95),
            'opa_thres': get_value_from_cfgs_field(self.cfgs, 'opa_thres', 0.01)
        }

        return params

    def get_optim_cfgs(self, key=None):
        """Get optim cfgs by optional key"""
        if key is None:
            return self.optim_cfgs

        return self.optim_cfgs[key]

    def set_optim_cfgs(self, key, value):
        """Set optim cfgs by key"""
        self.optim_cfgs[key] = value
