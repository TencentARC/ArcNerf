# -*- coding: utf-8 -*-

import numpy as np

from common.datasets.base_dataset import BaseDataset
from common.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class DummyConv(BaseDataset):
    """Dummpy Conv dataset for conv network"""

    def __init__(self, cfgs, data_dir, mode, transforms):
        super(DummyConv, self).__init__(cfgs, data_dir, mode, transforms)
        self.data = [np.ones(shape=(10, 10, 3), dtype=np.float32) * 127.0 for _ in range(1000)]
        self.gt = [np.ones(shape=(1, 10, 10), dtype=np.float32) for _ in range(1000)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """For dumpy numpy array, then will be extent and change to [b, xxx]
        shape when combined into torch tensor
        """
        data = {'img': self.data[idx], 'gt': self.gt[idx]}

        if self.transforms:
            data = self.transforms(data)

        return data
