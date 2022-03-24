# -*- coding: utf-8 -*-

import torch.utils.data as data


class BaseDataset(data.Dataset):
    """A base dataset class"""

    def __init__(self, cfgs, data_dir, mode, transforms=None):
        self.cfgs = cfgs
        self.data_dir = data_dir
        self.mode = mode
        self.transforms = transforms

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
