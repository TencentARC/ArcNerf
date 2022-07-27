# -*- coding: utf-8 -*-

import torch


class DataPrefetcher(object):
    """Data loader prefetcher. But the benefit is limited from this. Just keep it but no use."""

    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.batch = None
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return

        with torch.cuda.stream(self.stream):
            for k in self.batch:
                if k == 'img':
                    self.batch[k] = self.batch[k].cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()

        return batch
