# -*- coding: utf-8 -*-


class LossDictCounter(object):
    """A counter for averaging all the losses by batch in a loader
    Input loss_dict should have 'names', 'sum' and other fields accessible by names
    """

    def __init__(self):
        super(LossDictCounter, self).__init__()
        self.summary = None
        self.count = 0
        self.avg_summary = None

    def get_summary(self):
        """Get summary"""
        return self.summary

    def get_count(self):
        """Get total num of sample"""
        return self.count

    def reset(self):
        """Reset all to init mode"""
        self.summary = None
        self.count = 0
        self.avg_summary = None

    def __call__(self, loss, batch_size):
        """Write loss to summary."""
        if self.summary is None:
            self.summary = {}
            for name in loss['names']:
                self.summary[name] = float(loss[name]) * batch_size
            self.summary['sum'] = float(loss['sum']) * batch_size
            self.summary['names'] = loss['names']

        else:
            for name in loss['names']:
                self.summary[name] += float(loss[name]) * batch_size
            self.summary['sum'] += float(loss['sum']) * batch_size

        self.count += batch_size

    def get_avg_summary(self):
        """Mean by average"""
        return self.avg_summary

    def get_avg_sum(self):
        """Sum Mean by average"""
        return self.avg_summary['sum']

    def cal_average(self):
        """get mean summary"""
        self.avg_summary = {}
        for k in self.summary['names']:
            self.avg_summary[k] = self.summary[k] / float(self.count)
        self.avg_summary['sum'] = self.summary['sum'] / float(self.count)
        self.avg_summary['names'] = self.summary['names']
