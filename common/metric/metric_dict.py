# -*- coding: utf-8 -*-


class MetricDictCounter(object):
    """A counter for averaging all the metric by batch in a loader
    Input metric_dict should have 'names' and other fields accessible by names
    """

    def __init__(self):
        super(MetricDictCounter, self).__init__()
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

    def __call__(self, metric, batch_size):
        """Write loss to summary."""
        if self.summary is None:
            self.summary = {}
            for name in metric['names']:
                self.summary[name] = float(metric[name]) * batch_size
            self.summary['names'] = metric['names']

        else:
            for name in metric['names']:
                self.summary[name] += float(metric[name]) * batch_size

        self.count += batch_size

    def get_avg_summary(self):
        """Mean by average"""
        return self.avg_summary

    def cal_average(self):
        """cal metrics averages"""
        self.avg_summary = {}
        for k in self.summary['names']:
            self.avg_summary[k] = self.summary[k] / float(self.count)
        self.avg_summary['names'] = self.summary['names']

    def get_metric_info(self):
        """Return avg metric info in str"""
        if self.avg_summary is None:
            return None

        metric_info = 'Num of samples eval {}\n'.format(self.count)
        for name in self.avg_summary['names']:
            metric_info += '    Metric - [{}]: {:.02f} \n'.format(name, self.avg_summary[name])

        return metric_info
