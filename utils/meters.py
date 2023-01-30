import os
import numpy as np
to_gpu = lambda tensor: tensor.cuda()
to_cpu = lambda tensor: tensor.detach().cpu()
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val # latest added value
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count

    def average(self):
        return self.avg

class AverageMeters:
    def __init__(self):
        super().__init__()
        self.average_meters = {}

    def add_value(self, name, loss_val, n=1):
        if name not in self.average_meters:
            self.average_meters[name] = AverageMeter()
        self.average_meters[name].update(loss_val, n=n)

    def update(self, meters):
        self.average_meters.update(meters.average_meters)

    def avg(self, mode):
        return {(mode+'_' +name): avg_meter.avg for name, avg_meter  in self.average_meters.items()}
    
    
