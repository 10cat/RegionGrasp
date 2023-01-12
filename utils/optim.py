import os
import sys
sys.path.append('.')
sys.path.append('..')
import torch
import torch.nn as nn
import torch.optim as optim

from option import MyOptions as cfg


def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum
    return fn


class BNMomentumScheduler(object):
    
    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(
                    type(model).__name__
                )
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

    def get_momentum(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        return self.lmbd(epoch)

def build_lambda_sche(optimizer):
    lr_lbmd = lambda e: max(cfg.lr_decay ** (e / cfg.decay_step), cfg.lowest_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lbmd)
    return scheduler

def build_lambda_bnsche(model):
    bnm_lmbd = lambda e: max(cfg.bnm_momentum * cfg.bnm_lr_decay ** (e / cfg.bnm_decay_step), cfg.bnm_lowest_decay)
    bnm_scheduler = BNMomentumScheduler(model, bnm_lmbd)
    return bnm_scheduler

def build_optim_sche(model):
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr_pt, weight_decay=cfg.weight_decay_pt)
    scheduler = build_lambda_sche(optimizer)
    bnscheduler = build_lambda_bnsche(model)
    scheduler = [scheduler, bnscheduler]
    return optimizer, scheduler