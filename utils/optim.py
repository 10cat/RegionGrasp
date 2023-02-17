import os
import sys
sys.path.append('.')
sys.path.append('..')
import torch
import torch.nn as nn
import torch.optim as optim

from timm.scheduler import CosineLRScheduler

# from option import MyOptions as cfg


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
    
def add_weight_decay(named_params, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in named_params:
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

def build_lambda_sche(optimizer, cfg=None):
    if cfg.get('decay_step') is not None:
        warming_up_t = getattr(cfg, 'warmingup_e', 0) # NOTE: warmup set by 'warmingup_e' in cfg
        lr_lbmd = lambda e: max(cfg.lr_decay ** (e / cfg.decay_step), cfg.lowest_decay) if e >= warming_up_t else max(e / warming_up_t, 0.001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lbmd)
    else:
        raise NotImplementedError()
    return scheduler

def build_lambda_bnsche(model, cfg=None):
    if cfg.get('decay_step') is not None:
        bnm_lmbd = lambda e: max(cfg.bn_momentum * cfg.bn_decay ** (e / cfg.decay_step), cfg.lowest_decay)
        bnm_scheduler = BNMomentumScheduler(model, bnm_lmbd)
    else:
        raise NotImplementedError()
    return bnm_scheduler

def build_optim_sche(model, cfg=None):
    opt_cfg = cfg.optimizer
    optimizer = build_optim(opt_cfg, model.named_parameters())
        
    scheduler = build_scheduler(optimizer, cfg, model=model)
    return optimizer, scheduler

def build_optim(opt_cfg, named_parameters=None, param_groups=None):
    if opt_cfg.type == 'AdamW':
        # NOTE: add weight decay for AdamW
        param_groups = add_weight_decay(named_parameters, weight_decay=opt_cfg.kwargs.weight_decay)
        optimizer = optim.AdamW(param_groups, **opt_cfg.kwargs)
    elif opt_cfg.type == 'Adam':
        assert param_groups is not None
        optimizer = optim.Adam(param_groups, **opt_cfg.kwargs)
        
    return optimizer

def build_scheduler(optimizer, cfg, model=None):
    sche_cfg = cfg.scheduler
    if sche_cfg.type == 'LambdaLR':
            scheduler = build_lambda_sche(optimizer, sche_cfg.kwargs)
    elif sche_cfg.type == 'CosLR':
        scheduler = CosineLRScheduler(optimizer,
                t_initial=sche_cfg.kwargs.epochs,
                cycle_mul=1,
                cycle_limit=1,
                lr_min=1e-6,
                cycle_decay=0.1,
                # decay_rate=0.1,
                warmup_lr_init=1e-6,
                warmup_t=sche_cfg.kwargs.initial_epochs,
                t_in_epochs=True)
    elif sche_cfg.type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **sche_cfg.kwargs)
    elif sche_cfg.type == 'function':
        scheduler = None
    else:
        raise NotImplementedError()
    if cfg.get('bnmscheduler') is not None:
        bnsche_config = cfg.bnmscheduler
        if bnsche_config.type == 'Lambda':
            bnscheduler = build_lambda_bnsche(model, bnsche_config.kwargs)
        scheduler = [scheduler, bnscheduler]
    return scheduler
        
def build_optim_sche_grasp(full_model, part_model = None, cfg=None):
    optimizer, scheduler = [], []
    if part_model is not None:
        part_params_all = []
        part_params_named_all = []
        for name, part in part_model.items():
            part_cfg = cfg.optim[name]
            part_model_params = part.parameters()
            part_model_params_named = part.named_parameters()
            part_params_all += part_model_params
            part_params_named_all += part_model_params_named
    
            optimizer_part = build_optim(part_cfg.optimizer, named_parameters=part_model_params_named, param_groups=part_model_params)
            
            optimizer.append(optimizer_part)
            
            if part_cfg.get('scheduler'):
                scheduler_part = build_scheduler(optimizer_part, part_cfg)
                scheduler.append(scheduler_part)
        part_model_params_id = list(map(id, part_params_all))
        base_params = filter(lambda p: id(p) not in part_model_params_id, full_model.parameters())
        part_model_named_params_id = list(map(id, part_params_named_all))
        base_named_params = filter(lambda p: id(p) not in part_model_named_params_id, full_model.named_parameters())
        print("Multiple optimizers / schedulers loaded")
        
    else:
        assert cfg.optim.get('others')
        base_params = full_model.parameters()
        base_named_params = full_model.named_parameters()
    
    optimizer_other = build_optim(cfg.optim.others.optimizer, named_parameters=base_named_params, param_groups=base_params)
    
    if cfg.optim.others.get('scheduler'):
        scheduler_others = build_scheduler(optimizer_other, cfg.optim.others)
    else:
        scheduler_others = None
        
    if not optimizer:
        optimizer = optimizer_other
        scheduler = scheduler_others
    else:
        optimizer.append(optimizer_other)
        if not scheduler:
            scheduler.append(scheduler_others)
        else:
            scheduler = scheduler_others
    return optimizer, scheduler
    
    
    