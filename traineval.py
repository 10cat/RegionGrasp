from calendar import c
import os
import sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
from torch.utils import data
import argparse
import config
from option import MyOptions
from dataset.Dataset import GrabNetDataset
from epochbase import TrainEpoch, ValEpoch
import random
from utils.utils import set_random_seed


def train_val():
    traindataset = GrabNetDataset(config.dataset_dir, 'train', num_mask=cfg.num_mask)
    trainloader = data.DataLoader(traindataset, batch_size=cfg.batch_size, shuffle=True)

    valdataset = GrabNetDataset(config.dataset_dir, 'val', num_mask=cfg.num_mask)
    valloader = data.DataLoader(valdataset, batch_size=cfg.batch_size, shuffle=False)

    trainer = TrainEpoch(trainloader, traindataset, use_cuda=cfg.use_cuda, cuda_id=cfg.cuda_id)
    # import pdb; pdb.set_trace()
    valer = ValEpoch(valloader, valdataset, mode='val', use_cuda=cfg.use_cuda, cuda_id=cfg.cuda_id)
    # tester = ValEpoch(testloader, mode='test')
    best_val = None

    for epoch in range(cfg.start_epoch - 1, cfg.num_epoch):
        checkpoints, _ = trainer.one_epoch(epoch + 1, best_val=best_val)
        _, best_val = valer.one_epoch(epoch + 1, best_val=best_val, checkpoints=checkpoints)
        torch.cuda.empty_cache()
        
    
    # tester.epoch(epoch, best_val=best_val)
    # print(f"Done with experiment: {cfg.exp_name}")

def evaluation(testloader):
    testdataset = GrabNetDataset(config.dataset_dir, 'test', num_mask=cfg.num_mask)
    testloader = data.DataLoader(testdataset, batch_size=cfg.batch_size, shuffle=False)
    ValEpoch(testloader, testdataset)


if __name__ == "__main__":
    import wandb
    from omegaconf import OmegaConf

    set_random_seed(1024)

    cfg = MyOptions()

    conf = OmegaConf.structured(cfg)
    # import pdb; pdb.set_trace()

    if cfg.w_wandb:
        wandb.login()

        wandb.init(project="ConditionHOI",
                name=cfg.exp_name,
                config=OmegaConf.to_container(conf, resolve=True),
                dir=os.path.join(cfg.output_root, 'wandb')) # omegaconf: resolve=True即可填写自动变量
                # dir: set the absolute path for storing the metadata of each runs
    if cfg.mode == 'train':
        train_val()
    else:
        evaluation()

    


