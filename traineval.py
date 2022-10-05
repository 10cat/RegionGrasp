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

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"


def train_val(trainloader, valloader, testloader):

    for epoch in range(cfg.num_epochs):
        TrainEpoch(trainloader, epoch)
        ValEpoch(valloader, epoch)

def evaluation(testloader):
    ValEpoch(testloader)


if __name__ == "__main__":
    import wandb
    from omegaconf import OmegaConf

    cfg = MyOptions()

    conf = OmegaConf.structured(cfg)

    wandb.login()

    wandb.init(project="ConditionHOI",
               name=cfg.exp_name,
               config=OmegaConf.to_container(conf, resolve=True)) # omegaconf: resolve=True即可填写自动变量

    if cfg.mode == 'train':
        traindataset = GrabNetDataset(config.dataset_dir, 'train', num_mask=cfg.num_mask)
        trainloader = data.DataLoader(traindataset, batch_size=cfg.batch_size, shuffle=True)

        valdataset = GrabNetDataset(config.dataset_dir, 'val', num_mask=cfg.num_mask)
        valloader = data.DataLoader(valdataset, batch_size=cfg.batch_size, shuffle=False)

    testdataset = GrabNetDataset(config.dataset_dir, 'test', num_mask=cfg.num_mask)
    testloader = data.DataLoader(testdataset, batch_size=cfg.batch_size, shuffle=False)

    if cfg.mode == 'train':
        train_val(trainloader, valloader, testloader, cfg)
    else:
        evaluation(testloader)

    


