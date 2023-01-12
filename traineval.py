from calendar import c
import os
import sys

from tqdm import tqdm
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
from torch.utils import data
import torch.optim as optim

import argparse
import config
from option import MyOptions
from dataset.Dataset import GrabNetDataset

from epochbase import TrainEpoch, ValEpoch
import random
from utils.utils import set_random_seed

from dataset.obj_pretrain_preprocess import ObManObj
from models.ConditionNet import ConditionTrans, ConditionBERT
from traineval_utils.loss import ChamferDistanceL1Loss
from utils.optim import *
from utils.epoch_utils import MetersMonitor, model_update, PretrainEpoch


def train_val(traindataset, trainloader, valdataset, valloader):
    trainer = TrainEpoch(trainloader, traindataset, use_cuda=cfg.use_cuda, cuda_id=cfg.cuda_id)
    # import pdb; pdb.set_trace()
    valer = ValEpoch(valloader, valdataset, mode='val', use_cuda=cfg.use_cuda, cuda_id=cfg.cuda_id)
    # tester = ValEpoch(testloader, mode='test')
    best_val = None

    for epoch in range(cfg.start_epoch - 1, cfg.num_epoch):
        checkpoints, _ = trainer(epoch + 1, best_val=best_val)
        _, best_val = valer(epoch + 1, best_val=best_val, checkpoints=checkpoints)
        torch.cuda.empty_cache()
        
    # tester.epoch(epoch, best_val=best_val)
    # print(f"Done with experiment: {cfg.exp_name}")

    
def pretrain(mode='train', model='trans', bs=8):
    
    model = ConditionBERT if model == 'bert' else ConditionTrans
    net = model(embed_dim=cfg.embed_dim, num_heads=cfg.num_heads, mlp_ratio=cfg.mlp_ratio, glob_feat_dim=cfg.glob_feat_dim, depth={'encoder':cfg.depth, 'decoder':cfg.depth})
    
    if mode == 'train':
        trainset = ObManObj(root=config.OBMAN_ROOT, 
                           shapenet_root=config.SHAPENET_ROOT,
                           split='train',
                           use_cache=True)
        
        valset = ObManObj(root=config.OBMAN_ROOT, 
                           shapenet_root=config.SHAPENET_ROOT,
                           split='val',
                           use_cache=True)
        
        trainloader = data.DataLoader(trainset, batch_size=bs, shuffle=True)
        valloader = data.DataLoader(valset, batch_size=bs, shuffle=False)
        
        optimizer, scheduler = build_optim_sche(net)
        chloss = ChamferDistanceL1Loss()
        
        net = net.to('cuda')
        chloss = chloss.to('cuda')
        
        trainepoch = PretrainEpoch(chloss, optimizer, scheduler, output_dir=cfg.output_dir)
        valepoch = PretrainEpoch(chloss, optimizer, scheduler, output_dir=cfg.output_dir, mode='val', visual_interval=cfg.visual_interval_val)
        
        for epoch in range(cfg.pretrain_epochs):
            net = trainepoch(trainloader, epoch, net)
            _ = valepoch(valloader, epoch, net)
        
    

def evaluation(dataloader, dataset, checkpoint):
    tester = ValEpoch(dataloader, dataset, mode='test', use_cuda=cfg.use_cuda, cuda_id=cfg.cuda_id)
    tester(1, checkpoints=checkpoint)
    
def evaluation_val(args):
    model_folder = os.path.join(cfg.output_root, args.model_exp_name, 'model')
    assert os.listdir(model_folder) is not None, "The given exp has no model saved!"
    
    # CHECK: validate only a part of the saved models
    if args.start_epoch is not None:
        epochs = range(args.start_epoch, cfg.num_epoch+1, cfg.check_interval)
    else:
        epochs = range(cfg.check_interval, cfg.num_epoch+1, cfg.check_interval)
    valdataset = GrabNetDataset(dataset_root=config.DATASET_ROOT, 
                                ds_name="val", 
                                frame_names_file=cfg.frame_names, 
                                grabnet_thumb=True, 
                                obj_meshes_folder=cfg.obj_meshes, 
                                select_ids=True)
    valloader = data.DataLoader(valdataset, batch_size=cfg.batch_size, shuffle=False)
    valer = ValEpoch(valloader, valdataset, mode='test', use_cuda=cfg.use_cuda, cuda_id=cfg.cuda_id)
    
    for epoch in epochs:
        model_path = os.path.join(model_folder, 'checkpoint_'+str(epoch)+'.pth')
        checkpoint = torch.load(model_path)
        valer(epoch, checkpoints=checkpoint)
    
    
    


if __name__ == "__main__":
    import wandb
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--model_exp_name', type=str, default=None)
    parser.add_argument('--start_epoch', type=int, default=None)
    parser.add_argument('--eval_ds', type=str, default=None)
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--pt_mode', type=str, default='train')
    parser.add_argument('--pt_model', type=str, default='trans')

    args = parser.parse_args()

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
                
    print(f"================ {cfg.run_type} experiment running! ================") # NOTE: Checkpoint! 提醒一下当前实验的属性
    if cfg.run_type == 'train':
        traindataset = GrabNetDataset(dataset_root=config.DATASET_ROOT, 
                                  ds_name="train", 
                                  frame_names_file=cfg.frame_names, 
                                  grabnet_thumb=True, 
                                  obj_meshes_folder=cfg.obj_meshes, 
                                  select_ids=cfg.train_select_ids)
        # traindataset = GrabNetDataset(config.dataset_dir, 'train', num_mask=cfg.num_mask)
        trainloader = data.DataLoader(traindataset, batch_size=cfg.batch_size, shuffle=True)

        valdataset = GrabNetDataset(dataset_root=config.DATASET_ROOT, 
                                    ds_name="val", 
                                    frame_names_file=cfg.frame_names, 
                                    grabnet_thumb=True, 
                                    obj_meshes_folder=cfg.obj_meshes, 
                                    select_ids=True)
        valloader = data.DataLoader(valdataset, batch_size=cfg.batch_size, shuffle=False)
        train_val(traindataset, trainloader, valdataset, valloader)
        
    elif cfg.run_type == 'eval_val':
        assert args.model_exp_name is not None, "Requires trained models to evaluate validation set!"
        evaluation_val(args)
        
    elif args.pretrain:
        pretrain(mode=args.pt_mode, model=args.pt_model)
    else:
        assert args.eval_ds is not None, "eval mode requires input the dataset to evaluate!!"
        dataset = GrabNetDataset(dataset_root=config.DATASET_ROOT, 
                                  ds_name=args.eval_ds, 
                                  frame_names_file=cfg.frame_names, 
                                  grabnet_thumb=True, 
                                  obj_meshes_folder=cfg.obj_meshes, 
                                  select_ids=cfg.train_select_ids)
        dataloader = data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)
        if cfg.checkpoint_epoch < 0:
            assert args.checkpoint is not None, "No checkpoint pre-configed! Requires input checkpoint path"
            chk_path = args.checkpoint
        else:
            chk_path = os.path.join(cfg.model_root, f'checkpoint_{cfg.checkpoint_epoch}.pth')
        checkpoint = torch.load(chk_path)
        # import pdb; pdb.set_trace()
        evaluation(dataloader, dataset, checkpoint=checkpoint)

    


