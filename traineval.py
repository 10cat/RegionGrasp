import os
import sys

from tqdm import tqdm

sys.path.append('.')
sys.path.append('..')
import argparse
import random

import config
import numpy as np
import torch
import torch.optim as optim
# from option import MyOptions
from dataset.Dataset import GrabNetDataset
from dataset.obman_preprocess import ObManObj
from epochbase import TrainEpoch, ValEpoch
from GraspTTA.network.cmapnet_objhand import pointnet_reg
from models.cGrasp_vae import cGraspvae
from models.ConditionNet import (ConditionBERT, ConditionMAE,
                                 ConditionMAE_origin, ConditionTrans)
from models.pointnet_encoder import ObjRegionConditionEncoder
from torch.utils import data
from traineval_utils.loss import (ChamferDistanceL2Loss,
                                  PointCloudCompletionLoss, cGraspvaeLoss)
from utils.datasets import get_dataset
from utils.epoch_utils import (EpochVAE_mae, EvalEpochVAE_mae, MetersMonitor,
                               PretrainEpoch, model_update)
from utils.optim import *
from utils.utils import set_random_seed


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
def cgrasp(cfg=None):
    # import pdb; pdb.set_trace()
    mode = cfg.run_mode
    # model_type = cfg.model.cnet.type
    bs = cfg.batch_size
    
    if cfg.mae:
        cnet = ConditionMAE(cfg.model.cnet.kwargs) if not cfg.cmae_orig else ConditionMAE_origin(cfg.model.cnet.kwargs)
        
            
        if cfg.model.cnet.chkpt_path:
            checkpoint = torch.load(os.path.join(cfg.output_root, cfg.model.cnet.chkpt_path))
            mae_state_dict = {}
            for key, param in checkpoint['state_dict'].items():
                if key.startswith('MAE_encoder.'):
                    key = key.replace('MAE_encoder.', '')
                    mae_state_dict[key] = param
            cnet.MAE_encoder.load_state_dict(mae_state_dict)
            print('checkpoint for MAE_encoder in cnet loaded!')
    elif cfg.base:
        cnet = ObjRegionConditionEncoder(config = cfg.model.cnet.kwargs)
    
    model = cGraspvae(cnet,
                      **cfg.model.vae.kwargs, cfg=cfg)
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    print(f"the parameter size is {param_size}MB")
    
    part_model_dict = {}
    if 'hand_encoder' in cfg.optim.keys():
        part_model_dict.update({'hand_encoder':model.hand_encoder})
    if 'cnet_mae' in cfg.optim.keys():
        part_model_dict.update({'cnet_mae': model.cnet.MAE_encoder})
    optimizer, scheduler = build_optim_sche_grasp(model, part_model=part_model_dict, cfg=cfg)
    
    device = 'cuda' if cfg.use_cuda else 'cpu'
    model = model.to(device)
    cgrasp_loss = cGraspvaeLoss(device, cfg)
    cgrasp_loss.to(device)
    
    if mode == 'train':
        
        if cfg.resume:
            assert cfg.chkpt is not None, "Checkpoint not configured!"
            
            checkpoint = torch.load(os.path.join(cfg.output_dir, 'models', cfg.chkpt+'.pth'))
            model.load_state_dict(checkpoint['state_dict'])
            if isinstance(optimizer, list):
                for i, optim_state in enumerate(checkpoint['optimizer']):
                    optimizer[i].load_state_dict(optim_state)
            if isinstance(scheduler, list):
                for i, sche_state in enumerate(checkpoint['scheduler']):
                    scheduler[i].load_state_dict(sche_state)
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resumed the exp from {cfg.chkpt}, start_epoch = {start_epoch}")
        else:
            start_epoch = 0
        
        valset = get_dataset(cfg, mode='val')
        trainset = get_dataset(cfg, mode='train')
        
        
        trainloader = data.DataLoader(trainset, batch_size=bs, shuffle=True)
        valloader = data.DataLoader(valset, batch_size=bs, shuffle=False)
        
        
        trainepoch = EpochVAE_mae(cgrasp_loss, trainset, output_dir=cfg.output_dir, mode='train', cfg=cfg)
        valepoch = EpochVAE_mae(cgrasp_loss, valset, output_dir=cfg.output_dir, mode='val', cfg=cfg)
        
        for epoch in range(start_epoch, cfg.num_epoch):
            set_random_seed(3407 + epoch)
            model, optimizer, scheduler, _ = trainepoch(trainloader, epoch, model, optimizer, scheduler)
            # set_random_seed(3407)
            _, _, _, stop_flag = valepoch(valloader, epoch, model, optimizer, scheduler)
            if stop_flag:
                print("Early stopping occur!")
                break
            
    elif mode == 'test':
        testset = get_dataset(cfg, mode=cfg.eval_ds)
        testloader = data.DataLoader(testset, batch_size=bs, shuffle=False)
        
        cmap_model = pointnet_reg(with_rgb=False)
        checkpoint_cmap = torch.load('../tta_checkpoints/model_cmap_best.pth', map_location=torch.device('cpu'))['network']
        cmap_model.load_state_dict(checkpoint_cmap)
        cmap_model = cmap_model.to(device)
        
        # optimizer, scheduler = build_optim_sche_grasp(model, part_model={'cnet_mae': model.cnet.MAE_encoder}, cfg=cfg)
        # import pdb; pdb.set_trace()
        # checkpoint = torch.load(os.path.join(cfg.output_dir, 'models', cfg.chkpt+'.pth'))
        chkpt_path = os.path.join(cfg.model_root, f'checkpoint_98.pth')
        checkpoint = torch.load(chkpt_path)
        model.load_state_dict(checkpoint['state_dict'])
        epoch = checkpoint['epoch']
        print(f"test {cfg.chkpt}")
        
        
        kwargs_cmap = cmap_model if cfg.refine else None
        testepoch = EvalEpochVAE_mae(cgrasp_loss, testset, output_dir=cfg.output_dir, mode='test', cfg=cfg)
        
        _, _ = testepoch(testloader, epoch, model, optimizer, scheduler, save_pred=True, cmap_model=kwargs_cmap)
        
    elif mode == 'val_only':
        valset = get_dataset(cfg, mode='val')
        valloader = data.DataLoader(valset, batch_size=bs, shuffle=False) 
        
        device = 'cuda' if cfg.use_cuda else 'cpu'
        model = model.to(device)
        cgrasp_loss = cGraspvaeLoss(device, cfg)
        cgrasp_loss.to(device)
        
        valepoch = EvalEpochVAE_mae(cgrasp_loss, valset, optimizer, scheduler, output_dir=cfg.output_dir, mode='val', cfg=cfg)
        
        
        interval = cfg.check_interval
        epoch = interval
        ckpt_path = os.path.join(cfg.model_root, f'checkpoint_{epoch}.pth')
        
        # model_cpu = model
        while os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt['state_dict'])
            _, _ = valepoch(valloader, epoch, model)
            
            epoch += interval
            ckpt_path = os.path.join(cfg.model_root, f'checkpoint_{epoch}.pth')
            # import pdb; pdb.set_trace()
            
    else:
        raise NotImplementedError()

            

if __name__ == "__main__":
    import argparse

    import utils.cfgs as cfgsu
    import wandb
    from easydict import EasyDict
    from omegaconf import OmegaConf
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfgs', type=str, default='config')
    parser.add_argument('--cfgs_fodler', type=str, default='cgrasp')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--cuda_id', type=str, default="0")
    parser.add_argument('--machine', type=str, default='97')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--resume', action='store_true')
    
    
    parser.add_argument('--mae', action='store_true')
    parser.add_argument('--comp', action='store_true')
    parser.add_argument('--base', action='store_true')

    parser.add_argument('--chkpt', type=str, default=None)
    
    parser.add_argument('--start_epoch', type=int, default=None)
    parser.add_argument('--eval_ds', type=str, default='test')
    parser.add_argument('--batch_intv', type=str, default=1)
    parser.add_argument('--sample_intv', type=str, default=None)
    parser.add_argument('--run_mode', type=str, default='train')
    parser.add_argument('--pt_model', type=str, default='trans')
    parser.add_argument('--run_check', action='store_true')
    parser.add_argument('--no_save', action='store_true')
    
    parser.add_argument('--dloss_type', type=str, default=None)
    parser.add_argument('--eval_iter', type=int, default=10)
    parser.add_argument('--cmae_orig', action='store_true')
    parser.add_argument('--use_pos', action='store_true')
    parser.add_argument('--refine', action='store_true')
    parser.add_argument('--rand', action='store_true')
    parser.add_argument('--rand_id', type=int, default=None)
    parser.add_argument('--tta', action='store_true')
    parser.add_argument('--grabnet', action='store_true')
    parser.add_argument('--grabnet_rnum', type=int, default=2048)
    parser.add_argument('--region_rand', action='store_true')

    args = parser.parse_args()

    set_random_seed(3407)
    
    exp_name = cfgsu.config_exp_name(args)
    paths = cfgsu.config_paths(args.machine, exp_name['exp_name'])
    
    
    conf = cfgsu.get_config(args, paths, args.cfgs_fodler)
    conf.update(exp_name)
    conf.update(paths)
    conf.update(args.__dict__) 
    
    cfg = EasyDict()
    # DONE: transform the dict config to easydict
    cfg = cfgsu.merge_new_config(cfg, conf)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.cuda_id
    os.environ['OMP_NUM_THREAD'] = '1'
    torch.set_num_threads(1)
    
    cfgsu.save_experiment_config(cfg)
    
    if cfg.wandb:
        wandb.login()
        wandb.init(project=cfg.project_name,
                name=cfg.exp_name,
                config=conf,
                dir=os.path.join(cfg.output_root, 'wandb')) # omegaconf: resolve=True can be applied to config the value

    print(f"================ {cfg.run_type} experiment running! ================") 
        
    if cfg.run_type == 'cgrasp':
        cgrasp(cfg=cfg)
    else:
        raise NotImplementedError

    


