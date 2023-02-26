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
# from option import MyOptions
from dataset.Dataset import GrabNetDataset

from epochbase import TrainEpoch, ValEpoch
import random
from utils.utils import set_random_seed

from dataset.obman_preprocess import ObManObj
from models.pointnet_encoder import ObjRegionConditionEncoder
from models.ConditionNet import ConditionMAE, ConditionMAE_origin, ConditionTrans, ConditionBERT
from models.cGrasp_vae import cGraspvae
from traineval_utils.loss import ChamferDistanceL2Loss, PointCloudCompletionLoss, cGraspvaeLoss
from utils.optim import *
from utils.datasets import get_dataset
from utils.epoch_utils import EpochVAE_mae, EvalEpochVAE_mae,  MetersMonitor, model_update, PretrainEpoch

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
    
# def cgrasp_comp(cfg=None):
#     mode = cfg.run_mode
#     model_type = cfg.model.cnet.type
#     bs = cfg.batch_size
    
#     # load the condition net first
#     model = ConditionBERT if model_type == 'bert' else ConditionTrans
#     cnet = model(**cfg.model.cnet.kwargs)
#     # CHECK: load the checkpoint
#     if cfg.model.cnet.chkpt_path:
#         checkpoint = torch.load(os.path.join(cfg.output_root, cfg.model.cnet.chkpt_path))
#         cnet.load_state_dict(checkpoint['state_dict'])
#         print('checkpoint for cnet loaded!')
        
#     model = cGraspvae(cnet,
#                       **cfg.model.vae.kwargs, cfg=cfg)
    
#     # model.named_parameters()
#     # import pdb; pdb.set_trace()
    
#     if mode == 'train':
#         # DONE: dataset -- obj_points / obj_point_normals / obj_trans / input_pc / hand_verts
#         valset = get_dataset(cfg, mode='val')
#         trainset = get_dataset(cfg, mode='train')
        
        
#         trainloader = data.DataLoader(trainset, batch_size=bs, shuffle=True)
#         valloader = data.DataLoader(valset, batch_size=bs, shuffle=False)
        
#         # TODO: cnet/vae其他参数设置不同学习率
#         optimizer, scheduler = build_optim_sche_grasp(model, part_model={'cnet': model.cnet}, cfg=cfg)
        
#         # TODO: loss改写
#         device = 'cuda' if cfg.use_cuda else 'cpu'
#         model = model.to(device)
#         cgrasp_loss = cGraspvaeLoss(device, cfg)
#         cgrasp_loss.to(device)
        
        
#         trainepoch = EpochVAE_comp(cgrasp_loss, trainset, optimizer, scheduler, output_dir=cfg.output_dir, mode='train', cfg=cfg)
#         valepoch = ValEpochVAE_comp(cgrasp_loss, valset, optimizer, scheduler, output_dir=cfg.output_dir, mode='val', cfg=cfg)
        
#         for epoch in range(cfg.num_epoch):
#             model, _ = trainepoch(trainloader, epoch, model)
#             _, stop_flag = valepoch(valloader, epoch, model)
#             if stop_flag:
#                 print("Early stopping occur!")
#                 break
            
def cgrasp(cfg=None):
    # import pdb; pdb.set_trace()
    mode = cfg.run_mode
    # model_type = cfg.model.cnet.type
    bs = cfg.batch_size
    
    if cfg.mae:
        cnet = ConditionMAE(cfg.model.cnet.kwargs) if not cfg.cmae_orig else ConditionMAE_origin(cfg.model.cnet.kwargs)
        
            
        if cfg.model.cnet.chkpt_path:
            checkpoint = torch.load(os.path.join(cfg.output_root, cfg.model.cnet.chkpt_path))
            # 只载入state_dict = 'MAE_encoder'部分的参数 至 state_dict = 'MAE_encoder'
            # import pdb; pdb.set_trace()
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
    
    part_model_dict = {}
    # TODO: cnet/vae其他参数设置不同学习率
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
                    # optimizer[i].to(device)
            if isinstance(scheduler, list):
                for i, sche_state in enumerate(checkpoint['scheduler']):
                    scheduler[i].load_state_dict(sche_state)
                    # optimizer[i].to(device)
            # scheduler = checkpoint['scheduler']
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resumed the exp from {cfg.chkpt}, start_epoch = {start_epoch}")
        else:
            start_epoch = 0
        
        # DONE: dataset -- obj_points / obj_point_normals / obj_trans / input_pc / hand_verts
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
        
        # optimizer, scheduler = build_optim_sche_grasp(model, part_model={'cnet_mae': model.cnet.MAE_encoder}, cfg=cfg)
        checkpoint = torch.load(os.path.join(cfg.output_dir, 'models', cfg.chkpt+'.pth'))
        model.load_state_dict(checkpoint['state_dict'])
        epoch = checkpoint['epoch']
        print(f"test {cfg.chkpt}")
        
        
        # device = 'cuda' if cfg.use_cuda else 'cpu'
        # model = model.to(device)
        # cgrasp_loss = cGraspvaeLoss(device, cfg)
        # cgrasp_loss.to(device)
        # import pdb; pdb.set_trace()
        
        testepoch = EvalEpochVAE_mae(cgrasp_loss, testset, output_dir=cfg.output_dir, mode='test', cfg=cfg)
        
        _, _ = testepoch(testloader, epoch, model, optimizer, scheduler, save_pred=True)
        
    elif mode == 'val_only':
        valset = get_dataset(cfg, mode='val')
        valloader = data.DataLoader(valset, batch_size=bs, shuffle=False) 
        
        # optimizer, scheduler = build_optim_sche_grasp(model, part_model={'cnet_mae': model.cnet.MAE_encoder}, cfg=cfg)
        
        
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
    import wandb
    import argparse
    from omegaconf import OmegaConf
    from easydict import EasyDict
    import utils.cfgs as cfgsu
    
    # import psutil
    # p = psutil.Process()
    # p.cpu_affinity(range(16))
    # print(p.cpu_affinity())
    
    # import pdb; pdb.set_trace()
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfgs', type=str, default='config')
    parser.add_argument('--cfgs_fodler', type=str, default='cgrasp')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--cuda_id', type=str, default="0")
    parser.add_argument('--machine', type=str, default='41')
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
    
    # parser.add_argument('--no_loss_edge', action='store_true')
    # parser.add_argument('--no_loss_mesh_rec', action='store_true')
    # parser.add_argument('--no_loss_dist_h', action='store_true')
    # parser.add_argument('--no_loss_dist_o', action='store_true')
    # parser.add_argument('--loss_penetr', action='store_false')
    # parser.add_argument('--loss_mano', action='store_false')
    parser.add_argument('--dloss_type', type=str, default=None)
    parser.add_argument('--eval_iter', type=int, default=10)
    parser.add_argument('--cmae_orig', action='store_true')
    parser.add_argument('--use_pos', action='store_true')

    args = parser.parse_args()

    set_random_seed(3407)
    
    # cfg = MyOptions()
    # DONE: 读取配置文件并转化成字典，同时加入args的配置
    
    exp_name = cfgsu.config_exp_name(args)
    paths = cfgsu.config_paths(args.machine, exp_name['exp_name'])
    
    # import pdb; pdb.set_trace()
    
    conf = cfgsu.get_config(args, paths, args.cfgs_fodler)
    conf.update(exp_name)
    conf.update(paths)
    conf.update(args.__dict__) # args的配置也记录下来
    
    cfg = EasyDict()
    # DONE: transform the dict config to easydict
    cfg = cfgsu.merge_new_config(cfg, conf)
    
    # cfg = cfgsu.adjust_config(cfg, args)
    # conf = OmegaConf.structured(cfg)
    # import pdb; pdb.set_trace()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.cuda_id
    os.environ['OMP_NUM_THREAD'] = '1'
    torch.set_num_threads(1)
    
    cfgsu.save_experiment_config(cfg)
    

    if cfg.wandb:
        wandb.login()
        wandb.init(project=cfg.project_name,
                name=cfg.exp_name,
                config=conf,
                dir=os.path.join(cfg.output_root, 'wandb')) # omegaconf: resolve=True即可填写自动变量
                # dir: set the absolute path for storing the metadata of each runs
                
    print(f"================ {cfg.run_type} experiment running! ================") # NOTE: Checkpoint! 提醒一下当前实验的属性
        
    if cfg.run_type == 'cgrasp':
        if cfg.comp:
            cgrasp_comp(cfg=cfg)
        # elif cfg.mae:
        #     cgrasp_mae(cfg=cfg)
        # elif cfg.base:
        #     cgrasp_base(cfg=cfg)
        else:
            cgrasp(cfg=cfg)
    else:
        raise NotImplementedError

    


