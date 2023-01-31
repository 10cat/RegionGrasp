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
from models.ConditionNet import ConditionTrans, ConditionBERT
from models.cGrasp_vae import cGraspvae
from traineval_utils.loss import ChamferDistanceL2Loss, PointCloudCompletionLoss, cGraspvaeLoss
from utils.optim import *
from utils.datasets import get_dataset
from utils.epoch_utils import MetersMonitor, model_update, PretrainEpoch, EpochVAE, ValEpochVAE


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
    
def cgrasp(cfg=None):
    mode = cfg.run_mode
    cnet_type = cfg.model.cnet.type
    bs = cfg.batch_size
    
    # load the condition net first
    model = ConditionBERT if cnet_type == 'bert' else ConditionTrans
    cnet = model(**cfg.model.cnet.kwargs)
    # CHECK: load the checkpoint
    if cfg.model.cnet.chkpt_path:
        checkpoint = torch.load(cfg.model.cnet.chkpt_path)
        cnet.load_state_dict(checkpoint['state_dict'])
        print('checkpoint for cnet loaded!')
        
    model = cGraspvae(cnet,
                      **cfg.model.vae.kwargs, cfg=cfg)
    
    # model.named_parameters()
    # import pdb; pdb.set_trace()
    
    if mode == 'train':
        # DONE: dataset -- obj_points / obj_point_normals / obj_trans / input_pc / hand_verts
        valset = get_dataset(cfg, mode='val')
        valset.__getitem__(0)
        trainset = get_dataset(cfg, mode='train')
        
        
        trainloader = data.DataLoader(trainset, batch_size=bs, shuffle=True)
        valloader = data.DataLoader(valset, batch_size=bs, shuffle=False)
        
        # TODO: cnet/vae其他参数设置不同学习率
        optimizer, scheduler = build_optim_sche_grasp(model, cfg=cfg)
        
        # TODO: loss改写
        device = 'cuda' if cfg.use_cuda else 'cpu'
        model = model.to(device)
        cgrasp_loss = cGraspvaeLoss(device, cfg)
        cgrasp_loss.to(device)
        
        
        trainepoch = EpochVAE(cgrasp_loss, trainset, optimizer, scheduler, output_dir=cfg.output_dir, mode='train', cfg=cfg)
        valepoch = ValEpochVAE(cgrasp_loss, trainset, optimizer, scheduler, output_dir=cfg.output_dir, mode='val', cfg=cfg)
        
        for epoch in range(cfg.num_epoch):
            model, _ = trainepoch(trainloader, epoch, model)
            _, stop_flag = valepoch(valloader, epoch, model)
            if stop_flag:
                print("Early stopping occur!")
                break
    
    
def pretrain(cfg=None):
    mode = cfg.run_mode
    model = cfg.model.type
    bs = cfg.batch_size
    
    model = ConditionBERT if model == 'bert' else ConditionTrans
    # net = model(embed_dim=cfg.embed_dim, num_heads=cfg.num_heads, mlp_ratio=cfg.mlp_ratio, glob_feat_dim=cfg.glob_feat_dim, depth={'encoder':cfg.depth, 'decoder':cfg.depth}, knn_layer=cfg.knn_layer_num, fps=True)
    net = model(**cfg.model.kwargs)
    if mode == 'train':
        # TODO: dataset传入cfg参数
        trainset = get_dataset(cfg, mode='train')
        valset = get_dataset(cfg, mode='val')
         
        trainloader = data.DataLoader(trainset, batch_size=bs, shuffle=True)
        valloader = data.DataLoader(valset, batch_size=bs, shuffle=False)
        
        optimizer, scheduler = build_optim_sche( net, cfg=cfg)
        chloss = PointCloudCompletionLoss()
        
        net = net.to('cuda')
        chloss = chloss.to('cuda')
        
        trainepoch = PretrainEpoch(chloss, optimizer, scheduler, output_dir=cfg.output_dir, cfg=cfg)
        valepoch = PretrainEpoch(chloss, optimizer, scheduler, output_dir=cfg.output_dir, mode='val', cfg=cfg)
        # import pdb; pdb.set_trace()
        
        for epoch in range(cfg.num_epoch):
            net, _ = trainepoch(trainloader, epoch, net)
            if epoch % cfg.check_interval:
                _, stop_flag = valepoch(valloader, epoch, net)
            if stop_flag:
                print("Early stopping occur!")
                break
    

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
    from easydict import EasyDict
    import utils.cfgs as cfgsu
    
    # import pdb; pdb.set_trace()
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--cuda_id', type=str, default="0")
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--grasp', action='store_true')
    parser.add_argument('--machine', type=str, default='41')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    
    parser.add_argument('--start_epoch', type=int, default=None)
    parser.add_argument('--eval_ds', type=str, default=None)
    parser.add_argument('--run_mode', type=str, default='train')
    parser.add_argument('--pt_model', type=str, default='trans')

    args = parser.parse_args()

    set_random_seed(1024)
    
    # cfg = MyOptions()
    # DONE: 读取配置文件并转化成字典，同时加入args的配置
    conf = cfgsu.get_config(args, 'pretrain')
    conf.update(cfgsu.config_exp_name(args))
    conf.update(cfgsu.config_paths(args.machine, conf['exp_name']))
    conf.update(args.__dict__) # args的配置也记录下来
    
    cfg = EasyDict()
    # DONE: transform the dict config to easydict
    cfg = cfgsu.merge_new_config(cfg, conf)
    # conf = OmegaConf.structured(cfg)
    # import pdb; pdb.set_trace()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.cuda_id
    
    

    if cfg.wandb:
        wandb.login()
        wandb.init(project=cfg.project_name,
                name=cfg.exp_name,
                config=conf,
                dir=os.path.join(cfg.output_root, 'wandb')) # omegaconf: resolve=True即可填写自动变量
                # dir: set the absolute path for storing the metadata of each runs
                
    print(f"================ {cfg.run_type} experiment running! ================") # NOTE: Checkpoint! 提醒一下当前实验的属性
    if cfg.run_type == 'train':
        # TODO: train--传入cfg参数
        
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
        # TODO: eval_val--传入cfg参数
        assert args.model_exp_name is not None, "Requires trained models to evaluate validation set!"
        evaluation_val(args)
        
    elif cfg.run_type =='pretrain':
        # TODO: pretrain--传入cfg参数
        pretrain(cfg=cfg)
        
    elif cfg.run_type == 'cgrasp':
        cgrasp(cfg=cfg)
    else:
        # TODO: test--传入cfg参数
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

    


