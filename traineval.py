from calendar import c
import os
import sys

from tqdm import tqdm
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

def evaluation(checkpoint):
    testdataset = GrabNetDataset(dataset_root=config.DATASET_ROOT, 
                                 ds_name="test", 
                                 frame_names_file=cfg.frame_names, 
                                 grabnet_thumb=True, 
                                 obj_meshes_folder=cfg.obj_meshes, 
                                 select_ids=True)
    testloader = data.DataLoader(testdataset, batch_size=cfg.batch_size, shuffle=False)
    tester = ValEpoch(testloader, testdataset, mode='test', use_cuda=cfg.use_cuda, cuda_id=cfg.cuda_id)
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
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--model_exp_name', type=str, default=None)
    parser.add_argument('--start_epoch', type=int, default=None)

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
    if cfg.run_type == 'train':
        train_val()
        
    elif cfg.run_type == 'eval_val':
        assert args.model_exp_name is not None, "Requires trained models to evaluate validation set!"
        evaluation_val(args)
    else:
        checkpoint = torch.load(args.checkpoint)
        # import pdb; pdb.set_trace()
        evaluation(checkpoint=checkpoint)

    


