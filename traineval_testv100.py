import os
import sys

from tqdm import tqdm

from dataloader import Batchsampler_testv100
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
    parser.add_argument('--cfgs', type=str)
    parser.add_argument('--cfgs_fodler', type=str, default='cgrasp')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--cuda_id', type=str, default="0")
    parser.add_argument('--machine', type=str, default='41')
    parser.add_argument('--wandb', action='store_true')
    
    
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--mae', action='store_true')
    parser.add_argument('--comp', action='store_true')
    parser.add_argument('--grasp', action='store_true')

    parser.add_argument('--chkpt', type=str, default=None)
    
    parser.add_argument('--start_epoch', type=int, default=None)
    parser.add_argument('--eval_ds', type=str, default='test')
    parser.add_argument('--batch_intv', type=str, default=1)
    parser.add_argument('--sample_intv', type=str, default=None)
    parser.add_argument('--run_mode', type=str, default='train')
    parser.add_argument('--pt_model', type=str, default='trans')

    args = parser.parse_args()

    set_random_seed(1024)
    
    # cfg = MyOptions()
    # DONE: 读取配置文件并转化成字典，同时加入args的配置
    conf = cfgsu.get_config(args, args.cfgs_fodler)
    conf.update(cfgsu.config_exp_name(args))
    conf.update(cfgsu.config_paths(args.machine, conf['exp_name']))
    conf.update(args.__dict__) # args的配置也记录下来
    
    cfg = EasyDict()
    # DONE: transform the dict config to easydict
    cfg = cfgsu.merge_new_config(cfg, conf)
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
        elif cfg.mae:
            cgrasp_mae(cfg=cfg)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    


