# from genericpath import exists
import os
import sys

from utils.utils import makepath

sys.path.append('.')
sys.path.append('..')
from easydict import EasyDict
from omegaconf import OmegaConf


def get_config(args, paths, folder=None):
    cfg_path = os.path.join(paths['output_dir'], 'config.yaml')
    if args.resume and os.path.exists(cfg_path):
        print(f'Resume yaml from {cfg_path}')
        
    elif args.cfgs != 'config':
        cfg_root = './cfgs'
        cfg_path = os.path.join(cfg_root, folder, args.cfgs+'.yaml')
    # import pdb; pdb.set_trace()
    config = OmegaConf.load(cfg_path)
    config = OmegaConf.to_container(config, resolve=True)
    config.update({'cfg_path': cfg_path, 'run_type':folder})
    return config

def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == '_base_':
                # import pdb; pdb.set_trace()
                conf = OmegaConf.load(val)
                conf = OmegaConf.to_container(conf, resolve=True)
                config[key] = EasyDict()
                merge_new_config(config[key], conf)
            else:
                config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config

def save_experiment_config(config):
    makepath(config.output_dir)
    config_path = os.path.join(config.output_dir, 'config.yaml')
    os.system('cp %s %s' % (config.cfg_path, config_path))
    print(f'Copying the config file from {config.cfg_path} to config_path')
    

def config_exp_name(args):
    name = {}
    exp_name = args.exp_name
    if exp_name is None:
        exp_name = args.cfgs
    name['exp_name'] = exp_name
    return name

def config_paths(machine, exp_name):
    paths = {}
    if machine == '97':
        grabnet_root = "/home/datassd/yilin/GrabNet"
        obman_root = "/home/datassd/yilin/obman"
        shapenet_root = "/home/datassd/yilin/ShapeNet_obman"
        output_root = "/home/datassd/yilin/Outputs/ConditionHOI/"
        output_dir = "/home/datassd/yilin/Outputs/ConditionHOI/"+exp_name
        mano_root = "/home/datassd/yilin/Codes/_toolbox/mano"
        mano_rh_path = f"/home/datassd/yilin/Codes/_toolbox/mano/models/MANO_RIGHT.pkl"
    
    model_root = os.path.join(output_dir, 'models')
    
    paths['grabnet_root'] = grabnet_root
    paths['obman_root'] = obman_root
    paths['shapenet_root'] = shapenet_root
    paths['output_root'] = output_root
    paths['output_dir'] = output_dir
    paths['mano_root'] = mano_root
    paths['mano_rh_path'] = mano_rh_path
    paths['model_root'] = model_root
    
    return paths


