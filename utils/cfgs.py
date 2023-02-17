from genericpath import exists
import os
import sys
from utils.utils import makepath
sys.path.append('.')
sys.path.append('..')
from easydict import EasyDict
from omegaconf import OmegaConf

def get_config(args, folder=None):
    cfg_path = os.path.join(args.exp_name, 'config.yaml')
    if args.resume and os.path.exists(cfg_path):
        print(f'Resume yaml from {cfg_path}')
    else:
        cfg_root = './cfgs'
        cfg_path = os.path.join(cfg_root, folder, args.cfgs+'.yaml')
        
    config = OmegaConf.load(cfg_path)
    config = OmegaConf.to_container(config, resolve=True)
    config.update({'cfg_path': cfg_path, 'run_type':folder})
    return config

# def adjust_config(config, args):
#     config.loss.train.loss_edge = False if args.no_loss_edge else True
#     config.loss.train.loss_mesh_rec = False if args.no_loss_mesh_rec else True
#     config.loss.train.loss_dist_h = False if args.no_loss_dist_h else True
#     config.loss.train.loss_dist_o = False if args.no_loss_dist_o else True
#     config.loss.train.loss_penetr = True if args.loss_penetr else False
#     config.loss.train.loss_mano = True if args.loss_mano else False
    

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
        shapenet_root = "/home/datassd/yilin/ShapeNetCore.v2"
        output_root = "/home/datassd/yilin/Outputs/ConditionHOI/"
        output_dir = "/home/datassd/yilin/Outputs/ConditionHOI/"+exp_name
        mano_root = "/home/datassd/yilin/Codes/_toolbox/mano"
        mano_rh_path = f"/home/datassd/yilin/Codes/_toolbox/mano/models/MANO_RIGHT.pkl"
        
    if machine == '208' or machine == '50':
        grabnet_root = "/home/shihao/yilin/GrabNet"
        obman_root = "/home/shihao/yilin/obman"
        shapenet_root = "/home/shihao/yilin/ShapeNetCore.v2"
        output_root = "/home/shihao/yilin/Outputs/ConditionHOI/"
        output_dir = "/home/shihao/yilin/Outputs/ConditionHOI/"+exp_name
        mano_root = "/home/shihao/yilin/Codes/_toolbox/mano"
        mano_rh_path = f"/home/shihao/yilin/Codes/_toolbox/mano/models/MANO_RIGHT.pkl"
        
    if machine == '195':
        grabnet_root = "/home/jupyter-yiling/GrabNet"
        obman_root = "/home/jupyter-yiling/obman"
        shapenet_root = "/home/jupyter-yiling/ShapeNetCore.v2_obman"
        output_root = "/home/jupyter-yiling/Outputs/ConditionHOI/"
        output_dir = "/home/jupyter-yiling/Outputs/ConditionHOI/"+exp_name
        mano_root = "/home/jupyter-yiling/Codes/_toolbox/mano"
        mano_rh_path = f"/home/jupyter-yiling/Codes/_toolbox/mano/models/MANO_RIGHT.pkl"
    if machine == '41':
        grabnet_root = "/ssd_data/yilin/GrabNet"
        obman_root = "/ssd_data/yilin/obman"
        shapenet_root = "/ssd_data/yilin/ShapeNetCore.v2"
        output_root = "/ssd_data/yilin/Outputs/ConditionHOI/"
        output_dir = "/ssd_data/yilin/Outputs/ConditionHOI/"+exp_name
        mano_root = "/home/yilin/smpl_models/mano"
        mano_rh_path = f"/home/yilin/smpl_models/mano/MANO_RIGHT.pkl"
    
    model_root = os.path.join(output_dir, 'model')
    
    paths['grabnet_root'] = grabnet_root
    paths['obman_root'] = obman_root
    paths['shapenet_root'] = shapenet_root
    paths['output_root'] = output_root
    paths['output_dir'] = output_dir
    paths['mano_root'] = mano_root
    paths['mano_rh_path'] = mano_rh_path
    paths['model_root'] = model_root
    # paths['machine'] = machine
    
    return paths


