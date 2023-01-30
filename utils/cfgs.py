import os
import sys
sys.path.append('.')
sys.path.append('..')
from easydict import EasyDict
from omegaconf import OmegaConf

def get_config(args):
    if args.resume:
        cfg_path = os.path.join(args.exp_name, 'config.yaml')
        if not os.path.exists(cfg_path):
            print('Failed to resume')
            raise FileNotFoundError()
        print(f'Resume yaml from {cfg_path}')
    else:
        cfg_root = './cfgs'
        if args.pretrain:
            cfg_path = os.path.join(cfg_root, 'pretrain.yaml')
            run_type = 'pretrain'
        elif args.grasp:
            cfg_path = os.path.join(cfg_root, 'grasp.yaml')
            run_type = 'grasp'
        else:
            cfg_path = os.path.join(cfg_root, 'condition_grasp.yaml')
            run_type = 'cgrasp'
    config = OmegaConf.load(cfg_path)
    config = OmegaConf.to_container(config, resolve=True)
    config.update({'cfg_path': cfg_path, 'run_type':run_type})
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
    config_path = os.path.join(config.exp_name, 'config.yaml')
    os.system('cp %s %s' % (config.cfg_path, config_path))
    print(f'Copying the config file from {config.cfg_path} to config_path')
    

def config_exp_name(exp_name):
    name = {}
    name['exp_name'] = exp_name
    return name

def config_paths(machine, exp_name):
    paths = {}
    if machine == '97':
        grabnet_root = "/home/datassd/yilin/GrabNet"
        obman_root = "/home/dataset/yilin/obman"
        # shapenet_root --> 97上没装shapenetcore
        output_root = "/home/datassd/yilin/Outputs/ConditionHOI/"
        output_dir = "/home/datassd/yilin/Outputs/ConditionHOI/"+exp_name
        mano_rh_path = f"/home/datassd/yilin/Codes/_toolbox/mano/models/MANO_RIGHT.pkl"
    if machine == '41':
        grabnet_root = "/ssd_data/yilin/GrabNet"
        obman_root = "/ssd_data/yilin/obman"
        shapenet_root = "/ssd_data/yilin/ShapeNetCore.v2"
        output_root = "/ssd_data/yilin/Outputs/ConditionHOI/"
        output_dir = "/ssd_data/yilin/Outputs/ConditionHOI/"+exp_name
        mano_rh_path = f"/home/yilin/smpl_models/mano/MANO_RIGHT.pkl"
    model_root = os.path.join(output_dir, 'model')
    
    paths['grabnet_root'] = grabnet_root
    paths['obman_root'] = obman_root
    paths['shapenet_root'] = shapenet_root
    paths['output_root'] = output_root
    paths['output_dir'] = output_dir
    paths['mano_rh_path'] = mano_rh_path
    paths['model_root'] = model_root
    # paths['machine'] = machine
    
    return paths


