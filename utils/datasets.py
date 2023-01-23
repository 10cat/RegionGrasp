import os
import sys
sys.path.append('.')
sys.path.append('..')

from dataset.obman_preprocess import ObManObj
from dataset.Dataset import GrabNetDataset, ObManDataset


def get_dataset(cfg, mode='train'):
    ds_name = cfg.dataset.name
    
    if ds_name == 'obman':
        ds_root = cfg.obman_root
        shapenet_root = cfg.shapenet_root
        configs = cfg.dataset[mode]._base_.kwargs
        # import pdb; pdb.set_trace() # dataset配置问题
        if cfg.run_type == 'pretrain':
            dataset = ObManObj(ds_root = ds_root,
                               shapenet_root = shapenet_root,
                               split = mode,
                               **configs)
        else:
            dataset = ObManDataset(ds_root = ds_root,
                                  shapenet_root = shapenet_root,
                                  split = mode,
                                  **configs)
            
    return dataset
        
        
if __name__ == "__main__":
    import config
    from tqdm import tqdm
    