import os
import sys
sys.path.append('.')
sys.path.append('..')

from dataset.obman_preprocess import ObManObj, ObManObj_MAE
from dataset.Dataset import GrabNetDataset, ObManDataset, ObManDataset_obj_comp, PretrainDataset


def get_dataset(cfg, mode='train'):
    ds_name = cfg.dataset.name
    
    if ds_name == 'obman_pretrain':
        ds_root = cfg.obman_root
        shapenet_root = cfg.shapenet_root
        configs = cfg.dataset[mode]._base_.kwargs
        # import pdb; pdb.set_trace() # dataset配置问题
        if cfg.mae:
            dataset = ObManObj_MAE(ds_root = ds_root,
                                shapenet_root = shapenet_root,
                                mano_root = cfg.mano_root,
                                split = mode,
                                **configs)
        elif cfg.comp:
            dataset = ObManObj(ds_root = ds_root,
                            shapenet_root = shapenet_root,
                            mano_root = cfg.mano_root,
                            split = mode,
                            **configs)
        else:
            raise NotImplementedError
        
    elif ds_name == 'pretrain':
        obman_root = cfg.obman_root
        shapenet_root = cfg.shapenet_root
        grabnet_root = cfg.grabnet_root
        mano_root = cfg.mano_root
        configs = cfg.dataset[mode]._base_.kwargs
        dataset = PretrainDataset(obman_root=obman_root,
                                  shapenet_root=shapenet_root,
                                  mano_root=mano_root, 
                                  grabnet_root=grabnet_root,
                                  split=mode,
                                  **configs
                                  )
        
    elif ds_name == 'obman':
        ds_root = cfg.obman_root
        shapenet_root = cfg.shapenet_root
        configs = cfg.dataset[mode]._base_.kwargs
        if cfg.mae:
            assert cfg.dataset[mode]._base_.type == 'mae'
            dataset = ObManDataset(ds_root = ds_root,
                                    shapenet_root = shapenet_root,
                                    mano_root = cfg.mano_root,
                                    split = mode,
                                    **configs)
        elif cfg.comp:
            assert cfg.dataset[mode]._base_.type == 'comp'
            dataset = ObManDataset_obj_comp(ds_root = ds_root,
                                    shapenet_root = shapenet_root,
                                    mano_root = cfg.mano_root,
                                    split = mode,
                                    **configs)
        else:
            raise NotImplementedError
        
    elif ds_name == 'grabnet':
        ds_root = cfg.grabnet_root
        configs = cfg.dataset[mode]._base_.kwargs
        dataset = GrabNetDataset(dataset_root=ds_root, 
                                 ds_name=mode,
                                 mano_path=cfg.mano_rh_path,
                                 **configs)
        
        
    else:
        raise NotImplementedError
            
    return dataset
        
        
if __name__ == "__main__":
    import config
    from tqdm import tqdm
    