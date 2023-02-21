import os
from socket import CAN_ISOTP
import sys

from tqdm import tqdm
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
from torch.utils import data
import torch.optim as optim
import mano
import trimesh

import argparse
import config
# from option import MyOptions
from dataset.Dataset import GrabNetDataset, ObManDataset_test

from epochbase import TrainEpoch, ValEpoch
import random
from utils.utils import set_random_seed, makepath

from traineval_utils import contact, interpenetraion, simulation
from utils.meters import AverageMeter, AverageMeters

from dataset.obman_preprocess import ObManObj
from models.ConditionNet import ConditionMAE, ConditionTrans, ConditionBERT
from models.cGrasp_vae import cGraspvae
from traineval_utils.loss import ChamferDistanceL2Loss, PointCloudCompletionLoss, cGraspvaeLoss
from utils.optim import *
from utils.datasets import get_dataset
from utils.epoch_utils import EpochVAE_comp, ValEpochVAE_comp, EpochVAE_mae, ValEpochVAE_mae,  MetersMonitor, model_update, PretrainEpoch
    
def testmetrics(cfg):
    mode = cfg.run_mode
    ds_root = cfg.obman_root
    shapenet_root = cfg.shapenet_root
    configs = cfg.dataset[mode]._base_.kwargs
    dataset = ObManDataset_test(ds_root = ds_root,
                                shapenet_root = shapenet_root,
                                mano_root = cfg.mano_root,
                                split = mode,
                                cfg = cfg,
                                **configs)
    with torch.no_grad():
        rh_model = mano.load(model_path=cfg.mano_rh_path,
                            model_type='mano',
                            use_pca=cfg.use_mano,
                            num_pca_comps=45,
                            batch_size=cfg.eval_iter,
                            flat_hand_mean=True)
    
    rh_faces = rh_model.faces.astype(np.int32)
    
    Metrics = MetersMonitor()
    pbar = tqdm(range(dataset.__len__()), desc='Testing metrics')
    for idx in pbar:
        sample = dataset.__getitem__(idx)
        index = int(sample['sample_id'].numpy()[0])
        obj_trans = sample['obj_trans'].numpy()
        
        hand_params_pred_t = sample['hand_params_pred']
        hand_params_pred_iters = {'global_orient':hand_params_pred_t[:, :3], 'hand_pose':hand_params_pred_t[:, 3:48], 'transl':hand_params_pred_t[:, 48:]}
        hand_verts_pred_iters = rh_model(**hand_params_pred_iters)
        hand_verts_pred_iters = hand_verts_pred_iters.vertices.detach().numpy()
        
        obj_mesh = dataset.get_sample_obj_mesh(index)
        obj_verts, _ = dataset.get_obj_verts_faces(index)
        obj_verts -= obj_trans
        
        obj_faces = obj_mesh['faces']
        
        hand_verts = sample['hand_verts'].numpy()
        sample_info_gt = {'hand_verts': hand_verts,
                           'hand_faces': rh_faces,
                           'obj_verts': obj_verts,
                           'obj_faces': obj_faces,
                           'index': index}
        gt_CA, gt_contact_rh_faces, gt_contact_rh_verts = contact.get_contact_area(sample_info_gt)
        gt_IV, gt_ID = interpenetraion.main(sample_info_gt, cfg)
        
        
        Metrics_iters = AverageMeters()
        for i in range(cfg.eval_iter):
            hand_verts_pred = hand_verts_pred_iters[i]
            sample_info = {'hand_verts': hand_verts_pred,
                           'hand_faces': rh_faces,
                           'obj_verts': obj_verts,
                           'obj_faces': obj_faces,
                           'index': index}
            dict_metrics_iters = {}
            if cfg.CA:
                CA, pred_contact_rh_faces, pred_contact_rh_verts = contact.get_contact_area(sample_info)
                shared_contact_faces = list(set(gt_contact_rh_faces) & set(pred_contact_rh_faces))
                dict_metrics_iters['CA'] = CA
                dict_metrics_iters['CA_gt'] = gt_CA
                dict_metrics_iters['CA_ratio'] = CA / gt_CA
                dict_metrics_iters['CA_hand_verts_ratio_pred'] = len(pred_contact_rh_verts) / 778
                dict_metrics_iters['CA_hand_verts_ratio_gt'] = len(gt_contact_rh_verts) / 778
                
                
            if cfg.IV:
                IV, ID = interpenetraion.main(sample_info, cfg)
                dict_metrics_iters['IV_pred'] = IV
                dict_metrics_iters['ID_pred'] = ID
                dict_metrics_iters['IV_gt'] = gt_IV
                dict_metrics_iters['ID_gt'] = gt_ID
                
                
            if cfg.CA and cfg.IV and IV > 0 and gt_IV > 0:
                CA_IV_ratio = CA / IV
                gt_CA_IV_ratio = gt_CA / gt_IV
                dict_metrics_iters['CA_IV_ratio_pred'] = CA_IV_ratio
                dict_metrics_iters['CA_IV_ratio_gt'] = gt_CA_IV_ratio
                dict_metrics_iters['CA_IV_ratio_pred/gt'] = CA_IV_ratio / gt_CA_IV_ratio
            
            if cfg.cond:    
            # TODO: calculate the condition hit metrics
                condition_hit_rate = condition.main()
            # TODO: compute conditioned variety metrics
            
                
            if cfg.sim:
                save_gif_folder = os.path.join(cfg.output_dir, 'gt_sim', 'gif')
                makepath(save_gif_folder)
                save_obj_folder = os.path.join(cfg.output_dir, 'gt_sim', 'obj')
                makepath(save_obj_folder)
                
                sim_dist = simulation.main(sample_idx=sample_info['index'], 
                                        sample=sample_info, 
                                        save_gif_folder=save_gif_folder, 
                                        save_obj_folder=save_obj_folder)
                
                dict_metrics_iters['sim_dist'] = sim_dist
                
            for key, val in dict_metrics_iters.items():
                Metrics_iters.add_value(key, val)
       
        dict_metrics = Metrics_iters.avg(mode=None) # 取5个iter的平均loss
        
        msg_loss, metrics = Metrics.report(dict_metrics, dtype='numpy')
        msg = msg_loss
        pbar.set_postfix_str(msg)
        
        Allmeters = Metrics.get_avg()
        if cfg.wandb: wandb.log(Allmeters)
        
            
    return
        
    
    


if __name__ == "__main__":
    import wandb
    import argparse
    from omegaconf import OmegaConf
    from easydict import EasyDict
    import utils.cfgs as cfgsu
    
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
    
    parser.add_argument('--CA', action='store_false')
    parser.add_argument('--IV', action='store_false')
    parser.add_argument('--sim', action='store_false')

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
        wandb.init(project='ConditionHOI_metrics',
                name=cfg.exp_name,
                config=conf,
                dir=os.path.join(cfg.output_root, 'wandb')) # omegaconf: resolve=True即可填写自动变量
                # dir: set the absolute path for storing the metadata of each runs
                
    print(f"================ {cfg.run_type} experiment running! ================") # NOTE: Checkpoint! 提醒一下当前实验的属性
        
    testmetrics(cfg)

    


