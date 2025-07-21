import os
import sys

from models.PointMAE import Grouper

sys.path.append('.')
sys.path.append('..')

import random

import config
import mano
# from option import MyOptions as cfg
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import wandb
from dataset import data_transforms
from GraspTTA.utils import loss, utils, utils_loss
from torchvision import transforms
from tqdm import tqdm
from utils.meters import AverageMeter, AverageMeters
from utils.utils import (decode_hand_params_batch, func_timer, get_region_mask,
                         makepath, mask_region_patch, region_masked_pointwise)
from utils.visualization import colors_like, visual_hand, visual_obj

train_transforms = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslate()
    ]
)

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def model_update(optimizer, total_loss, scheduler, epoch=None):
    if isinstance(optimizer, list):
        for optim in optimizer:
            optim.zero_grad()
    else:
        optimizer.zero_grad()
        
    total_loss.backward()
    
    if isinstance(optimizer, list):
        for optim in optimizer:
            optim.step()
    else:
        optimizer.step()
        
    if scheduler is not None:
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step()
        else:
            if epoch is not None:
                scheduler.step(epoch)
            else:
                scheduler.step()
                
    return optimizer, scheduler

class MetersMonitor(object):
    def __init__(self):
        self.Meters = AverageMeters()
        return
    
    def update(self, dict, dtype='torch'):
        for key in dict.keys():
            if dtype == 'torch':
                self.Meters.add_value(key, float(dict[key].detach().to('cpu')))
            else:
                self.Meters.add_value(key, float(dict[key]))
            # self.Meters.add_value(key, dict[key].item())
    
    def get_avg(self, mode=None):
        return self.Meters.avg(mode)
    
    def report(self, dict_loss, total_loss=None, mode=None, dtype='torch'):
        self.update(dict_loss, dtype)
        if total_loss is not None:
            self.update({'total_loss':total_loss}, dtype)
        # self.update({'epoch': epoch})
        logdict = self.get_avg(mode)
        msg = ''
        for key, val in logdict.items():
            msg += f'{key}:{val}; '
        return msg, logdict
        
    
class CheckpointsManage(object):
    def __init__(self, root, interval=1, check_interval=None):
        self.root = root
        self.interval = interval
        self.best_metrics = None
        # self.check_interval = check_interval
        self.no_improve = 0
        return
    
    def save_checkpoints(self, epoch, model,  optimizer, scheduler, metric_value):
        if isinstance(optimizer, list):
            optimizer_states = [optim.state_dict() for optim in optimizer]
        else:
            optimizer_states = optimizer.state_dict()
            
        if isinstance(scheduler, list):
            scheduler_states = [sche.state_dict() for sche in scheduler]
        else:
            scheduler_states = scheduler.state_dict()
            
        if self.best_metrics is None:
            self.best_metrics = metric_value
        elif metric_value < self.best_metrics:
            best_model_path = os.path.join(self.root, f'best_model.pth')
            torch.save({'epoch':epoch,
                        'metrics':metric_value,
                        'state_dict':model.state_dict(),
                        'optimizer': optimizer_states,
                        'scheduler': scheduler_states
                        },
                    best_model_path)
            self.best_metrics = metric_value
            print('best_model saved!')
        else:
            self.no_improve += 1
        
        if epoch % self.interval == 0 and epoch != 0:
            checkpoint_path = os.path.join(self.root, f'checkpoint_{epoch}.pth')
            torch.save({'epoch':epoch,
                        'metrics':metric_value,
                        'state_dict':model.state_dict(),
                        'optimizer': optimizer_states,
                        'scheduler': scheduler_states
                        },
                    checkpoint_path)
            print('checkpoint saved!')
        return self.no_improve
    
    def load_checkpoints(self, model, chkpt_epoch, stdict):
        checkpoint_path = os.path.join(self.root, f'checkpoint_{chkpt_epoch}.pth')
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint[stdict])
        return
    
class VisualPC(object):
    def __init__(self, root, cfg, suffix):
        self.root = root
        self.colors = config.colors
        self.cfg = cfg
        self.suffix = suffix # very important for not overwrite previous results
        return
    
    def create_path(self, epoch, sample_id, name):
        """
        create path name for outputs
        
        -
        """
        folder = os.path.join(self.root, f'epoch_{epoch}'+self.suffix)
        makepath(folder)
        path = os.path.join(folder, f'{sample_id}_{name}.ply')
        return path
    
    
    def visual(self, pcs, pc_colors, sample_id, epoch, name):
        if isinstance(pcs, list):
            assert isinstance(pc_colors, list)
            colors = []
            for idx, pc in enumerate(pcs):
                N = pc.shape[0]
                color = np.expand_dims(colors_like(self.colors[pc_colors[idx]]), axis=0)
                colors.append(color.repeat(N, axis=0))
                # import pdb;pdb.set_trace()
            pcs = np.concatenate(pcs, axis=0)
            colors = np.concatenate(colors, axis=0)
        else:
            assert not isinstance(pc_colors, list), "the colors should not be in list"
            colors = colors_like(self.colors[pc_colors])
        # import pdb; pdb.set_trace()
        PC_visual = trimesh.PointCloud(vertices=pcs, colors=colors)
        PC_visual.export(self.create_path(epoch, sample_id, name+'_pc'))
                
        return 
    
class VisualMesh(VisualPC):
    def __init__(self, root, cfg, suffix):
        super().__init__(root, cfg, suffix)
    
    def visual(self, vertices, faces, mesh_color, sample_id, epoch, name):
        Mesh_visual = trimesh.Trimesh(vertices=vertices, faces=faces, face_colors=colors_like(self.colors[mesh_color]))
        Mesh_visual.export(self.create_path(epoch, sample_id, name))
        return
    
    
class PretrainEpoch():
    def __init__(self, loss, output_dir, mode='train', cfg=None):
        self.loss = loss
        self.mode = mode
        self.output_dir = output_dir
        makepath(self.output_dir)
        self.model_dir = os.path.join(self.output_dir, 'models')
        makepath(self.model_dir)
        self.cfg = cfg
        if self.cfg.run_mode == 'train':
            self.visual_dir = os.path.join(self.output_dir, 'visual', mode)
        else:
            self.visual_dir = os.path.join(self.output_dir, 'visual', self.cfg.run_mode)
        makepath(self.visual_dir)
        
        self.Losses = MetersMonitor()
        self.Checkpt = CheckpointsManage(root=self.model_dir, interval=cfg.check_interval)
        self.Visual = VisualPC(root=self.visual_dir, cfg=cfg, suffix='')
        self.batch_interval = cfg.visual_interval[mode].batch
        self.sample_interval = cfg.visual_interval[mode].sample
        
        return
    
    def loss_update(self, dict_loss, coarse_pc, fine_pc,  gt_points):
        dict_loss.update(self.loss.forward(coarse_pc, fine_pc, gt_points, dict_loss))
        return dict_loss
        
    def model_forward(self, model, input, gt_points):
        dict_loss = {}
        vec, coarse_pc, fine_pc = model.forward(input.to('cuda'))
        # import pdb; pdb.set_trace() # DONE: CUBLAS_STATUS error -- 因为num_pred/num_query参数设置的不对
        dict_loss = self.loss_update(dict_loss, coarse_pc, fine_pc, gt_points=gt_points.to('cuda'))
        
        return vec, coarse_pc, fine_pc, dict_loss
    
    def visual(self, batch_idx, coarse_pc, fine_pc, gt_points, sample_ids, epoch, batch_interval=None, sample_interval=None):
        if batch_idx % batch_interval == 0:
            if sample_interval is None:
                i = 0
                pcs = [gt_points[i], fine_pc[i], coarse_pc[i]]
                pcs = [pc.detach().to('cpu').numpy() for pc in pcs]
                gt_colors = 'green'
                pred_colors = ['green', 'yellow']
                
                self.Visual.visual(pcs=pcs[0], pc_colors=gt_colors, sample_id=int(sample_ids[i]), epoch=epoch, name='gt')
                self.Visual.visual(pcs=[pcs[1][:2048], pcs[1][2048:]], pc_colors=pred_colors, sample_id=int(sample_ids[i]), epoch=epoch, name='fine')
                self.Visual.visual(pcs=[pcs[2][:128], pcs[2][128:]], pc_colors=pred_colors, sample_id=int(sample_ids[i]), epoch=epoch, name='coarse')
            else:
                raise NotImplementedError()
    
    def log(self, losses, epoch):
        Allmeters = losses.get_avg(self.mode)
        Allmeters.update({'epoch':epoch})
        if self.cfg.wandb: wandb.log(Allmeters)
        
    def __call__(self, dataloader, epoch, model):
        stop_flag = False
        pbar = tqdm(dataloader, desc=f"{self.mode} epoch {epoch}:")
        for batch_idx, sample in enumerate(pbar):
            input = sample['input_points']
            gt_points = sample['gt_points'] # gt for the masked points
            sample_ids = sample['sample_id']
            
            if self.mode != 'train':
                with torch.no_grad(): vec, coarse_pc, fine_pc, dict_loss = self.model_forward(model, input, gt_points)
            else:
                vec, coarse_pc, fine_pc, dict_loss = self.model_forward(model, input, gt_points)
            
            total_loss = sum(dict_loss.values())
            
            if self.mode == 'train':
                self.optimizer, self.scheduler = model_update(self.optimizer, total_loss, self.scheduler, epoch=epoch)
            
            msg_loss, losses = self.Losses.report(dict_loss, total_loss, mode=self.mode)
            msg = msg_loss
            pbar.set_postfix_str(msg)
            # import pdb;pdb.set_trace()
            if self.mode == 'val':
                self.visual(batch_idx, coarse_pc, fine_pc, gt_points, sample_ids, epoch, 
                            batch_interval=self.batch_interval, 
                            sample_interval=self.sample_interval)
            
            if self.cfg.run_check:
                if batch_idx > 5:
                    break 
            
        if self.mode == 'val':
            no_improve_epochs = self.Checkpt.save_checkpoints(epoch, model, metric_value=losses[f'{self.mode}_total_loss'])
            if no_improve_epochs > self.cfg.early_stopping:
                stop_flag = True
            
        self.log(self.Losses)
        return model, stop_flag
    
    
    
class PretrainMAEEpoch(PretrainEpoch):
    def __init__(self, loss, output_dir, mode='train', cfg=None):
        super().__init__(loss, output_dir, mode, cfg)
        
    def model_forward(self, pointmae, input, vis=False):
        dict_loss = {}
        input = input.to('cuda')
        if vis==False:
            rebuild_points, gt_points = pointmae(input, vis=True)
            full_vis, full_pred, full_center = None, None, None
        else:
            full_vis, full_pred, full_center, rebuild_points, gt_points = pointmae(input, vis=True)
        dict_loss = self.loss.forward(rebuild_points, gt_points, dict_loss)
        return full_vis, full_pred, full_center, rebuild_points, gt_points, dict_loss
    
    def visual(self, batch_idx, full_vis, full_pred, full,
               sample_ids, epoch, batch_interval=None, sample_interval=None):
        if batch_idx % batch_interval == 0:
            if sample_interval is None:
                i = 0
                # pcs = [full[i], full_pred[i], full_vis[i], gt_centers[i], rebuild_centers[i], vis_centers[i]]
                pcs = [full[i], full_pred[i], full_vis[i]]
                pcs = [pc.detach().to('cpu').numpy() for pc in pcs]
                gt_colors = 'green'
                pred_colors = 'yellow'
                
                self.Visual.visual(pcs=pcs[0], pc_colors=gt_colors, sample_id=int(sample_ids[i]), epoch=epoch, name='orig')
                self.Visual.visual(pcs=pcs[1], pc_colors=pred_colors, sample_id=int(sample_ids[i]), epoch=epoch, name='pred')
                self.Visual.visual(pcs=pcs[2], pc_colors=gt_colors, sample_id=int(sample_ids[i]), epoch=epoch, name='vis')
            else:
                raise NotImplementedError()
    
    def __call__(self, dataloader, epoch, model, optimizer, scheduler):
        
        Losses = MetersMonitor()
        stop_flag = False
        pbar = tqdm(dataloader, desc=f"{self.mode} epoch {epoch}:")
        if self.mode == 'train':
            model.train()
            model.apply(fix_bn)
        else:
            model.eval()
        for batch_idx, sample in enumerate(pbar):
            input = sample['input_points']
            sample_ids = sample['ids']
            if self.mode != 'train':
                with torch.no_grad(): full_vis, full_pred, full_center, rebuild_points, gt_points, dict_loss = self.model_forward(model, input, vis=True)
            else:
                input = input.to('cuda')
                input = train_transforms(input)
                full_vis, full_pred, full_center, rebuild_points, gt_points, dict_loss = self.model_forward(model, input, vis=True)

            
            total_loss = 0.
            
            for k, v in dict_loss.items():
                total_loss += self.cfg.coef[k] * v
            
            # total_loss = sum(dict_loss.values())
            
            if self.mode == 'train':
                optimizer, scheduler = model_update(optimizer, total_loss, scheduler, epoch=epoch)
            
            msg_loss, losses = Losses.report(dict_loss, total_loss, mode=self.mode)
            msg = msg_loss
            pbar.set_postfix_str(msg)
            
            # TODO: validation的可视化部分
            if epoch % self.cfg.check_interval == 0:
                self.visual(batch_idx, full_vis, full_pred, input,
                            sample_ids, 
                            epoch, 
                            batch_interval=self.batch_interval, 
                            sample_interval=self.sample_interval)
            if self.cfg.run_check:
                if batch_idx > 5:
                    break 
            
        if self.mode == 'val' and epoch % self.cfg.check_interval == 0 and epoch != 0:
            no_improve_epochs = self.Checkpt.save_checkpoints(epoch, model, metric_value=losses[f'{self.mode}_total_loss'], optimizer=optimizer, scheduler=scheduler)
            if no_improve_epochs > self.cfg.early_stopping:
                stop_flag = True
            
        self.log(Losses, epoch=epoch)
        return model, stop_flag    
        
    
class EpochVAE_mae():
    def __init__(self, loss, dataset, output_dir, mode='train', cfg=None):
        self.cfg = cfg
        self.loss = loss
        self.dataset = dataset
        self.mode = mode
        self.output_dir = output_dir
        makepath(self.output_dir)
        self.model_dir = os.path.join(self.output_dir, 'models')
        makepath(self.model_dir)
        if cfg.run_mode == 'train':
            self.visual_dir = os.path.join(self.output_dir, 'visual', mode)
        else:
            self.visual_dir = os.path.join(self.output_dir, 'visual', cfg.run_mode)
        makepath(self.visual_dir)
        
        suffix = ''
        if self.cfg.grabnet:
            suffix = '_grabnet' + suffix
        if self.cfg.rand_id is not None:
            suffix += f'_rand_{self.cfg.rand_id}'
        self.suffix = suffix
        
        # self.Losses = MetersMonitor()
        self.Checkpt = CheckpointsManage(root=self.model_dir, interval=cfg.check_interval)
        self.VisualPC = VisualPC(root=self.visual_dir, cfg=cfg, suffix=suffix)
        self.VisualMesh = VisualMesh(root=self.visual_dir, cfg=cfg, suffix=suffix)
        self.batch_interval = cfg.batch_intv if cfg.run_mode == 'test' else cfg.visual_interval[mode].batch
        self.sample_interval = cfg.sample_intv if cfg.run_mode == 'test' else cfg.visual_interval[mode].sample
        
    def model_forward(self, model, obj_input, hand_input=None, mask_center=None):
        device = 'cuda' if self.cfg.use_cuda else 'cpu'
        if self.mode == 'train':
            # hand_params, sample_stats, mask = model(obj_input, hand_input, mask_center=mask_center)
            hand_params, sample_stats, mask = model(obj_input.to(device), hand_input.to(device), mask_center=mask_center.to(device))
            # return hand_params, sample_stats, mask
        else:
            with torch.no_grad():
                hand_params, sample_stats, mask = model(obj_input.to(device), hand_input.to(device), mask_center=mask_center.to(device))
        return hand_params, sample_stats, mask
        
    def loss_compute(self, hand_params, sample_stats, obj_points, gt_rhand_vs, region_mask, trans=None, cam_extr=None, gt_hand_params=None, obj_normals=None):
        return self.loss(hand_params, sample_stats, obj_points, gt_rhand_vs, region_mask, trans=trans, cam_extr=cam_extr, gt_hand_params=gt_hand_params, obj_normals=obj_normals, mode=self.mode)
        
    def log(self, losses, epoch):
        Allmeters = losses.get_avg(self.mode)
        Allmeters.update({'epoch':epoch})
        if self.cfg.wandb: wandb.log(Allmeters)
    
    def visual_gt(self, rhand_vs, rhand_faces, obj_pc, region_mask, obj_trans, sample_ids, batch_idx, epoch, batch_interval=None, sample_interval=None):
        # if batch_idx % batch_interval == 0:
        if sample_interval is None:
            i = 0
            sample_id = int(sample_ids[i])
            rhand_vs = rhand_vs[i]
            rhand_faces = rhand_faces[i]
            # import pdb; pdb.set_trace()
            self.VisualMesh.visual(vertices=rhand_vs, faces=rhand_faces, mesh_color='skin', sample_id=sample_id, epoch=epoch, name='gt_hand')
            # if self.cfg.dataset.name == 'obman'
            # obj_mesh = self.dataset.get_sample_obj_mesh(sample_id)
            obj_verts, obj_faces = self.dataset.get_obj_verts_faces(sample_id)
            if self.cfg.dataset.name == 'obman' and not self.cfg.tta and not self.cfg.grabnet:
                obj_verts -= obj_trans[i]
            self.VisualMesh.visual(vertices=obj_verts, faces=obj_faces, mesh_color='grey', sample_id=sample_id, epoch=epoch, name='obj')
            region_mask = region_mask[i]
            obj_pc = obj_pc[i]
            # obj_pc = obj_pc[i] + obj_trans[i]
            mask = region_mask > 0.
            self.VisualPC.visual(pcs=obj_pc[mask], pc_colors='blue', sample_id=sample_id, epoch=epoch, name='obj')
            # self.VisualPC.visual(pcs=[obj_pc[~mask], obj_pc[mask]], pc_colors=['white', 'blue'], sample_id=sample_id, epoch=epoch, name='obj')
        return
        
    def __call__(self, dataloader, epoch, model, optimizer, scheduler):
        Losses = MetersMonitor()
        stop_flag = False
        pbar = tqdm(dataloader, desc=f"{self.mode} epoch {epoch}:")
        if self.mode == 'train':
            model.train()
            model.apply(fix_bn)
        else:
            model.eval()
        for batch_idx, sample in enumerate(pbar):
            # if batch_idx == 0:
                # import pdb; pdb.set_trace()
            obj_input_pc = sample['input_pc']
            gt_rhand_vs = sample['hand_verts'].transpose(2, 1)
            mask_centers = sample['contact_center']
            # import pdb; pdb.set_trace()
            sample_ids = sample['sample_id']
            
            hand_params, sample_stats, region_mask = self.model_forward(model, obj_input_pc, gt_rhand_vs, mask_centers)
            
            obj_points = obj_input_pc
            
            if self.cfg.use_mano:
                _, dict_loss, _, rhand_vs_pred, rhand_faces = self.loss_compute(hand_params, sample_stats, obj_points, gt_rhand_vs, region_mask, trans=sample['obj_trans'], cam_extr=sample['cam_extr'], gt_hand_params=sample['hand_params'], obj_normals=sample['obj_point_normals'])
            else:
                _, dict_loss, _, rhand_vs_pred, rhand_faces = self.loss_compute(hand_params, sample_stats, obj_points, gt_rhand_vs, region_mask, obj_normals=sample['obj_point_normals'])
            
            total_loss = sum(dict_loss.values())
            
            if self.mode == 'train':
                optimizer, scheduler = model_update(optimizer, total_loss, scheduler, epoch=epoch)
            
            # torch.cuda.empty_cache()
            msg_loss, losses = Losses.report(dict_loss, total_loss, mode=self.mode)
            msg = msg_loss
            pbar.set_postfix_str(msg)
            
            if epoch % self.cfg.check_interval == 0 and batch_idx % self.batch_interval == 0:
                if self.cfg.dataset.name == 'obman':
                    obj_trans = sample['obj_trans'].detach().to('cpu').numpy()
                elif self.cfg.dataset.name == 'grabnet':
                    obj_trans = None
                else:
                    raise NotImplementedError
                rhand_vs_pred_0 = rhand_vs_pred[0].detach().to('cpu').numpy()
                rhand_faces_0 = rhand_faces[0].detach().to('cpu').numpy()
                sample_id = int(sample_ids.detach().to('cpu').numpy()[0])
                self.VisualMesh.visual(vertices=rhand_vs_pred_0, faces=rhand_faces_0, mesh_color='skin', sample_id=sample_id, epoch=epoch, name=f'pred_hand')
                self.visual_gt(rhand_vs = gt_rhand_vs.transpose(2, 1).detach().to('cpu').numpy(),
                            rhand_faces = rhand_faces.detach().to('cpu').numpy(),
                            obj_pc=obj_points.detach().to('cpu').numpy(),
                            region_mask=region_mask.detach().to('cpu').numpy(),
                            obj_trans=obj_trans,
                            sample_ids=sample_ids.detach().to('cpu').numpy(),
                            epoch=epoch,
                            batch_idx=batch_idx,
                            batch_interval=self.batch_interval,
                            sample_interval=self.sample_interval)
            if self.cfg.run_check:
                if batch_idx > 5:
                    break
        if self.mode == 'val' and not self.cfg.no_save:
            no_improve_epochs = self.Checkpt.save_checkpoints(epoch, model, metric_value=losses[f'{self.mode}_total_loss'], optimizer=optimizer, scheduler=scheduler)
            if no_improve_epochs > self.cfg.early_stopping:
                stop_flag = True
        
        self.log(Losses, epoch=epoch)
        return model, optimizer, scheduler, stop_flag
        
class EvalEpochVAE_mae(EpochVAE_mae):
    def __init__(self, loss, dataset, output_dir, mode='train', cfg=None):
        super().__init__(loss, dataset, output_dir, mode, cfg)
        cam_extr = np.array([[1., 0., 0., 0.], [0., -1., 0., 0.],
                                  [0., 0., -1., 0.]]).astype(np.float32)
        self.cam_extr = torch.from_numpy(cam_extr).unsqueeze(0)
    def model_forward(self, model, obj_input, hand_input=None, mask_center=None):
        device = 'cuda' if self.cfg.use_cuda else 'cpu'
        with torch.no_grad():
            hand_params_list = []
            # torch.cuda.manual_seed(3407)
            for iter in range(self.cfg.eval_iter):
                B = obj_input.shape[0]
                # hand_params, mask =  model.inference(obj_input.to(device), mask_center=mask_center.to(device))
                hand_params, mask =  model.inference(obj_input.to(device), mask_center=mask_center.to(device))
                hand_params_list.append(hand_params)
                # torch.cuda.empty_cache()
        return hand_params_list, mask
        
    def save_pred_pkl(self, recon_rhand_params):
        recon_rhand_params = torch.cat(recon_rhand_params)
        name = f'{self.cfg.chkpt}_{self.cfg.eval_ds}set' + self.suffix
        
        recon_rhand_params = recon_rhand_params.numpy()
        data = {
            'recon_params': recon_rhand_params
        }
        path = os.path.join(self.output_dir, name + '.pkl')
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"predicted rhand params saved!")
        
    def __call__(self, dataloader, epoch, model, optimizer, scheduler, save_pred=False, cmap_model=False):
        self.Losses = MetersMonitor()
        stop_flag = False
        if save_pred:
            recon_rhand_params = []
        pbar = tqdm(dataloader, desc=f"{self.mode} epoch {epoch}:")
        model.eval()
        for batch_idx, sample in enumerate(pbar):
            # model.apply(fix_bn)
            obj_input_pc = sample['input_pc']
            gt_rhand_vs = sample['hand_verts'].transpose(2, 1) # gt for the masked points
            mask_centers = sample['contact_center']
            # import pdb; pdb.set_trace()
            sample_ids = sample['sample_id']
            
            # import pdb; pdb.set_trace()
            
            hand_params_list, region_mask = self.model_forward(model, obj_input_pc, gt_rhand_vs, mask_centers)
            
            obj_points = obj_input_pc
            
            # NOTE: validation generation in several iters
            Loss_iters = AverageMeters() # loss/metrics计算方式：取5个iter的平均
            rhand_vs_pred_list = []
            for iter in range(self.cfg.eval_iter):
                hand_params = hand_params_list[iter] # 输出每个iter的生成效果fi
                
                # TODO: ========== refine procedure ========#
                if cmap_model:
                    recon_params = torch.cat([hand_params['global_orient'], hand_params['hand_pose'], hand_params['transl']], dim=1)
                    recon_params = torch.autograd.Variable(recon_params, requires_grad=True)
                    optimizer = torch.optim.SGD([recon_params], lr=0.00000625, momentum=0.8)
                    obj_points = obj_points.to('cuda')
                    # for j in tqdm(range(300)):
                    for j in range(30):
                        optimizer.zero_grad()
                        rh_model = mano.load(model_path=self.cfg.mano_rh_path,
                                            model_type='mano',
                                            use_pca=self.cfg.use_mano,
                                            num_pca_comps=45,
                                            batch_size=obj_points.shape[0],
                                            flat_hand_mean=True)
                        rh_model = rh_model.to('cuda')
                        rhand_pred = rh_model(global_orient=recon_params[:, :3], hand_pose=recon_params[:, 3:48], transl=recon_params[:, 48:])
                        rh_faces = torch.from_numpy(rh_model.faces.astype(np.int32)).view(1, -1, 3).to('cuda')
                        rhand_vs_pred = rhand_pred.vertices
                        obj_nn_dist_affordance, _ = utils_loss.get_NN(obj_points[:, :, :3], rhand_vs_pred)
                        cmap_affordance = utils.get_pseudo_cmap(obj_nn_dist_affordance)  # [B,3000]
                        # import pdb; pdb.set_trace()
                        recon_cmap = cmap_model(obj_points.permute(0, 2, 1)[:, :3, :], rhand_vs_pred.permute(0, 2, 1).contiguous())
                        # import pdb; pdb.set_trace()
                        recon_cmap = (recon_cmap / torch.max(recon_cmap, dim=1)[0].reshape(-1, 1)).detach()
                        penetr_loss, consistency_loss, contact_loss = loss.TTT_loss(rhand_vs_pred, rh_faces,
                                                                    obj_points.contiguous(),
                                                                    cmap_affordance, recon_cmap)
                        loss_tta = 1 * contact_loss + 1 * consistency_loss + 7 * penetr_loss
                        loss_tta.backward()
                        optimizer.step()
                    recon_params = recon_params.detach()
                    hand_params = {'global_orient':recon_params[:, :3], 'hand_pose':recon_params[:, 3:48], 'transl': recon_params[:, 48:]}
                    hand_params_list[iter] = hand_params
                    # print(f"batch_idx:{batch_idx}; iter:{iter}/{self.cfg.eval_iter}")
                
                if not cmap_model and self.cfg.tta:
                    recon_params = hand_params
                    # import pdb; pdb.set_trace()
                    hand_params = {'global_orient':recon_params[:, 10:13], 'hand_pose':recon_params[:, 13:58], 'transl': recon_params[:, 58:]}
                    
                if self.cfg.use_mano:
                    _, dict_loss, _, rhand_vs_pred, rhand_faces = self.loss_compute(hand_params, None, obj_points, gt_rhand_vs, region_mask, trans=sample['obj_trans'], cam_extr=sample['cam_extr'], gt_hand_params=sample['hand_params'], obj_normals=sample['obj_point_normals'])
                else:
                    _, dict_loss, _, rhand_vs_pred, rhand_faces = self.loss_compute(hand_params, None, obj_points[:,:,:3], gt_rhand_vs, region_mask, obj_normals=sample['obj_point_normals'])
                
                rhand_vs_pred_list.append(rhand_vs_pred.unsqueeze(0))
                
                
                for key, val in dict_loss.items():
                    Loss_iters.add_value(key, val)
                    # import pdb; pdb.set_trace()
                # if epoch % self.cfg.check_interval == 0 and batch_idx % self.batch_interval == 0:
                rhand_vs_pred_0 = rhand_vs_pred[0].detach().to('cpu').numpy()
                rhand_faces_0 = rhand_faces[0].detach().to('cpu').numpy()
                sample_id = int(sample_ids.detach().to('cpu').numpy()[0])
                self.VisualMesh.visual(vertices=rhand_vs_pred_0, faces=rhand_faces_0, mesh_color='skin', sample_id=sample_id, epoch=epoch, name=f'pred_hand_{iter}')
                
            
            # TODO: diversity
            # import pdb; pdb.set_trace()
            random.shuffle(rhand_vs_pred_list)
            rhand_vs_preds = torch.cat(rhand_vs_pred_list)
            rhand_vs_preds_group1 = rhand_vs_preds[:5]
            rhand_vs_preds_group2 = rhand_vs_preds[5:]
            
            diverse_dists = F.mse_loss(rhand_vs_preds_group1, rhand_vs_preds_group2)
            
            Loss_iters.add_value('diversity', diverse_dists)
        
            
            bug_mode = self.mode # 其实要改掉成None，但只不过和之前的对比实验不好比较了
            dict_loss = Loss_iters.avg(mode=bug_mode) # 取5个iter的平均loss
            
            total_loss = sum(dict_loss.values())
            
            # 还是记录loss_dist_h, loss_dist_o但是不计入total_loss
            msg_loss, losses = self.Losses.report(dict_loss, total_loss, mode=self.mode)
            msg = msg_loss
            pbar.set_postfix_str(msg)
            
            if self.cfg.dataset.name == 'obman' and not self.cfg.grabnet:
                obj_trans = sample['obj_trans'].detach().to('cpu').numpy()
            elif self.cfg.dataset.name == 'grabnet' or self.cfg.grabnet:
                obj_trans = None
            else:
                raise NotImplementedError
            
            # if epoch % self.cfg.check_interval == 0:
            self.visual_gt(rhand_vs = gt_rhand_vs.transpose(2, 1).detach().to('cpu').numpy(),
                        rhand_faces = rhand_faces.detach().to('cpu').numpy(),
                        obj_pc=obj_points[:,:,:3].detach().to('cpu').numpy(),
                        region_mask=region_mask.detach().to('cpu').numpy(),
                        obj_trans=obj_trans,
                        sample_ids=sample_ids.detach().to('cpu').numpy(),
                        epoch=epoch,
                        batch_idx=batch_idx,
                        batch_interval=self.batch_interval,
                        sample_interval=self.sample_interval)
                
            if save_pred:
                recon_params_list = []
                keys = hand_params_list[0].keys()
                for hand_params in hand_params_list:
                    recon_params = torch.cat([hand_params['global_orient'], hand_params['hand_pose'], hand_params['transl']], dim=1)
                    recon_params_list.append(recon_params)
                # import pdb; pdb.set_trace()
                B = recon_params.shape[0]      
                
                hand_params_all = torch.cat(recon_params_list, dim=0) # B*iter_nums, D
                hand_params_all = hand_params_all.reshape(self.cfg.eval_iter, B, -1).transpose(0, 1) # B, iter_nums, D
                hand_params_all = hand_params_all.cpu()
                recon_rhand_params.append(hand_params_all)
            if self.cfg.run_check:
                if batch_idx > 5:
                    break
            
        if self.cfg.run_mode == 'train' and not self.cfg.no_save:
            no_improve_epochs = self.Checkpt.save_checkpoints(epoch, model, optimizer=optimizer, scheduler=scheduler, metric_value=losses[f'{self.mode}_total_loss'])
            if no_improve_epochs > self.cfg.early_stopping:
                stop_flag = True
                
        if save_pred:
            self.save_pred_pkl(recon_rhand_params)
                
        self.log(self.Losses, epoch=epoch)
        return model, stop_flag
    
class TestTTAEpoch(EvalEpochVAE_mae):
    def __init__(self, loss, dataset, output_dir, mode='train', cfg=None):
        super().__init__(loss, dataset, output_dir, mode, cfg)
        self.group_divider = Grouper(128, 32)
        
    def model_forward(self, model, obj_input, hand_input=None, mask_center=None):
        device = 'cuda' if self.cfg.use_cuda else 'cpu'
        with torch.no_grad():
            hand_params_list = []
            # torch.cuda.manual_seed(3407)
            obj_input = obj_input.to(device)
            mask_center = mask_center.to(device)
            neighborhood, center, p_idx = self.group_divider(obj_input[:, :, :3], return_idx=True)
            batch_mask = mask_region_patch(center, mask_center, self.cfg.region_size)
            region_mask = get_region_mask(batch_mask, p_idx, obj_input, self.cfg.region_size)
            for iter in range(self.cfg.eval_iter):
                B = obj_input.shape[0]
                # hand_params, mask =  model.inference(obj_input.to(device), mask_center=mask_center.to(device))
                hand_params =  model.inference(obj_input.transpose(1, 2))
                hand_params_list.append(hand_params)
                # torch.cuda.empty_cache()
        return hand_params_list, region_mask
        
    def __call__(self, dataloader, epoch, model, optimizer, scheduler, save_pred=False, cmap_model=False):
        self.Losses = MetersMonitor()
        stop_flag = False
        if save_pred:
            recon_rhand_params = []
        pbar = tqdm(dataloader, desc=f"{self.mode} epoch {epoch}:")
        model.eval()
        for batch_idx, sample in enumerate(pbar):
            # model.apply(fix_bn)
            obj_input_pc = sample['input_pc']
            gt_rhand_vs = sample['hand_verts'].transpose(2, 1) # gt for the masked points
            mask_centers = sample['contact_center']
            # import pdb; pdb.set_trace()
            sample_ids = sample['sample_id']
            
            # import pdb; pdb.set_trace()
            
            hand_params_list, region_mask = self.model_forward(model, obj_input_pc, gt_rhand_vs, mask_centers)
            
            obj_points = obj_input_pc
            
            # NOTE: validation generation in several iters
            Loss_iters = AverageMeters() # loss/metrics计算方式：取5个iter的平均
            rhand_vs_pred_list = []
            for iter in range(self.cfg.eval_iter):
                hand_params = hand_params_list[iter] # 输出每个iter的生成效果fi
                
                # TODO: ========== refine procedure ========#
                if cmap_model:
                    recon_params = torch.cat([hand_params['global_orient'], hand_params['hand_pose'], hand_params['transl']], dim=1)
                    recon_params = torch.autograd.Variable(recon_params, requires_grad=True)
                    optimizer = torch.optim.SGD([recon_params], lr=0.00000625, momentum=0.8)
                    obj_points = obj_points.to('cuda')
                    for j in range(100):
                        optimizer.zero_grad()
                        rh_model = mano.load(model_path=self.cfg.mano_rh_path,
                                            model_type='mano',
                                            use_pca=self.cfg.use_mano,
                                            num_pca_comps=45,
                                            batch_size=obj_points.shape[0],
                                            flat_hand_mean=True)
                        rh_model = rh_model.to('cuda')
                        params = {'beta': recon_params[:, :10],'global_orient':recon_params[:, 10:13], 'hand_pose':recon_params[:, 13:58], 'transl': recon_params[:, 58:]}
                        rhand_pred = rh_model(**params)
                        rh_faces = torch.from_numpy(rh_model.faces.astype(np.int32)).view(1, -1, 3).to('cuda')
                        rhand_vs_pred = rhand_pred.vertices
                        obj_nn_dist_affordance, _ = utils_loss.get_NN(obj_points[:, :, :3], rhand_vs_pred)
                        cmap_affordance = utils.get_pseudo_cmap(obj_nn_dist_affordance)  # [B,3000]
                        # import pdb; pdb.set_trace()
                        recon_cmap = cmap_model(obj_points.permute(0, 2, 1)[:, :3, :], rhand_vs_pred.permute(0, 2, 1).contiguous())
                        # import pdb; pdb.set_trace()
                        recon_cmap = (recon_cmap / torch.max(recon_cmap, dim=1)[0].reshape(-1, 1)).detach()
                        penetr_loss, consistency_loss, contact_loss = loss.TTT_loss(rhand_vs_pred, rh_faces,
                                                                    obj_points[:, :, :3].contiguous(),
                                                                    cmap_affordance, recon_cmap)
                        loss_tta = 1 * contact_loss + 1 * consistency_loss + 7 * penetr_loss
                        loss_tta.backward()
                        optimizer.step()
                    recon_params = recon_params.detach()
                    hand_params = recon_params
                    hand_params_list[iter] = hand_params
                
                
                # TODO: grasptta requires extra camera fitting coordinate
                cam_extr = self.cam_extr.repeat(obj_points.shape[0], 1, 1).to('cuda')
                # import pdb; pdb.set_trace()
                obj_points = obj_points.to('cuda')
                obj_points = torch.bmm(obj_points, cam_extr.transpose(1, 2))
            
                recon_params = hand_params
                # import pdb; pdb.set_trace()
                hand_params = {'beta': recon_params[:, :10],'global_orient':recon_params[:, 10:13], 'hand_pose':recon_params[:, 13:58], 'transl': recon_params[:, 58:]}
                
                    # import pdb; pdb.set_trace()
                if self.cfg.use_mano:
                    _, dict_loss, _, rhand_vs_pred, rhand_faces = self.loss_compute(hand_params, None, obj_points, gt_rhand_vs, region_mask, trans=sample['obj_trans'], cam_extr=sample['cam_extr'], gt_hand_params=sample['hand_params'], obj_normals=sample['obj_point_normals'])
                else:
                    _, dict_loss, _, rhand_vs_pred, rhand_faces = self.loss_compute(hand_params, None, obj_points[:,:,:3], gt_rhand_vs, region_mask, obj_normals=sample['obj_point_normals'])
                
                for key, val in dict_loss.items():
                    Loss_iters.add_value(key, val)
                    # import pdb; pdb.set_trace()
                rhand_vs_pred_list.append(rhand_vs_pred.unsqueeze(0))
                # if epoch % self.cfg.check_interval == 0 and batch_idx % self.batch_interval == 0:
                rhand_vs_pred_0 = rhand_vs_pred[0].detach().to('cpu').numpy()
                rhand_faces_0 = rhand_faces[0].detach().to('cpu').numpy()
                sample_id = int(sample_ids.detach().to('cpu').numpy()[0])
                self.VisualMesh.visual(vertices=rhand_vs_pred_0, faces=rhand_faces_0, mesh_color='skin', sample_id=sample_id, epoch=epoch, name=f'pred_hand_{iter}')
                
            random.shuffle(rhand_vs_pred_list)
            rhand_vs_preds = torch.cat(rhand_vs_pred_list)
            rhand_vs_preds_group1 = rhand_vs_preds[:5]
            rhand_vs_preds_group2 = rhand_vs_preds[5:]
            
            diverse_dists = F.mse_loss(rhand_vs_preds_group1, rhand_vs_preds_group2)
            
            Loss_iters.add_value('diversity', diverse_dists)
            
            bug_mode = self.mode # 其实要改掉成None，但只不过和之前的对比实验不好比较了
            dict_loss = Loss_iters.avg(mode=bug_mode) # 取5个iter的平均loss
            
            total_loss = sum(dict_loss.values())
            
            msg_loss, losses = self.Losses.report(dict_loss, total_loss, mode=self.mode)
            msg = msg_loss
            pbar.set_postfix_str(msg)
            
            if self.cfg.dataset.name == 'obman' and not self.cfg.grabnet:
                obj_trans = sample['obj_trans'].detach().to('cpu').numpy()
            elif self.cfg.dataset.name == 'grabnet' or self.cfg.grabnet:
                obj_trans = None
            else:
                raise NotImplementedError
            
            # if epoch % self.cfg.check_interval == 0:
            self.visual_gt(rhand_vs = gt_rhand_vs.transpose(2, 1).detach().to('cpu').numpy(),
                        rhand_faces = rhand_faces.detach().to('cpu').numpy(),
                        obj_pc=obj_points[:,:,:3].detach().to('cpu').numpy(),
                        region_mask=region_mask.detach().to('cpu').numpy(),
                        obj_trans=obj_trans,
                        sample_ids=sample_ids.detach().to('cpu').numpy(),
                        epoch=epoch,
                        batch_idx=batch_idx,
                        batch_interval=self.batch_interval,
                        sample_interval=self.sample_interval)
                
            if save_pred:
                recon_params_list = []
                for hand_params in hand_params_list:
                    recon_params = hand_params[:, 10:]
                    recon_params_list.append(recon_params)
                # import pdb; pdb.set_trace()
                B = recon_params.shape[0]      
                
                hand_params_all = torch.cat(recon_params_list, dim=0) # B*iter_nums, D
                hand_params_all = hand_params_all.reshape(self.cfg.eval_iter, B, -1).transpose(0, 1) # B, iter_nums, D
                hand_params_all = hand_params_all.cpu()
                recon_rhand_params.append(hand_params_all)
            if self.cfg.run_check:
                if batch_idx > 5:
                    break
            
        if self.cfg.run_mode == 'train' and not self.cfg.no_save:
            no_improve_epochs = self.Checkpt.save_checkpoints(epoch, model, optimizer=optimizer, scheduler=scheduler, metric_value=losses[f'{self.mode}_total_loss'])
            if no_improve_epochs > self.cfg.early_stopping:
                stop_flag = True
                
        if save_pred:
            self.save_pred_pkl(recon_rhand_params)
                
        self.log(self.Losses, epoch=epoch)
        return model, stop_flag
                    
        
        
    
if __name__ == "__main__":
    visualizer = VisualPC('test_visual', 'train')
    pc1 = np.random.rand(2048, 3)
    pc2 = np.random.rand(32, 3)
    visualizer.visual(pcs=[pc1, pc2], pc_colors=['grey', 'blue'], sample_id=0, epoch=0)
    
