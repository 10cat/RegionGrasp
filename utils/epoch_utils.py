import os
import sys
sys.path.append('.')
sys.path.append('..')

# from option import MyOptions as cfg
import numpy as np
import torch

from tqdm import tqdm
import wandb

import mano
import trimesh

import config

from utils.utils import decode_hand_params_batch, func_timer, makepath
from utils.meters import AverageMeter, AverageMeters
from utils.visualization import colors_like, visual_hand, visual_obj


def model_update(optimizer, total_loss, scheduler):
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    if scheduler is not None:
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step()
        else:
            scheduler.step()

class MetersMonitor(object):
    def __init__(self):
        self.Meters = AverageMeters()
        return
    
    def update(self, dict):
        for key in dict.keys():
            self.Meters.add_value(key, float(dict[key].detach().to('cpu')))
    
    def get_avg(self, mode):
        return self.Meters.avg(mode)
    
    def report(self, dict_loss, total_loss, mode):
        self.update(dict_loss)
        self.update({'total_loss':total_loss})
        logdict = self.get_avg(mode)
        msg = ''
        for item in logdict.items():
            msg += f'{item[0]}:{item[1]}; '
        return msg, logdict
        
    
class CheckpointsManage(object):
    def __init__(self, root, interval=1, check_interval=None):
        self.root = root
        self.interval = interval
        self.best_metrics = None
        self.no_improve = 0
        return
    
    def save_checkpoints(self, epoch, model, metric_value):
        if self.best_metrics is None:
            self.best_metrics = metric_value
        
        elif metric_value < self.best_metrics:
            best_model_path = os.path.join(self.root, f'best_model.pth')
            torch.save({'epoch':epoch,
                        'metrics':metric_value,
                        'state_dict':model.state_dict()},
                    best_model_path)
            self.best_metrics = metric_value
            
        else:
            self.no_improve += 1
        
        checkpoint_path = os.path.join(self.root, f'checkpoint_{epoch}.pth')
        torch.save({'epoch':epoch,
                    'metrics':metric_value,
                    'state_dict':model.state_dict()},
                checkpoint_path)
        
        return self.no_improve
    
    def load_checkpoints(self, model, chkpt_epoch, stdict):
        checkpoint_path = os.path.join(self.root, f'checkpoint_{chkpt_epoch}.pth')
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint[stdict])
        return
    
class VisualPC(object):
    def __init__(self, root):
        self.root = root
        self.colors = config.colors
        return
    
    def create_path(self, epoch, sample_id, name):
        folder = os.path.join(self.root, f'epoch_{epoch}')
        makepath(folder)
        path = os.path.join(folder, f'{sample_id}_{name}_pc.ply')
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
        PC_visual.export(self.create_path(epoch, sample_id, name))
                
        return 
    
class VisualMesh(object):
    def __init__(self, root):
        self.root = root
        self.colors = config.colors
        return
    def create_path(self, epoch, sample_id, name):
        folder = os.path.join(self.root, f'epoch_{epoch}')
        makepath(folder)
        path = os.path.join(folder, f'{sample_id}_{name}.ply')
        return path
    
    def visual(self, vertices, faces, mesh_color, sample_id, epoch, name):
        Mesh_visual = trimesh.Trimesh(vertices=vertices, faces=faces, face_colors=colors_like(self.colors[mesh_color]))
        Mesh_visual.export(self.create_path(epoch, sample_id, name))
        return
    
    
class PretrainEpoch():
    def __init__(self, loss, optimizer, scheduler, output_dir, mode='train', cfg=None):
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.mode = mode
        self.output_dir = output_dir
        makepath(self.output_dir)
        self.model_dir = os.path.join(self.output_dir, 'models')
        makepath(self.model_dir)
        self.visual_dir = os.path.join(self.output_dir, 'visual', mode)
        makepath(self.visual_dir)
        
        self.Losses = MetersMonitor()
        self.Checkpt = CheckpointsManage(root=self.model_dir, interval=cfg.check_interval)
        self.Visual = VisualPC(root=self.visual_dir)
        self.batch_interval = cfg.visual_interval[mode].batch
        self.sample_interval = cfg.visual_interval[mode].sample
        self.cfg = cfg
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
    
    def log(self, losses):
        Allmeters = losses.get_avg(self.mode)
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
                model_update(self.optimizer, total_loss, self.scheduler)
            
            msg_loss, losses = self.Losses.report(dict_loss, total_loss, mode=self.mode)
            msg = msg_loss
            pbar.set_postfix_str(msg)
            # import pdb;pdb.set_trace()
            if self.mode == 'val':
                self.visual(batch_idx, coarse_pc, fine_pc, gt_points, sample_ids, epoch, 
                            batch_interval=self.batch_interval, 
                            sample_interval=self.sample_interval)
                
            # break
            
        if self.mode == 'val':
            no_improve_epochs = self.Checkpt.save_checkpoints(epoch, model, metric_value=losses[f'{self.mode}_total_loss'])
            if no_improve_epochs > self.cfg.early_stopping:
                stop_flag = True
            
            
        self.log(self.Losses)
        return model, stop_flag
    
    
class EpochVAE():
    def __init__(self, loss, dataset, optimizer, scheduler, output_dir, mode='train', cfg=None):
        self.loss = loss
        self.dataset = dataset
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.mode = mode
        self.output_dir = output_dir
        makepath(self.output_dir)
        self.model_dir = os.path.join(self.output_dir, 'models')
        makepath(self.model_dir)
        self.visual_dir = os.path.join(self.output_dir, 'visual', mode)
        makepath(self.visual_dir)
        
        self.Losses = MetersMonitor()
        self.Checkpt = CheckpointsManage(root=self.model_dir, interval=cfg.check_interval)
        self.VisualPC = VisualPC(root=self.visual_dir)
        self.VisualMesh = VisualMesh(root=self.visual_dir)
        self.batch_interval = cfg.visual_interval[mode].batch
        self.sample_interval = cfg.visual_interval[mode].sample
        self.cfg = cfg
        
    def model_forward(self, model, obj_input, hand_input=None):
        device = 'cuda' if self.cfg.use_cuda else 'cpu'
        if self.mode == 'train':
            hand_params, sample_stats = model(obj_input.to(device), hand_input.to(device))
            return hand_params, sample_stats
        else:
            with torch.no_grad():
                hand_params_list = []
                for iter in range(self.cfg.eval_iter):
                    B = obj_input.shape[0]
                    hand_params =  model.inference(obj_input.to(device))
                    hand_params_list.append(hand_params)
                    torch.cuda.empty_cache()
            return hand_params_list
        
    
    def log(self, losses):
        Allmeters = losses.get_avg(self.mode)
        if self.cfg.wandb: wandb.log(Allmeters)
    
    def visual_gt(self, rhand_vs, rhand_faces, obj_pc, region_mask, obj_trans, sample_ids, batch_idx, epoch, batch_interval=None, sample_interval=None):
        if batch_idx % batch_interval == 0:
            if sample_interval is None:
                i = 0
                sample_id = int(sample_ids[i])
                rhand_vs = rhand_vs[i]
                rhand_faces = rhand_faces[i]
                # import pdb; pdb.set_trace()
                self.VisualMesh.visual(vertices=rhand_vs, faces=rhand_faces, mesh_color='skin', sample_id=sample_id, epoch=epoch, name='gt_hand')
                obj_mesh = self.dataset.get_sample_obj_mesh(sample_id)
                obj_verts, _ = self.dataset.get_obj_verts_faces(sample_id)
                obj_verts -= obj_trans[i]
                self.VisualMesh.visual(vertices=obj_verts, faces=obj_mesh['faces'], mesh_color='grey', sample_id=sample_id, epoch=epoch, name='obj')
                region_mask = region_mask[i]
                obj_pc = obj_pc[i]
                mask = region_mask > 0.
                self.VisualPC.visual(pcs=[obj_pc[~mask], obj_pc[mask]], pc_colors=['green', 'yellow'], sample_id=sample_id, epoch=epoch, name='obj')
        return
        
    def __call__(self, dataloader, epoch, model):
        stop_flag = False
        pbar = tqdm(dataloader, desc=f"{self.mode} epoch {epoch}:")
        for batch_idx, sample in enumerate(pbar):
            obj_input_pc = sample['input_pc']
            gt_rhand_vs = sample['hand_verts'].transpose(2, 1) # gt for the masked points
            sample_ids = sample['sample_id']
            
            hand_params, sample_stats = self.model_forward(model, obj_input_pc, gt_rhand_vs)
            
            obj_points = sample['obj_points']
            obj_normals = sample['obj_point_normals']
            region_mask = sample['region_mask']
            
            _, dict_loss, signed_dists, rhand_vs_pred, rhand_faces = self.loss(hand_params, sample_stats, obj_points, gt_rhand_vs, region_mask, obj_normals=obj_normals)
            
            total_loss = sum(dict_loss.values())
            
            model_update(self.optimizer, total_loss, self.scheduler)
            
            torch.cuda.empty_cache()
            msg_loss, losses = self.Losses.report(dict_loss, total_loss, mode=self.mode)
            msg = msg_loss
            pbar.set_postfix_str(msg)
                
            # break
            
            
        self.log(self.Losses)
        return model, stop_flag
    
class ValEpochVAE(EpochVAE):
    def __init__(self, loss, dataset, optimizer, scheduler, output_dir, mode='train', cfg=None):
        super().__init__(loss, dataset, optimizer, scheduler, output_dir, mode, cfg)
        
    def __call__(self, dataloader, epoch, model):
        stop_flag = False
        pbar = tqdm(dataloader, desc=f"{self.mode} epoch {epoch}:")
        for batch_idx, sample in enumerate(pbar):
            with torch.no_grad():
                obj_input_pc = sample['input_pc']
                gt_rhand_vs = sample['hand_verts'].transpose(2, 1) # gt for the masked points
                sample_ids = sample['sample_id']
                
                hand_params_list = self.model_forward(model, obj_input_pc, gt_rhand_vs)
                
                obj_points = sample['obj_points']
                obj_normals = sample['obj_point_normals']
                region_mask = sample['region_mask']
                
                # NOTE: validation generation in several iters
                Loss_iters = AverageMeters() # loss/metrics计算方式：取5个iter的平均
                for iter in range(self.cfg.eval_iter):
                    hand_params = hand_params_list[iter] # 输出每个iter的生成效果
                    _, dict_loss, signed_dists, rhand_vs_pred, rhand_faces = self.loss(hand_params, None, obj_points, gt_rhand_vs, region_mask, obj_normals=obj_normals)
                    for key, val in dict_loss.items():
                        Loss_iters.add_value(key, val)
                    if batch_idx % self.batch_interval == 0:
                        rhand_vs_pred_0 = rhand_vs_pred[0].detach().to('cpu').numpy()
                        rhand_faces_0 = rhand_faces[0].detach().to('cpu').numpy()
                        sample_id = int(sample_ids.detach().to('cpu').numpy()[0])
                        self.VisualMesh.visual(vertices=rhand_vs_pred_0, faces=rhand_faces_0, mesh_color='skin', sample_id=sample_id, epoch=epoch, name=f'pred_hand_{iter}')
                
                dict_loss = Loss_iters.avg(mode=self.mode) # 取5个iter的平均loss
                
                total_loss = sum(dict_loss.values())
                
                msg_loss, losses = self.Losses.report(dict_loss, total_loss, mode=self.mode)
                msg = msg_loss
                pbar.set_postfix_str(msg)
                
                if self.mode == 'val':
                    self.visual_gt(rhand_vs = gt_rhand_vs.transpose(2, 1).detach().to('cpu').numpy(),
                                rhand_faces = rhand_faces.detach().to('cpu').numpy(),
                                obj_pc=obj_points.detach().to('cpu').numpy(),
                                region_mask=region_mask.detach().to('cpu').numpy(),
                                obj_trans=sample['obj_trans'].detach().to('cpu').numpy(),
                                sample_ids=sample_ids.detach().to('cpu').numpy(),
                                epoch=epoch,
                                batch_idx=batch_idx,
                                batch_interval=self.batch_interval,
                                sample_interval=self.sample_interval)
                
            # break
            
        if self.mode == 'val':
            no_improve_epochs = self.Checkpt.save_checkpoints(epoch, model, metric_value=losses[f'{self.mode}_total_loss'])
            if no_improve_epochs > self.cfg.early_stopping:
                stop_flag = True
            
        self.log(self.Losses)
        return model, stop_flag
        
    
        
        
    
if __name__ == "__main__":
    visualizer = VisualPC('test_visual', 'train')
    pc1 = np.random.rand(2048, 3)
    pc2 = np.random.rand(32, 3)
    visualizer.visual(pcs=[pc1, pc2], pc_colors=['grey', 'blue'], sample_id=0, epoch=0)
    
    