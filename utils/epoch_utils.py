import os
import sys
sys.path.append('.')
sys.path.append('..')

from option import MyOptions as cfg
import numpy as np
import torch

from tqdm import tqdm
import wandb

import mano
import trimesh

import config

from utils.utils import func_timer, makepath
from utils.meters import AverageMeter, AverageMeters
from utils.visualization import colors_like, visual_hand, visual_obj


def model_update(optimizer, total_loss, scheduler):
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
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
    def __init__(self, root):
        self.root = root
        self.best_metrics = None
        return
    
    def save_checkpoints(self, epoch, model, metric_value):
        if self.best_metrics is None:
            self.best_metrics = metric_value
        elif metric_value < self.best_metrics:
            checkpoint_path = os.path.join(self.root, f'checkpoint_{epoch}.pth')
            torch.save({'epoch':epoch,
                        'metrics':metric_value,
                        'state_dict':model.state_dict()},
                    checkpoint_path)
            self.best_metrics = metric_value
        return
    
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
    
    def create_path(self, epoch, sample_id):
        folder = os.path.join(self.root, f'epoch_{epoch}')
        makepath(folder)
        path = os.path.join(folder, f'{sample_id}_pc.ply')
        return path
    
    
    def visual(self, pcs, pc_colors, sample_id, epoch):
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
            colors = colors_like(self.colors[pc_colors])
        # import pdb; pdb.set_trace()
        PC_visual = trimesh.PointCloud(vertices=pcs, colors=colors)
        PC_visual.export(self.create_path(epoch, sample_id))
                
        return 
    
    
class PretrainEpoch():
    def __init__(self, loss, optimizer, scheduler, output_dir, mode='train', visual_interval=None):
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
        self.visual_interval = visual_interval
        self.Losses = MetersMonitor()
        self.Checkpt = CheckpointsManage(root=self.model_dir)
        self.Visual = VisualPC(root=self.visual_dir)
        return
    
    def loss_update(self, dict_loss, pred_points, gt_points):
        dict_loss.update(self.loss.forward(pred_points, gt_points, dict_loss))
        return dict_loss
        
    def model_forward(self, model, input, gt_points):
        dict_loss = {}
        vec, pred_points = model.forward(input.to('cuda'))
        dict_loss = self.loss_update(dict_loss, pred_points=pred_points, gt_points=gt_points.to('cuda'))
        
        return vec, pred_points, dict_loss
    
    def visual(self, batch_idx, pred_points, gt_points, sample_ids, epoch, visual_interval=None):
        if batch_idx % visual_interval == 0:
            i = 0
            pcs = [gt_points[i], pred_points[i]]
            pcs = [pc.detach().to('cpu').numpy() for pc in pcs]
            colors = ['orange', 'green']
            self.Visual.visual(pcs=pcs, pc_colors=colors, sample_id=int(sample_ids[i]), epoch=epoch)
    
    def log(self, losses):
        Allmeters = losses.get_avg(self.mode)
        if cfg.w_wandb: wandb.log(Allmeters)
        
    def __call__(self, dataloader, epoch, model):
        pbar = tqdm(dataloader, desc=f"{self.mode} epoch {epoch}:")
        for batch_idx, sample in enumerate(pbar):
            input = sample['input_points']
            gt_points = sample['mask_points'] # gt for the masked points
            sample_ids = sample['sample_id']
            
            if self.mode != 'train':
                with torch.no_grad(): vec, pred_points, dict_loss = self.model_forward(model, input, gt_points)
            else:
                vec, pred_points, dict_loss = self.model_forward(model, input, gt_points)
            
            total_loss = sum(dict_loss.values())
            
            if self.mode == 'train':
                model_update(self.optimizer, total_loss, self.scheduler)
            
            msg_loss, losses = self.Losses.report(dict_loss, total_loss, mode=self.mode)
            msg = msg_loss
            pbar.set_postfix_str(msg)
            # import pdb;pdb.set_trace()
            
            if self.mode == 'val':
                self.Checkpt.save_checkpoints(epoch, model, metric_value=losses[f'{self.mode}_total_loss'])
                self.visual(batch_idx, pred_points, gt_points, sample_ids, epoch, visual_interval=self.visual_interval)
            
        self.log(self.Losses)
        return model
        
    
if __name__ == "__main__":
    visualizer = VisualPC('test_visual', 'train')
    pc1 = np.random.rand(2048, 3)
    pc2 = np.random.rand(32, 3)
    visualizer.visual(pcs=[pc1, pc2], pc_colors=['grey', 'blue'], sample_id=0, epoch=0)
    