import os
import config
import sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
from torch import Tensor, optim
import torch.nn as nn
from tqdm import tqdm

from utils.utils import makepath, retrieve_name
from utils.meters import AverageMeter, AverageMeters
from utils.logger import Monitor
from models.ConditionNet import ConditionNet
from models.cGrasp_vae import cGraspvae
from train_utils.loss import ConditionNetLoss, cGraspvaeLoss
from train_utils.metrics import ConditionNetMetrics
from utils.utils import func_timer
from option import MyOptions as cfg
import wandb

to_dev = lambda tensor, device: tensor.to(device)
to_gpu = lambda tensor: tensor.to('cuda')
to_cpu = lambda tensor: tensor.detach().cpu()
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

class Epoch(nn.Module):
    def __init__(self, dataloader, mode='train', use_cuda=True, device=None, save_visual=True):
        super(Epoch, self).__init__()
        self.dataloader = dataloader
        # self.opt = cfg
        self.mode = mode
        self.use_cuda = use_cuda
        self.device = device
        self.save_visual = save_visual
        self.output_root = cfg.output_root
        self.model_root = cfg.model_root
        self.log_monitor = Monitor(log_folder_path=cfg.output_root)
        self.models = self.init_models()
        self.init_trainutils()
    
    def init_models(self, checkpoints=None):
        # self.model = config.Model(self.opt)
        self.ConditionNet = ConditionNet(input_channel_obj=3, input_channel_hand=3)
        self.cGraspVAE = cGraspvae()
        return [self.ConditionNet, self.cGraspVAE]

    def init_trainutils(self):
        # self.optimizer_cond = config.optimizer_cond(self.opt)
        self.optimizer_cond = optim.Adam(self.ConditionNet.parameters(), lr=cfg.learning_rate)
        self.optimizer_cgrasp = optim.Adam(self.cGraspVAE.parameters(), lr=cfg.learning_rate)

        self.ConditionNetLoss = ConditionNetLoss().to('cuda') if self.use_cuda else ConditionNetLoss()
        self.cGraspVAELoss = cGraspvaeLoss() if self.use_cuda else cGraspvaeLoss()
        self.ConditionNetMetrics = ConditionNetMetrics() # metrics always on cpu

    def load_checkpoints(self, checkpoints=None):
        if checkpoints is not None:
            self.ConditionNet.load_state_dict(checkpoints[0])
            if len(checkpoints) > 1:
                self.cGraspVAE.load_state_dict(checkpoints[1])

    def model_mode_setting(self):
        for model in self.models:
            model.train() if self.mode == 'train' else model.eval()
            model.to('cuda') if self.use_cuda else model.to('cpu')

    def to_device(self, tensor):
        if self.use_cuda:
            return tensor.to('cuda')
        else:
            # not to_cpu because here we still need the gradient while to_cpu has detach()
            return tensor


    def read_data(self, sample):

        region = self.to_device(sample['region_mask'].transpose(2,1)) # B,C,N
        region_centers = self.to_device(sample['region_centers'])
        obj_vs = self.to_device(sample['verts_obj'].transpose(2,1))
        obj_sdfs = self.to_device(sample['obj_sdf'].transpose(2,1))
        rhand_vs = self.to_device(sample['verts_rhand'].transpose(2,1))

        # import pdb; pdb.set_trace()
        # region = sample['region_mask'].transpose(2,1).to('cuda')

        return [region, region_centers, obj_vs, obj_sdfs, rhand_vs]

    # @func_timer
    def model_forward(self, data):
        region, region_centers, obj_vs, obj_sdfs, rhand_vs = data
        # model forward: 
        # 1) get condtion vector, along with train assisting cSDF_maps
        # 2) forward the cGraspvae model to obtain hand parameters as the key prediction
        condition_vec, SD_maps, hand_params, sample_stats = None, None, None, None
        # if self.use_cuda: self.ConditionNet = self.ConditionNet.to('cuda')
        # import pdb; pdb.set_trace()
        feats, SD_maps = self.ConditionNet(obj_vs, rhand_vs, region) # cSDF_maps = [pred_map_obj, pred_map_obj_masked]
        condition_vec = feats[0]
        if cfg.fit_cGrasp:
            # if self.use_cuda: self.cGraspVAE = self.cGraspVAE.to('cuda')
            hand_params, sample_stats = self.cGraspVAE([obj_vs, rhand_vs], condition_vec)
        
        # outputs as a dict
        outputs = {}
        # for i in [hand_params, sample_stats, condition_vec, SD_maps]:
        #     outputs[retrieve_name(i)[0]] = i   # TODO not gonna work when more than one var_val=None      
        outputs['condition_vec'] = condition_vec
        outputs['feats'] = feats
        outputs['SD_maps'] = SD_maps
        outputs['hand_params'] = hand_params
        outputs['sample_stats'] = sample_stats # sample_stats = [p_mean, p_std]
        return outputs

    # @func_timer
    def loss_compute(self, outputs, data):
        
        region, region_centers, obj_vs, obj_sdfs, rhand_vs = data

        dict_losses = {}
        dict_loss_cond = {}
        dict_loss_cgrasp = {}

        if cfg.fit_Condition:
            assert outputs['feats'] is not None and outputs['SD_maps'] is not None
            # SDmap_hand, SDmap_obj_masked = outputs['SD_maps']
            # if self.use_cuda: self.ConditionNetLoss.to('cuda')
            loss_cond, dict_loss_cond = self.ConditionNetLoss(outputs['feats'], outputs['SD_maps'], obj_sdfs)
            # dict_loss_cond stores the loss value for each sub loss terms in ConditionNetLoss
            dict_losses['total_loss_cond'] = cfg.lambda_cond * loss_cond

        if cfg.fit_cGrasp:
            assert outputs['hand_params'] is not None and outputs['sample_stats'] is not None
            p_mean, p_std = outputs['sample_stats']
            if self.use_cuda: self.cGraspVAELoss.to('cuda')
            loss_cgrasp, dict_loss_cgrasp = self.cGraspVAELoss(outputs['hand_params'], p_mean, p_std, obj_vs, rhand_vs)
            # dict_loss_cgrasp stores the loss value for each sub loss terms in cGraspVAELoss
            dict_losses['total_loss_cgrasp'] = loss_cgrasp
        
        # import pdb; pdb.set_trace()
        dict_losses['total_loss'] = sum(dict_losses.values())
        dict_losses.update(dict_loss_cond)
        dict_losses.update(dict_loss_cgrasp)

        return dict_losses

    def metrics_compute(self, outputs, data):
        region, region_centers, obj_vs, obj_sdfs, rhand_vs = data

        dict_metrics = {}
        SDmap_om = outputs['SD_maps'][0]
        metric_cond = self.ConditionNetMetrics(to_cpu(SDmap_om), to_cpu(obj_sdfs))
        dict_metrics['metric_cond'] = metric_cond

        return dict_metrics

    def model_freeze(self, model):

        return

    # @func_timer
    def model_update(self, dict_losses):
        ################ TODO This part should belongs to model files ##########################
        # if self.ConditionNet and self.opt.freeze_ConditionNet:
        #     self.model_freeze(self.ConditionNet)
        # elif self.cGraspVAE and self.opt.freeze_cGraspVAE:
        #     self.model_freeze(self.freeze_cGraspVAE)
        #######################################################################################
        if self.ConditionNet: self.ConditionNet.zero_grad() 
        if self.cGraspVAE: self.cGraspVAE.zero_grad()

        if cfg.fit_Condition:
            dict_losses['total_loss_cond'].backward()
            self.optimizer_cond.step()
        if cfg.fit_cGrasp:
            dict_losses['total_loss_cgrasp'].backward()
            self.optimizer_cgrasp.step()
            
        return

    def update_meters(self, dict_losses, dict_metrics):
        for name in dict_losses.keys():
            self.Losses.add_value(name, dict_losses[name])
        for name in dict_metrics.keys():
            self.Metrics.add_value(name, dict_metrics[name])
        
    def handparam_post(self, outputs):
        return
    
    def visual_post(self, outputs):
        return

    # @func_timer
    def one_batch(self, sample):
        # read data
        data_elements = self.read_data(sample)
        outputs = self.model_forward(data_elements)
        dict_losses = self.loss_compute(outputs, data_elements)
        if self.mode == 'train':
            self.model_update(dict_losses)
        dict_metrics = self.metrics_compute(outputs, data_elements)
        self.update_meters(dict_losses, dict_metrics)
        # self.handparam_post()
        # self.visual_post()
        # self.lognotes()

    def metrics_log(self, epoch):
        # import pdb; pdb.set_trace()
        losses_avg = self.Losses.avg(mode=self.mode)
        metrics_avg = self.Metrics.avg(mode=self.mode)
        
        loss_message = self.log_monitor.loss_message(losses_avg, epoch)
        metric_message = self.log_monitor.metric_message(metrics_avg, epoch)
        # import pdb; pdb.set_trace()
        print(loss_message, metric_message)
        # import pdb; pdb.set_trace()

        AllMeters_avg = losses_avg
        AllMeters_avg.update(metrics_avg)
        # turn averagemeters to
        if cfg.w_wandb: wandb.log(AllMeters_avg)

    def save_checkpoints(self, epoch, best_val):
        if epoch > 1 and epoch % cfg.check_interval == 0:
            makepath(self.model_root)
            checkpoint_path = os.path.join(self.model_root, f'checkpoint_{epoch}.pth')
            # import pdb; pdb.set_trace()
            torch.save({'epoch':epoch, 
                        'ConditonNet_state_dict': self.ConditionNet.state_dict()},
                        checkpoint_path)
            # checkpoint = torch.load(checkpoint_path)
            # self.ConditionNet.load_state_dict(checkpoint['ConditonNet_state_dict'])

    
    def epoch(self, epoch, best_val=None, checkpoints=None):
        if self.mode != 'train':
            self.load_checkpoints(checkpoints)
        self.model_mode_setting()
        self.Losses, self.Metrics = AverageMeters(), AverageMeters()
        for idx, sample in enumerate(tqdm(self.dataloader, desc=f'{self.mode} epoch:{epoch}')):
            if self.mode != 'train':
                with torch.no_grad():
                    self.one_batch(sample)

            else:
                self.one_batch(sample)
            # if idx > 20:
            #     break
        self.save_checkpoints(epoch, best_val)
        self.metrics_log(epoch)

        if cfg.fit_cGrasp:
            return [self.ConditionNet.state_dict(), self.cGraspVAE.state_dict()]
        else:
            return [self.ConditionNet.state_dict()]
        


class TrainEpoch(Epoch):
    def __init__(self, dataloader, mode='train', use_cuda=True, device=None, save_visual=True):
        super().__init__(dataloader, mode, use_cuda, device, save_visual)


class ValEpoch(Epoch):
    def __init__(self, dataloader, mode='train', use_cuda=True, device=None, save_visual=True):
        super().__init__(dataloader, mode, use_cuda, device, save_visual)

    def save_checkpoints(self, epoch, best_val):
        if cfg.fit_Condition and not cfg.fit_cGrasp:
            metric_val = self.Metrics.average_meters['metric_cond'].avg
            if best_val is not None and metric_val > best_val:
                checkpoint_path = os.path.join(self.model_root, f'bestmodel.pth')
                torch.save({'epoch':epoch, 
                        'ConditonNet_state_dict': self.ConditionNet.state_dict()},
                        checkpoint_path)

    def model_forward(self, data):
        region, region_centers, obj_vs, obj_sdfs, rhand_vs = data
        # model forward: 
        # 1) get condtion vector, along with train assisting cSDF_maps
        # 2) forward the cGraspvae model to obtain hand parameters as the key prediction
        condition_vec, SD_maps, hand_params, sample_stats = None, None, None, None
        # if self.use_cuda: self.ConditionNet.to('cuda')
        feats, SD_maps = self.ConditionNet.inference(obj_vs, region) # cSDF_maps = [pred_map_obj, pred_map_obj_masked]
        condition_vec = feats[0]
        if cfg.fit_cGrasp:
            # TODO cGraspVAE inference
            # if self.use_cuda: self.cGraspVAE.to('cuda')
            hand_params, sample_stats = self.cGraspVAE([obj_vs, rhand_vs], condition_vec)
        
        # outputs as a dict
        outputs = {}
        # for i in [hand_params, sample_stats, condition_vec, SD_maps]:
        #     outputs[retrieve_name(i)[0]] = i   # TODO not gonna work when more than one var_val=None      
        outputs['condition_vec'] = condition_vec
        outputs['feats'] = feats
        outputs['SD_maps'] = SD_maps
        outputs['hand_params'] = hand_params
        outputs['sample_stats'] = sample_stats # sample_stats = [p_mean, p_std]
        return outputs

    



        


