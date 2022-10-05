import os
import config
import sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.utils import retrieve_name
from utils.meters import AverageMeter, AverageMeters
from utils.logger import Monitor
import wandb

to_gpu = lambda tensor, device: tensor.to(device)
to_cpu = lambda tensor: tensor.detach().cpu()

class Epoch(nn.Module):
    def __init__(self, dataloader, opt, mode='train', use_cuda=True, device=None, save_visual=True):
        super(Epoch).__init__()
        self.dataloader = dataloader
        self.opt = opt
        self.mode = mode
        self.use_cuda = use_cuda
        self.device = device
        self.save_visual = save_visual
        self.log_monitor = Monitor()
        self.init_models()
        self.init_optimizers()
    
    def init_models(self):
        self.model = config.Model(self.opt)
        return

    def init_optimizers(self):
        self.optimizer_cond = config.optimizer_cond(self.opt)
        self.optimizer_cgrasp = config.optimizer_cgrasp(self.opt)

    def to_device(self, tensor):
        if self.use_cuda:
            assert self.device is not None
            return to_gpu(tensor, self.device)
        else:
            # not to_cpu because here we still need the gradient while to_cpu has detach()
            return tensor


    def read_data(self, sample):
        region = self.to_device(sample['region_mask'])
        region_centers = self.to_device(sample['region_centers'])
        obj_vs = self.to_device(sample['verts_obj'])
        obj_sdfs = self.to_device(sample['obj_sdf'])
        rhand_vs = self.to_device(sample['verts_rhand'])

        return [region, region_centers, obj_vs, obj_sdfs, rhand_vs]

    def model_forward(self, data):
        region, region_centers, obj_vs, obj_sdfs, rhand_vs = data

        # model forward: 
        # 1) get condtion vector, along with train assisting cSDF_maps
        # 2) forward the cGraspvae model to obtain hand parameters as the key prediction
        condition_vec, SD_maps, hand_params, sample_stats = None, None, None, None
        condition_vec, SD_maps = self.model.ConditionNet(obj_vs, rhand_vs, region) # cSDF_maps = [pred_map_obj, pred_map_obj_masked]
        if self.model.cGraspVAE:
            hand_params, sample_stats = self.model.cGraspVAE([obj_vs, rhand_vs], condition_vec)
        
        # outputs as a dict
        outputs = {}
        # for i in [hand_params, sample_stats, condition_vec, SD_maps]:
        #     outputs[retrieve_name(i)[0]] = i   # TODO not gonna work when more than one var_val=None      
        outputs['condition_vec'] = condition_vec
        outputs['SD_maps'] = SD_maps
        outputs['hand_params'] = hand_params
        outputs['sample_stats'] = sample_stats # sample_stats = [p_mean, p_std]
        return outputs

    def loss_compute(self, outputs, data):
        
        region, region_centers, obj_vs, obj_sdfs, rhand_vs = data

        dict_losses = {}

        if self.model.ConditionNet:
            assert outputs['condition_vec'] is not None and outputs['SD_maps'] is not None
            SDmap_obj, SDmap_obj_masked = outputs['SD_maps']
            loss_cond, dict_loss_cond = self.ConditionNetLoss(SDmap_obj, SDmap_obj_masked, obj_sdfs, self.opt)
            # dict_loss_cond stores the loss value for each sub loss terms in ConditionNetLoss
            dict_losses['total_loss_cond'] = self.opt.lambda_cond * loss_cond

        if self.model.cGraspVAE:
            assert outputs['hand_params'] is not None and outputs['sample_stats'] is not None
            p_mean, p_std = outputs['sample_stats']
            loss_cgrasp, dict_loss_cgrasp = self.cGraspVAELoss(outputs['hand_params'], p_mean, p_std, obj_vs, rhand_vs, self.opt)
            # dict_loss_cgrasp stores the loss value for each sub loss terms in cGraspVAELoss
            dict_losses['total_loss_cgrasp'] = loss_cgrasp
            
        
        dict_losses['total_loss'] = sum(dict_losses.values())
        dict_losses.update(dict_loss_cond)
        dict_losses.update(dict_loss_cgrasp)

        return dict_losses

    def metrics_compute(self, outputs, data):
        region, region_centers, obj_vs, obj_sdfs, rhand_vs = data

        dict_metrics = {}
        


        return dict_metrics

    def model_freeze(self, model):

        return


    def model_update(self, dict_losses):
        ################ TODO This part should belongs to model files ##########################
        # if self.ConditionNet and self.opt.freeze_ConditionNet:
        #     self.model_freeze(self.ConditionNet)
        # elif self.cGraspVAE and self.opt.freeze_cGraspVAE:
        #     self.model_freeze(self.freeze_cGraspVAE)
        #######################################################################################
        if self.model.ConditionNet: self.model.ConditionNet.zero_grad() 
        if self.model.cGraspVAE: self.model.cGraspVAE.zero_grad()

        if self.opt.fit_Condition:
            dict_losses['total_loss_cond'].backward()
            self.optimizer_cond.step()
        if self.opt.fit_cGrasp:
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

    def one_batch(self, sample):
        # read data
        data_elements = self.read_data(sample)
        outputs = self.model_forward(data_elements)
        dict_losses = self.loss_compute(outputs, data_elements, dict_losses)
        dict_metrics = self.metrics_compute(outputs, data_elements, dict_metrics)
        if self.mode == 'train':
            self.model_update(dict_losses)
        self.update_meters(dict_losses, dict_metrics)
        # self.handparam_post()
        # self.visual_post()
        # self.lognotes()
        
        return
    
    def epoch(self, epoch):
        self.Losses, self.Metrics = AverageMeters(), AverageMeters()
        for idx, sample in enumerate(tqdm(self.dataloader, desc=f'Epoch:{epoch}')):
            if self.mode != 'train':
                with torch.no_grad():
                    self.one_batch(sample)
            else:
                self.one_batch(sample)

        loss_message = self.log_monitor.loss_message(self.Losses, epoch)
        metric_message = self.log_monitor.metric_mesage(self.Metrics, epoch)
        print(loss_message, metric_message)

        AllMeters = self.Losses.update(self.Metrics)
        wandb.log(AllMeters)



        


