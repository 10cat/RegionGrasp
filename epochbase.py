from copy import deepcopy
import os
import sys
sys.path.append('.')
sys.path.append('..')
from option import MyOptions as cfg
os.environ['CUDA_VISIBLE_DEVICES'] = cfg.visible_device
import config
import numpy as np
import torch
from torch import Tensor, optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import mano
import trimesh

from utils.utils import makepath, retrieve_name, edges_for
from utils.meters import AverageMeter, AverageMeters
from utils.logger import Monitor
from models.ConditionNet import ConditionNet
from models.cGrasp_vae import cGraspvae
from traineval_utils.loss import ConditionNetLoss, cGraspvaeLoss
from traineval_utils.metrics import ConditionNetMetrics, cGraspvaeMetrics
from utils.utils import func_timer
from utils.visualization import visual_hand, visual_obj
# from option import MyOptions as cfg
import wandb

to_dev = lambda tensor, device: tensor.to(device)
# to_gpu = lambda tensor: tensor.to('cuda')
to_cpu = lambda tensor: tensor.detach().cpu()

class Epoch(nn.Module):
    def __init__(self, dataloader, dataset, mode='train', use_cuda=True, cuda_id=0, save_visual=True):
        super(Epoch, self).__init__()
        self.dataloader = dataloader
        # self.opt = cfg
        self.mode = mode
        self.use_cuda = use_cuda
        self.device = torch.device("cuda:%d" % cuda_id if self.use_cuda and torch.cuda.is_available() else 'cpu')
        self.save_visual = save_visual
        self.output_dir = cfg.output_dir
        self.model_root = cfg.model_root
        self.dataset = dataset
        self.log_monitor = Monitor(log_folder_path=cfg.output_dir)
        self.models = self.init_models()
        self.init_mano()
        self.init_optimizers()
        self.init_losses() # losses initialization requires the repeated rhand faces
        # import pdb; pdb.set_trace()
        
    
    def init_models(self, checkpoints=None):
        # self.model = config.Model(self.opt)
        self.ConditionNet = ConditionNet(input_channel_obj=3, input_channel_hand=3)
        # import pdb; pdb.set_trace()
        self.cGraspVAE = cGraspvae(in_channel_obj=3, in_channel_hand=3, encoder_sizes=cfg.VAE_encoder_sizes)
        # import pdb; pdb.set_trace()
        
        return [self.ConditionNet, self.cGraspVAE]

    def init_optimizers(self):
        # self.optimizer_cond = config.optimizer_cond(self.opt)
        self.optimizer_cond = optim.Adam(self.ConditionNet.parameters(), lr=cfg.learning_rate)
        self.optimizer_cgrasp = optim.Adam(self.cGraspVAE.parameters(), lr=cfg.learning_rate)

    def init_losses(self):
        self.ConditionNetLoss = ConditionNetLoss().to(self.device)
        self.cGraspVAELoss = cGraspvaeLoss(self.rh_f, self.rh_model, self.device).to(self.device)
        self.ConditionNetMetrics = ConditionNetMetrics() # metrics always on cpu
        self.cGraspvaeMetrics = cGraspvaeMetrics(self.rh_model, self.rh_f, self.device) # metrics always on cpu

    def init_mano(self):
        with torch.no_grad():
            rh_model = mano.load(model_path=cfg.mano_rh_path,
                                      model_type='mano',
                                      num_pca_comps=45,
                                      batch_size=cfg.batch_size,
                                      flat_hand_mean=True)
            self.rh_model = rh_model.to(self.device)
            self.rh_f_single = torch.from_numpy(self.rh_model.faces.astype(np.int32)).view(1, 3, -1)
            self.rh_f = self.rh_f_single.repeat(cfg.batch_size, 1, 1).to(self.device).to(torch.long)
        

    def load_checkpoints(self, checkpoints=None):
        if checkpoints is not None:
            if cfg.mode == 'train':
                self.ConditionNet.load_state_dict(checkpoints[0])
                if len(checkpoints) > 1:
                    # import pdb; pdb.set_trace()
                    self.cGraspVAE.load_state_dict(checkpoints[1])
            else:
                self.cGraspVAE.load_state_dict(checkpoints['cGraspVAE_state_dict'])

    def model_mode_setting(self):
        for model in self.models:
            model.train() if self.mode == 'train' else model.eval()
            model.to(self.device) if self.use_cuda else model.to('cpu')
            

    def to_device(self, tensor):
        if self.use_cuda:
            return to_dev(tensor, self.device)
        else:
            # not to_cpu because here we still need the gradient while to_cpu has detach()
            return tensor


    def read_data(self, sample):

        region = self.to_device(sample['region_mask'].transpose(2,1)) # B,C,N
        region_centers = self.to_device(sample['region_centers'])
        obj_vs = self.to_device(sample['verts_obj'].transpose(2,1))
        obj_sdfs = self.to_device(sample['obj_sdf'].transpose(2,1)) if cfg.use_gtsdm else None
        rhand_vs = self.to_device(sample['verts_rhand'].transpose(2,1))

        # import pdb; pdb.set_trace()
        # #region = sample['region_mask'].transpose(2,1).to('cuda')

        return [region, region_centers, obj_vs, obj_sdfs, rhand_vs]

    # @func_timer
    def model_forward(self, data):
        region, region_centers, obj_vs, obj_sdfs, rhand_vs = data
        # model forward: 
        # 1) get condtion vector, along with train assisting cSDF_maps
        # 2) forward the cGraspvae model to obtain hand parameters as the key prediction
        condition_vec, SD_maps, hand_params, sample_stats, feats = None, None, None, None, None
        # if self.use_cuda: self.ConditionNet = self.ConditionNet.to('cuda')
        # import pdb; pdb.set_trace()
        if cfg.forward_Condition:
            feats, SD_maps = self.ConditionNet(obj_vs, rhand_vs, region) # cSDF_maps = [pred_map_obj, pred_map_obj_masked]
            
        if cfg.forward_cGrasp:
            if cfg.forward_Condition is not True:
                hand_params, sample_stats = self.cGraspVAE(obj_vs, rhand_vs, region)
            # if self.use_cuda: self.cGraspVAE = self.cGraspVAE.to('cuda')
            else:
                condition_vec = feats[0]
                hand_params, sample_stats = self.cGraspVAE(obj_vs, rhand_vs, condition_vec)
        
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
    def loss_compute(self, outputs, data, sample_ids=None):
        
        region, region_centers, obj_vs, obj_sdfs, rhand_vs = data

        dict_losses = {}
        dict_loss_cond = {}
        dict_loss_cgrasp = {}
        signed_dists = None

        if cfg.fit_Condition:
            assert outputs['feats'] is not None and outputs['SD_maps'] is not None
            # SDmap_hand, SDmap_obj_masked = outputs['SD_maps']
            # if self.use_cuda: self.ConditionNetLoss.to('cuda')
            loss_cond, dict_loss_cond = self.ConditionNetLoss(outputs['feats'], outputs['SD_maps'], obj_sdfs)
            # dict_loss_cond stores the loss value for each sub loss terms in ConditionNetLoss
            dict_losses['total_loss_cond'] = cfg.lambda_cond * loss_cond

        if cfg.fit_cGrasp:
            assert outputs['hand_params'] is not None
            # if self.use_cuda: self.cGraspVAELoss.to('cuda')
            if cfg.use_h2osigned:
                assert sample_ids is not None
                # import pdb; pdb.set_trace()
                sample_ids = sample_ids.reshape(-1).tolist()
                obj_names = self.dataset.frame_objs[sample_ids]
                # ObjMeshes = self.dataset.object_meshes[obj_names]
                obj_mesh_faces = [torch.Tensor(self.dataset.object_meshes[name].faces).to(self.device) for name in obj_names]
            else:
                obj_mesh_faces = None
            
            
            loss_cgrasp, dict_loss_cgrasp, signed_dists = self.cGraspVAELoss(outputs['hand_params'], outputs['sample_stats'], obj_vs, rhand_vs, region, obj_mesh_faces=obj_mesh_faces)
            # dict_loss_cgrasp stores the loss value for each sub loss terms in cGraspVAELoss
            dict_losses['total_loss_cgrasp'] = loss_cgrasp
        
        # import pdb; pdb.set_trace()
        dict_losses['total_loss'] = sum(dict_losses.values())
        dict_losses.update(dict_loss_cond)
        dict_losses.update(dict_loss_cgrasp)

        return dict_losses, signed_dists

    def metrics_compute(self, outputs, data, signed_dists=None):
        region, region_centers, obj_vs, obj_sdfs, rhand_vs = data

        dict_metrics = {}
        if cfg.forward_Condition:
            SDmap_om = outputs['SD_maps'][0]
            metric_cond = self.ConditionNetMetrics(to_cpu(SDmap_om), to_cpu(obj_sdfs))
            dict_metrics['metric_cond'] = metric_cond
         
        if cfg.forward_cGrasp:
            assert signed_dists is not None
            hand_params = outputs['hand_params']
            dict_metrics_cgrasp = self.cGraspvaeMetrics(signed_dists)

        dict_metrics.update(dict_metrics_cgrasp)

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
    
    def visual(self, rhand_vs_pred, data, sample_ids, batch_id):
        region, region_centers, obj_vs, obj_sdfs, rhand_vs = data

        batch_size = obj_vs.size(0)
        # import pdb; pdb.set_trace()

        output_mesh_root = os.path.join(self.output_dir, self.mode + '_meshes')
        makepath(output_mesh_root)
        output_mesh_folder = os.path.join(output_mesh_root, 'batch_'+str(batch_id), f'epoch_{self.epoch}')
        makepath(output_mesh_folder)

        rhand_vs_pred = to_cpu(rhand_vs_pred) # pred is output from mano forward, no need to transpose
        rhand_vs = to_cpu(rhand_vs).transpose(2, 1)
        obj_vs = to_cpu(obj_vs).transpose(2, 1)
        # rh_faces = to_cpu(self.rh_f).transpose(2, 1)[0]
        rh_faces = self.rh_model.faces
        for idx in range(0, batch_size, cfg.visual_sample_interval):
            sample_idx = sample_ids[idx]
            # import pdb; pdb.set_trace()
            rhand_mesh_pred = trimesh.base.Trimesh(vertices=rhand_vs_pred[idx], faces=rh_faces)
            rhand_mesh = trimesh.base.Trimesh(vertices=rhand_vs[idx], faces=rh_faces)
            # obj_mesh = trimesh.base.Trimesh(vertices=obj_vs[idx], faces=obj_faces[idx])
            obj_name = self.dataset.frame_objs[sample_idx]
            ObjMesh = self.dataset.object_meshes[obj_name]
            obj_mesh = trimesh.base.Trimesh(vertices=obj_vs[idx], faces=ObjMesh.faces)

            region_faces_ids = self.dataset.region_face_ids[str(int(sample_idx))]
            # #region_centers = region_centers.tolist()

            visual_obj(obj_mesh, region_faces_ids=region_faces_ids)
            visual_hand(rhand_mesh)
            visual_hand(rhand_mesh_pred)

            output_name = f"{str(idx)}_{obj_name}_"
            rhand_mesh_pred.export(os.path.join(output_mesh_folder, output_name+'rh_pred.ply'))
            rhand_mesh.export(os.path.join(output_mesh_folder, output_name+'rh_gt.ply'))
            obj_mesh.export(os.path.join(output_mesh_folder, output_name+'obj.ply'))
        
        return

    def decode_batch_hand_params(self, outputs, batch_size):
        hand_params = outputs['hand_params']
        B = batch_size
        if B == cfg.batch_size:
            rhand_pred = self.rh_model(**hand_params)
        else:
            import mano
            rh_model = mano.load(model_path=cfg.mano_rh_path,
                                    model_type='mano',
                                      num_pca_comps=45,
                                      batch_size=B,
                                      flat_hand_mean=True)
            rh_model = rh_model.to(self.device)
            rhand_pred = rh_model(**hand_params)
        rhand_vs_pred = rhand_pred.vertices
        return rhand_vs_pred


    # @func_timer
    def one_batch(self, sample, idx):
        # read data
        data_elements = self.read_data(sample)
        # self.rh_f repeat according to the true batchsize, or may lead to bug on the last batch
        B = data_elements[0].shape[0]
        # import pdb; pdb.set_trace()

        if B != cfg.batch_size:
            self.rh_f = self.rh_f_single.repeat(B, 1, 1).to(self.device).to(torch.long)
            self.init_losses() # losses initialization requires the repeated rhand faces

        outputs = self.model_forward(data_elements)
        dict_losses, signed_dists = self.loss_compute(outputs, data_elements, sample_ids=sample['sample_idx'])
        if self.mode == 'train':
            self.model_update(dict_losses)

        rhand_vs_pred = self.decode_batch_hand_params(outputs, B)
        dict_metrics = self.metrics_compute(outputs, data_elements, signed_dists)
        self.update_meters(dict_losses, dict_metrics)
        if self.mode != 'train':
            if idx % cfg.visual_interval_val == 0: self.visual(rhand_vs_pred, data_elements, sample_ids=sample['sample_idx'], batch_id=idx)
        else:
            if idx == 0: self.visual(rhand_vs_pred, data_elements, sample_ids=sample['sample_idx'], batch_id=idx)

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
                        'best_val': best_val,
                        'ConditonNet_state_dict': self.ConditionNet.state_dict(),
                        'cGraspVAE_state_dict': self.cGraspVAE.state_dict(),},
                        checkpoint_path)
            # checkpoint = torch.load(checkpoint_path)
            # self.ConditionNet.load_state_dict(checkpoint['ConditonNet_state_dict'])
        return best_val

    
    def one_epoch(self, epoch, best_val=None, checkpoints=None):
        if checkpoints is not None:
            self.load_checkpoints(checkpoints)
        self.epoch = epoch
        self.model_mode_setting()
        # import pdb; pdb.set_trace()
        self.Losses, self.Metrics = AverageMeters(), AverageMeters()
        for idx, sample in enumerate(tqdm(self.dataloader, desc=f'{self.mode} epoch:{epoch}')):
            # if idx > 10:
            #     break
            # if idx < len(self.dataloader) - 1:
            #     continue
            if self.mode != 'train':
                with torch.no_grad():
                    self.one_batch(sample, idx)

            else:  
                self.one_batch(sample, idx)
        
        best_val = self.save_checkpoints(epoch, best_val)
        self.metrics_log(epoch)

        if cfg.fit_cGrasp:
            return [self.ConditionNet.state_dict(), self.cGraspVAE.state_dict()], best_val
        else:
            return [self.ConditionNet.state_dict()], best_val
        


class TrainEpoch(Epoch):
    def __init__(self, dataloader, dataset, mode='train', use_cuda=True, cuda_id=0, save_visual=True):
        super().__init__(dataloader, dataset, mode, use_cuda, cuda_id, save_visual)


class ValEpoch(Epoch):
    def __init__(self, dataloader, dataset, mode='train', use_cuda=True, cuda_id=0, save_visual=True):
        super().__init__(dataloader, dataset, mode, use_cuda, cuda_id, save_visual)
        self.v_weights = self.cGraspVAELoss.v_weights
        self.v_weights2 = self.cGraspVAELoss.v_weights2
        self.vpe = self.cGraspVAELoss.vpe
        self.l1loss = self.cGraspVAELoss.LossL1

    def save_checkpoints(self, epoch, best_val):
        if cfg.fit_cGrasp:
            metric_val = self.Metrics.average_meters['max_depth_ratio'].avg # TODO use max_depth_ratio as the main metrics
            if best_val is not None and metric_val > 1 and metric_val < best_val:
                checkpoint_path = os.path.join(self.model_root, f'bestmodel.pth')
                torch.save({'epoch':epoch,
                        'best_val': best_val,
                        'ConditonNet_state_dict': self.ConditionNet.state_dict(),
                        'cGraspVAE_state_dict':self.cGraspVAE.state_dict()},
                        checkpoint_path)
                best_val = metric_val
                print("Saved the latest best model!")
            elif best_val is None:
                best_val = metric_val

        return best_val

    def model_forward(self, data):
        region, region_centers, obj_vs, obj_sdfs, rhand_vs = data
        # model forward: 
        # 1) get condtion vector, along with train assisting cSDF_maps
        # 2) forward the cGraspvae model to obtain hand parameters as the key prediction
        condition_vec, SD_maps, hand_params, sample_stats, feats = None, None, None, None, None
        # if self.use_cuda: self.ConditionNet.to('cuda')
        if cfg.fit_cGrasp:
            # TODO cGraspVAE inference
            # if self.use_cuda: self.cGraspVAE.to('cuda')
            hand_params = self.cGraspVAE.inference(obj_vs,region_mask=region)
        
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


    def select_best_grasp(self, vs_pred_iters, vs_gt, outputs_iters_list):
        # import pdb; pdb.set_trace()
        vs_rec_errs = torch.mean(torch.einsum('bijk,j->bijk', torch.abs((vs_gt - vs_pred_iters)), self.v_weights2), dim=[2,3]) # dim should be (num_iters, B)
        indices = torch.argmin(vs_rec_errs, dim=0) # dim (B,)
        B = indices.shape[0]
        indices = indices.tolist()
        outputs = {}
        outputs_sample = outputs_iters_list[0]
        keys = list(outputs_sample.keys())
        for key in keys:
            if outputs_sample[key] is None:
                outputs[key] = None
            elif key == 'hand_params':
                outputs[key] = outputs_sample[key]
                for batch_idx in range(B):
                    iter_idx = indices[batch_idx]
                    outputs[key]['global_orient'][batch_idx] = outputs_iters_list[iter_idx][key]['global_orient'][batch_idx] 
                    outputs[key]['hand_pose'][batch_idx] = outputs_iters_list[iter_idx][key]['hand_pose'][batch_idx]
                    outputs[key]['transl'][batch_idx] = outputs_iters_list[iter_idx][key]['transl'][batch_idx]
                    outputs[key]['fullpose'][batch_idx] = outputs_iters_list[iter_idx][key]['fullpose'][batch_idx]

            else:
                outputs[key] = torch.zeros_like(outputs_sample[key])
                for batch_idx in range(B):
                    iter_idx = indices[batch_idx]
                    outputs[key][batch_idx] = outputs_iters_list[iter_idx][key][batch_idx]
             
                    
        return outputs

    def test_metrics(self):
        return

    # @func_timer
    def one_batch(self, sample, idx, num_iters=10):
        
        # read data
        data_elements = self.read_data(sample)
        # self.rh_f repeat according to the true batchsize, or may lead to bug on the last batch
        B = data_elements[0].shape[0]
        # import pdb; pdb.set_trace()
        self.rh_f = self.rh_f_single.repeat(B, 1, 1).to(self.device).to(torch.long)
        self.init_losses() # losses initialization requires the repeated rhand faces
        
        # ----- Select the best grasp w.r.t. ground truth ----- #
        vs_pred_iters_list = []
        outputs_iters_list = []
        # import pdb; pdb.set_trace()
        for i in range(num_iters):
            with torch.no_grad(): outputs_iter = self.model_forward(data_elements)
            rhand_vs_pred = self.decode_batch_hand_params(outputs_iter, B)
            # rhand_vs_pred = rhand_vs_pred.transpose(2,1)
            vs_pred_iters_list.append(rhand_vs_pred.unsqueeze(dim=0))
            outputs_iters_list.append(outputs_iter)
            torch.cuda.empty_cache() # 因为需要在一个batch里面iterate很多次，所以

        vs_pred_iters = torch.cat(vs_pred_iters_list) # (num_iters, B, x, y, z)
        vs_gt = data_elements[4] # gt verts, dim (B, x, y, z)
        vs_gt = vs_gt.transpose(2, 1)

        outputs = self.select_best_grasp(vs_pred_iters, vs_gt, outputs_iters_list)

        #--- LOSS compute ----#
        if self.mode != 'test': # validation阶段计算loss是有必要的 -> 监视过拟合情况；test阶段就没有必要 
            dict_losses, signed_dists = self.loss_compute(outputs, data_elements, sample_ids=sample['sample_idx'])
            if self.mode == 'train':
                self.model_update(dict_losses)

        # hand_params = outputs['hand_params']
        rhand_vs_pred = self.decode_batch_hand_params(outputs, B)
        # rhand_vs_pred = rhand_pred.vertices

        #--- Metrics compute ----#
        # 目前可以进行batch计算的metrics: 
        #  -- interpenetration depth => 几何结构合理性指标
        #  -- interpenetration volume （？）=> 几何结构合理性指标
        # 目前不可以进行batch计算的metrics：
        #  -- simulation displacement => 物理合理性指标
        ## 
        dict_metrics = self.metrics_compute(outputs, data_elements, signed_dists)
        self.update_meters(dict_losses, dict_metrics)
        ## TODO 只有test阶段计算没有batch operating的指标
        if self.mode == 'test':
            self.update_meters()

        # --- Visualization --- #
        if self.mode != 'train':
            if idx % cfg.visual_interval_val == 0: self.visual(rhand_vs_pred, data_elements, sample_ids=sample['sample_idx'], batch_id=idx)
        else:
            if idx == 0: self.visual(rhand_vs_pred, data_elements, sample_ids=sample['sample_idx'], batch_id=idx)

        # self.handparam_post()
        # self.visual_post()
        # self.lognotes()

    



        


