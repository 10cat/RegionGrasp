# torch related
import sys


sys.path.append('.')
sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

from models.pointnet_encoder import ObjRegionConditionEncoder, PointNetEncoder
from models.CVAE import VAE
from utils.utils import CRot2rotmat, region_masked_pointwise, rotmat2aa

# from option import MyOptions as cfg


class cGraspvae(nn.Module):
    def __init__(self, ConditionNet, in_channel_obj=3, in_channel_hand=3, encoder_sizes=[1024, 512, 256],
                latent_size=64, decoder_sizes=[1024, 256, [16*6, 3]], condition_size=1024, cfg=None):
        super(cGraspvae, self).__init__()

        self.in_channel_obj = in_channel_obj
        self.in_channel_hand = in_channel_hand
        self.encoder_sizes = encoder_sizes
        self.latent_size = latent_size
        self.decoder_sizes = decoder_sizes
        self.condition_size = condition_size

        # self.obj_encoder = PointNetEncoder(global_feat=False, feature_transform=False, channel=self.in_channel_obj)
        self.hand_encoder = PointNetEncoder(global_feat=True, feature_transform=False, channel=self.in_channel_hand)
        # if cfg.fit_Condition is not True:
        #     self.obj_rc_encoder = ObjRegionConditionEncoder()
        self.cnet = ConditionNet
        # import pdb; pdb.set_trace()

        self.cvae = VAE(encoder_layer_sizes=self.encoder_sizes,
                        latent_size=self.latent_size,
                        decoder_layer_sizes=self.decoder_sizes,
                        condition_size=self.condition_size, cfg=cfg)
        self.cfg = cfg

    def forward(self, obj_pc, hand_xyz, mask_center=None, region_mask=None, condition_vec=None):
        """
        :param obj_pc:[B, 3+n, N1]
        :param hand_xyz: [B, 3, 778]
        :return: reconstructed hand vertex
        """

        B = obj_pc.size(0)
        # obj_pc_masked = region_masked_pointwise(obj_pc, region_mask)
        # obj_glb_feature, _, _ = self.obj_encoder(obj_pc) # [B, 1024]
        hand_glb_feature, _, _ = self.hand_encoder(hand_xyz) # [B, 1024]
        # if condition_vec is None:
        #     assert region_mask is not None
        #     obj_rc_glb_feature, _, _ = self.obj_rc_encoder(obj_pc, region_mask)
        #     condition_vec = obj_rc_glb_feature
        mask = None
        if self.cfg.model.cnet_type == 'mae':
            assert mask_center is not None, "Requires the center point of the masked region"
            condition_vec, mask = self.cnet(obj_pc, mask_center=mask_center)
        elif self.cfg.model.cnet_type == 'obj_comp':
            condition_vec = self.cnet(obj_pc, feat_only=True)
        else:
            raise NotImplementedError
        recon, means, log_var, z = self.cvae(x=hand_glb_feature, c=condition_vec)
        # import pdb; pdb.set_trace()
        pose, trans = recon
        recon = hand_params_decode(pose, trans)
        # recon = recon.contiguous().view(B, 61)
        return recon, [means, log_var, z], mask

    def inference(self, obj_pc, mask_center=None, region_mask=None, condition_vec=None):
        B = obj_pc.size(0)
        # obj_pc_masked = region_masked_pointwise(obj_pc, region_mask)
        # if condition_vec is None:
        #     assert region_mask is not None
        #     obj_rc_glb_feature, _, _ = self.obj_rc_encoder(obj_pc, region_mask)
        #     condition_vec = obj_rc_glb_feature
        mask = None
        if self.cfg.model.cnet_type == 'mae':
            assert mask_center is not None, "Requires the center point of the masked region"
            condition_vec, mask = self.cnet(obj_pc, mask_center=mask_center)
        elif self.cfg.model.cnet_type == 'obj_comp':
            condition_vec = self.cnet(obj_pc, feat_only=True)
        else:
            raise NotImplementedError
        
        recon, z = self.cvae.inference(n=B, c=condition_vec)
        # recon = recon.contiguous().view(B, 61)
        pose, trans = recon
        recon = hand_params_decode(pose, trans)
        return recon, mask

def hand_params_decode(pose, trans):
    batch_size = trans.shape[0]

    pose_full = CRot2rotmat(pose) # [bs*16, 3, 3]
    pose = pose_full.view([batch_size, 1, -1, 9]) # [bs, 1, 16, 9]
    pose = rotmat2aa(pose).view(batch_size, -1) # [bs, 48]

    global_orient = pose[:, :3]
    hand_pose = pose[:, 3:]
    pose_full = pose_full.view([batch_size, -1, 3, 3])

    hand_params = {'global_orient':global_orient, 'hand_pose':hand_pose, 'transl':trans, 'fullpose':pose_full}

    return hand_params



if __name__ == "__main__":
    B = 32
    obj_verts = torch.randn(B, 3, 3000)
    region_mask = torch.randn(B, 1, 3000)
    hand_verts = torch.randn(B, 3, 778)

    model = cGraspvae(in_channel_obj=3)

    recon, stats = model(obj_verts, hand_verts, region_mask)
    means, log_var, z = stats
    import pdb; pdb.set_trace()


    recon_inf = model.inference(obj_verts)
    
    