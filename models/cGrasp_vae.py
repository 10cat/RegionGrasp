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

from network.pointnet_encoder import PointNetEncoder
from network.CVAE import VAE


class cGraspvae(nn.Module):
    def __init__(self, in_channel_obj=4, in_channel_hand=3, encoder_sizes=[1024, 512, 256], \
                latent_size=64, decoder_sizes=[1024, 256, 61], condition_size=1024):
        super(cGraspvae).__init__()

        self.in_channel_obj = in_channel_obj
        self.in_channel_hand = in_channel_hand
        self.encoder_sizes = encoder_sizes
        self.latent_size = latent_size
        self.decoder_sizes = decoder_sizes
        self.condition_size = condition_size

        self.obj_encoder = PointNetEncoder(global_feat=True, feature_transform=False, channel=self.in_channel_obj)
        self.hand_encoder = PointNetEncoder(global_feat=True, feature_transform=False, channel=self.in_channel_hand)

        self.cvae = VAE(encoder_layer_sizes=self.encoder_size,
                        latent_size=self.latent_size,
                        decoder_layer_size=self.decoder_sizes,
                        condition_size=self.condition_size)

    def forward(self, obj_pc, hand_xyz):
        """
        :param obj_pc:[B, 3+n, N1]
        :param hand_param: [B, 61]
        :return: reconstructed hand vertex
        """

        B = obj_pc.size(0)
        obj_glb_feature, _, _ = self.obj_encoder(obj_pc) # [B, 1024]
        hand_glb_feature, _, _ = self.hand_encoder(hand_xyz) # [B, 1024]
        recon, means, log_var, z = self.cvae(x=hand_glb_feature, c=obj_glb_feature)
        recon = recon.contiguous().view(B, 61)
        return recon, means, log_var, z

    def inference(self, obj_pc):
        B = obj_pc.size(0)
        obj_glb_feature, _, _ = self.obj_encoder
        recon, z = self.cvae.inference(n=B, c=obj_glb_feature)
        recon = recon.contiguous().view(B, 61)
        return recon