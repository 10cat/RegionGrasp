import sys
sys.path.append('.')
sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.functional as F
from option import MyOptions as cfg

from pointnet_encoder import PointNetEncoder


class ConditionNet(nn.Module):
    def __init__(self, input_channel_obj, input_channel_hand):
        super(ConditionNet, self).__init__()
        self.in_channel_obj = input_channel_obj
        self.in_channel_hand = input_channel_hand

        self.obj_encoder = PointNetEncoder(global_feat=False, feature_transform=False, channel=self.in_channel_obj)
        self.obj_masked_encoder = PointNetEncoder(global_feat=False, feature_transform=False, channel=self.in_channel_obj)
        self.hand_encoder = PointNetEncoder(global_feat=False, feature_transform=False, channel=self.in_channel_hand)
        self.mapnet = SDmapNet(input_dim=cfg.SDmapNet.input_dim, layer_dims=cfg.SDmapNet.layer_dims,output_dim=cfg.SDmapNet.output_dim)
        self.convfuse = nn.Conv1d(3778, 3000, 1)
        self.bnfuse = nn.BatchNorm1d(3000)

    def feat_fusion(self, x, hand):
        x = torch.cat((x, hand), dim=2).permute(0, 2, 1).contiguous()
        x = F.relu(self.bnfuse(self.convfuse(x)))
        x = x.permute(0, 2, 1).contiguous()
        return x

    
    def forward(self, obj_pc, region_mask, hand_xyz):
        """
        :param obj_pc: [B, 3, N]
        :param region_mask: [B, 1, N]
        :param hand_xyz: [B, 3, 778]

        :return: predicted 
        """
        
        B = obj_pc.size(0)
        N = obj_pc.size(2)

        # TODO mask the obj pointcloud
        obj_masked_pc = obj_pc * (- region_mask) # !! not complete !not minus, but negate
        # embed features
        feat_o, trans, trans_feat = self.obj_encoder(obj_pc)
        feat_h, trans2, trans_feat2 = self.hand_encoder(hand_xyz)
        feat_om, trans3, trans_feat3 = self.obj_masked_encoder(obj_masked_pc)

        x1 = self.feat_fusion(feat_o, feat_h)
        x2 = self.feat_fusion(feat_om, feat_h)

        Map1 = self.mapnet(x1, N)
        Map2 = self.mapnet(x2, N)

        return feat_om, Map1, Map2

        
class SDmapNet(nn.Module):
    def __init__(self, input_dim, layer_dims,output_dim):
        super(SDmapNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        dim1, dim2, dim3 = layer_dims
        self.conv1 = nn.Conv1d(input_dim, dim1, 1)
        self.conv2 = nn.Conv1d(dim1, dim2, 1)
        self.conv3 = nn.Conv1d(dim2, dim3, 1)
        self.conv4 = nn.Conv1d(dim3, output_dim, 1)
        self.bn1 = nn.BatchNorm1d(dim1)
        self.bn2 = nn.BatchNorm1d(dim2)
        self.bn3 = nn.BatchNorm1d(dim3)

    def forward(self, x, N):
        """
        :param x: fused feature vector
        :param N: corresponds to the dim of obj verts

        :return 
        """
        batch_size = cfg.batch_size
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)

        x = x.transpose(2, 1).contiguous()
        x = torch.sigmoid(x) # TODO also need to sigmoid the target SDmap
        x = x.view(batch_size, N)

        return x
        

if __name__ == "__main__":
    obj_pc = torch.randn(16, 3, 3000)
    pointnet = PointNetEncoder(global_feat=False, feature_transform=False, channel=3)

    import pdb; pdb.set_trace()

    feats, trans, trans_feat = pointnet(obj_pc)
    import pdb; pdb.set_trace()
    feats.shape
    
