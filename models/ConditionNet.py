import sys
sys.path.append('.')
sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.functional as F
from option import MyOptions as cfg
from models.pointnet_encoder import PointNetEncoder

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
        self.convfuse_m1 = nn.Conv1d(6000, 4096, 1)
        self.convfuse_m2 = nn.Conv1d(4096, 3000, 1)
        self.bnfuse = nn.BatchNorm1d(3000)
        self.bnfuse_m1 = nn.BatchNorm1d(4096)
        self.bnfuse_m2 = nn.BatchNorm1d(3000)

    def mask_obj_pts(self, x, mask):
        x = x * (1 - mask) # multiplied by negate mask
        return x

    def feat_hand_fusion(self, x, hand):
        x = torch.cat((x, hand), dim=2).permute(0, 2, 1).contiguous()
        x = F.relu(self.bnfuse(self.convfuse(x)))
        x = x.permute(0, 2, 1).contiguous()
        return x

    def feat_om_fusion(self, x, feat_om):
        x = torch.cat((x, feat_om), dim=2).permute(0, 2, 1).contiguous()
        x = F.relu(self.bnfuse_m1(self.convfuse_m1(x)))
        x = F.relu(self.bnfuse_m2(self.convfuse_m2(x)))
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

        # mask the obj pointcloud
        obj_masked_pc = self.mask_obj_pts(obj_pc, region_mask)
        # embed features
        feat_o, trans, trans_feat = self.obj_encoder(obj_pc)
        feat_h, trans2, trans_feat2 = self.hand_encoder(hand_xyz)
        feat_om, trans3, trans_feat3 = self.obj_masked_encoder(obj_masked_pc)

        feat_oh = self.feat_hand_fusion(feat_o, feat_h)
        feat_oom = self.feat_om_fusion(feat_o, feat_om)

        map_hand = self.mapnet(feat_oh, N) # w/ hand
        map_om = self.mapnet(feat_oom, N) # wo hand; w/ om

        return [feat_oom, feat_oh], [map_om, map_hand]

    def inference(self, obj_pc, region_mask):
        B = obj_pc.size(0)
        N = obj_pc.size(2)

        obj_masked_pc = self.mask_obj_pts(obj_pc, region_mask)
        # embed features
        feat_o, trans, trans_feat = self.obj_encoder(obj_pc)
        # feat_h, trans2, trans_feat2 = self.hand_encoder(hand_xyz)
        feat_om, trans3, trans_feat3 = self.obj_masked_encoder(obj_masked_pc)

        # feat_oh = self.feat_hand_fusion(feat_o, feat_h)
        feat_oom = self.feat_om_fusion(feat_o, feat_om)

        map_om = self.mapnet(feat_oom, N) # wo hand; w/ om


        return [feat_oom], [map_om]


        
class SDmapNet(nn.Module):
    def __init__(self, input_dim, layer_dims, output_dim):
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
        self.activate = nn.LeakyReLU(negative_slope=cfg.SDmapNet.leaky_slope)

    def forward(self, x, N):
        """
        :param x: fused feature vector
        :param N: corresponds to the dim of obj verts

        :return 
        """
        batch_size = cfg.batch_size
        x = self.activate(self.bn1(self.conv1(x)))
        x = self.activate(self.bn2(self.conv2(x)))
        x = self.activate(self.bn3(self.conv3(x)))
        x = self.conv4(x)

        x = x.transpose(2, 1).contiguous()
        # x = torch.sigmoid(x)  # w/ sigmoid -- sigmoid the annotated target
        x = x.view(batch_size, N) # w/o sigmoid -- target directly use annotated target

        return x
        

if __name__ == "__main__":
    B = cfg.batch_size
    obj_pc = torch.randn(B, 3, 3000)
    region_mask = torch.zeros(B, 1, 3000)
    hand_xyz = torch.randn(B, 3, 778)

    # pointnet = PointNetEncoder(global_feat=False, feature_transform=False, channel=3)
    model = ConditionNet(input_channel_obj=3, input_channel_hand=3)

    feat_om, map1, map2 = model(obj_pc, region_mask, hand_xyz)


    # import pdb; pdb.set_trace()
    # feats, trans, trans_feat = pointnet(obj_pc)
    # import pdb; pdb.set_trace()
    # feats.shape
    
