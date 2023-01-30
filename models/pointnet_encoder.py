"""
Developed with reference to https://github.com/hwjiang1510/GraspTTA/blob/master/network/pointnet_encoder.py
"""
"""
[Coding notes]
- torch.bmm: https://pytorch.org/docs/stable/generated/torch.bmm.html
"""


# torch related import
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable

import numpy as np
import torch.nn.functional as F
from utils.utils import region_masked_pointwise, size_splits, func_timer
from option import MyOptions as cfg

class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9) # output_dim = 9
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.shape[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        #TODO why do we need the 'iden' additive component？
        iden = torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)).view(1, 9).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)

        return x

class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)).view(1, self.k * self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()

        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)
    
    def pre_pointfeat_forward(self, x):

        B, D, N = x.size()
        # - input transfrom based on STN3d
        trans = self.stn(x) 
        # - matrix-matrix product
        x = x.transpose(2, 1)
        if D > 3:
            x, feature = size_splits(x, [3, D-3], dim=2) 
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1) 
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None
            
        # NOTE: local feature (B, 64, N), 64-dim for every point

        return x, trans, trans_feat, N

    def pointfeat_forward(self, point_feat):
        x = point_feat
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        return x

    # @func_timer
    def forward(self, x):
        # x = x.transpoes()
        x, trans, trans_feat, N = self.pre_pointfeat_forward(x)

        pointfeat = x # local feature for every given points -- (B, 64, N)
        
        x = self.pointfeat_forward(x)

        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N) # (B, 1024) -> (B, 1024, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat
        

class ObjRegionConditionEncoder(PointNetEncoder):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super().__init__(global_feat, feature_transform, channel)
        output_dim = cfg.VAE_condition_size
        self.conv2_m = nn.Conv1d(64, 128, 1)
        self.conv3_m = nn.Conv1d(128, 1024, 1)
        self.bn2_m = nn.BatchNorm1d(128)
        self.bn3_m = nn.BatchNorm1d(1024)
        self.convfuse = nn.Conv1d(in_channels=2048, out_channels=output_dim, kernel_size=1)
        self.bnfuse = nn.BatchNorm1d(output_dim)

    def pointfeat_masked_forward(self, pointfeat_masked):
        x = pointfeat_masked
        x = F.relu(self.bn2_m(self.conv2_m(x)))
        x = self.bn3_m(self.conv3_m(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        return x


    def forward(self, obj_pc, region_mask):

        B = obj_pc.shape[0]
        x, trans, trans_feat, N = self.pre_pointfeat_forward(obj_pc)

        #  NOTE: obj pointfeat -> local feature: (B, 64, N)
        pointfeat = x
        # import pdb; pdb.set_trace()
        # NOTE: obj masked pointfeat -> mask方式是region_mask(B, 1, N)直接与obj pointfeat点乘(B, 64, N)
        # FIXME: 由于选取出来的点一般相对于N来说较少，二值化的region_mask(B, 1, N)直接与obj pointfeat点乘(B, 64, N)的特征向量(B, 64, N)会特别稀疏
        pointfeat_masked = region_masked_pointwise(pointfeat, region_mask) 
        
        x1 = self.pointfeat_forward(pointfeat)
        x2 = self.pointfeat_masked_forward(pointfeat_masked)

        x = self.bnfuse(self.convfuse(torch.cat((x1, x2), dim=-1).unsqueeze(2))) # use global fuse feature

        x = x.view(B, -1)
        
        return x, trans, trans_feat