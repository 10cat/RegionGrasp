import os
import sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from models.FoldingNet import Fold
from traineval_utils import pointnet_util
from pytorch3d.ops import sample_farthest_points

import random
from models.PointTr import fps, knn_point, Attention

# --复习用：
# class Attention(nn.Module):
#     def __init__(self, dim, num_heads, qk_scale, qkv_bias, attn_drop, proj_drop):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.qk_scale = qk_scale or head_dim ** -0.5
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
        
#     def forward(self, x):
#         B, N, C = x.shape
        
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]
        
#         attn = (q @ k.transpoes(-1, -2)) * self.qk_scale
#         attn = self.attn_drop(attn)
#         attn = torch.softmax(attn, dim=-1)
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x

class Encoder(nn.Module):   ## Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)
    
    
class Grouper(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        
    def forward(self, xyz):
        B, N, _ = xyz.shape
        center = fps(xyz, self.num_group)
        idx = knn_point(self.group_size, xyz, center)
        
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        
        idx_base = torch.arange(0, B, device=xyz.device).view(-1, 1, 1) * N
        idx = idx_base + idx
        neighborhood = xyz.view(B * N, -1)[idx, :]
        neighborhood = neighborhood.view(B, self.num_group, self.group_size, 3).contiguous()
        
        # normalize based on centers
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    """使用VisionTransformer的Block结构：先norm后multihead + 先norm后MLP

    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        self.mlp = Mlp(dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        self.attn = Attention(dim, num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
        
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
                )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x
    
class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)

        x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel
        return x
    
    
# Pretrain model
class MaskTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.mask_ratio = config.mask_ratio 
        self.trans_dim = config.trans_dim
        self.depth = config.depth 
        self.drop_path_rate = config.drop_path_rate
        self.num_heads = config.num_heads 
        # print_log(f'[args] {config.transformer_config}', logger = 'Transformer')
        # embedding
        self.encoder_dims =  config.encoder_dims
        self.encoder = Encoder(encoder_channel = self.encoder_dims)

        self.mask_type = config.mask_type

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def mask_center_block(self, center, noaug=False):
        B, G, _ = center.shape
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        mask_idx = []
        for points in center:
            points = points.unsqueeze(0) # [1, G, 3]
            # 从所有center中随机选取mask block的中心
            index = random.randint(0, points.size(1) - 1)
            # 计算该中心到其他points的距离
            dists = torch.norm(points[:, index].unsqueeze() - points, p=2, dim=-1) # 1,G,3 - 1,G,3 ->1,G
            
            idx = torch.argsort(dists, dim=-1, descending=False)[0] # G (list)
            num_mask = int(self.mask_ratio * G)
            mask = torch.zeros(len(idx))
            mask[idx[:num_mask]] = 1
            mask_idx.append(mask.bool())
            
        bool_masked_pos = torch.stack(mask_idx).to(center.device) # B, G
        return bool_masked_pos
                
    def mask_center_rand(self, center, noaug=False):
        B, G, _ = center.shape
        # import pdb; pdb.set_trace()
        if noaug or self.mask_ratio == 0:
            # 不进行任何的mask
            return torch.zeros(center.shape[:2]).bool()
        num_mask = int(self.mask_ratio * G)
        batch_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.ones(num_mask),
                np.zeros(G - num_mask)
            ])
            np.random.shuffle(mask)
            batch_mask[i, :] = mask
        batch_mask = torch.from_numpy(batch_mask).to(torch.bool)
        
        return batch_mask.to(center.device)
            
                
    def forward(self, neighborhood, center, noaug = False):
        B, G, _ = center.shape
        if self.mask_type == 'block':
            bool_masked_pos = self.mask_center_block(center, noaug=noaug)
        else:
            bool_masked_pos = self.mask_center_rand(center, noaug=noaug)
        
        group_input_tokens = self.encoder(neighborhood)
        
        B, S, C = group_input_tokens.size()
        assert S == G, "Make sure the second dim of the encoder output equal to the center number"
        
        
        x_vis = group_input_tokens[~bool_masked_pos].reshape(B, -1, C) # B, 52(int 0.3*128), 384
        center_vis = center[~bool_masked_pos].reshape(B, -1, 3)
        
        # pos embed for visible patch only
        pos = self.pos_embed(center_vis)
        
        # transformer
        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)
        
        return x_vis, bool_masked_pos
