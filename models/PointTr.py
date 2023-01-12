import os
import sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_

from option import MyOptions as cfg

def knn_point(nsample, xyz, xyz_q):
    
    sq_dist = square_distance(xyz_q, xyz) # [B, Nq, N]
    _, group_index = torch.topk(sq_dist, nsample, dim=-1, largest=False, sorted=False)
    return group_index # [B, Nq, k]

def square_distance(src, dst):
    
    B, N, _ = src.shape
    _, M, _ = dst.shape
    
    cor = -2 * torch.einsum('bij,bjk->bik', src, dst.transpose(2,1))
    src_sq = torch.sum(src**2, dim=-1).view(B, N, 1)
    dst_sq = torch.sum(dst**2, dim=-1).view(B, 1, M)
    dist = src_sq + cor + dst_sq # [B, N, M]
    
    return dist


def get_knn_index(coor_q, coor_k=None, k=cfg.knn_k):
    coor_k = coor_k if coor_k is not None else coor_q
    
    bs, np_q, _ = coor_q.size()
    _, np_k, _ = coor_k.size()
    
    with torch.no_grad():
        idx = knn_point(k, coor_k, coor_q) # [B, np_q, k]
        idx = idx.transpose(-1, -2).contiguous() # [B, k, np_q]
        #NOTE 将idx对应成一个batch中的索引编号
        idx_base = torch.arange(0, bs, device=coor_q.device).view(-1, 1, 1) * np_k
        idx += idx_base  # [bs[0]: +0; bs[1]: +np_k; ... bs[i]: +i*np_k ...]
        idx = idx.view(-1)
    return idx

def get_knn_feature(x, knn_index, x_q=None, k=cfg.knn_k):
    
    bs, num_points, dims = x.size()
    num_query = x_q.size(1) if x_q is not None else num_points
    
    feature = x.view(bs*num_points, dims)[knn_index, :] # knn_index维度[bs*k*num_query,] 即索引x_q中每个点在x中对应k个最临近点
    feature = feature.view(bs, k, num_query, dims)
    
    x_q = x if x_q is None else x_q
    x_q = x_q.view(bs, 1, num_query, dims).expand(-1, k, -1, -1)
    
    out = torch.cat((feature - x_q, x_q), dim=-1)
    return out


class PoseEmbedding(nn.Module):
    def __init__(self, in_chans, embed_dim, conv_dim=128, negative_slope=0.2):
        super(PoseEmbedding, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_chans, conv_dim, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv1d(conv_dim, embed_dim, 1)
        )
    
    def forward(self, x):
        return self.net(x)
    
class Attention(nn.Module):
    def __init__(self, dim_in, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_in // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.qkv = nn.Linear(dim_in, dim_in*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_in, dim_in)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        qk_scale_dot = (q @ k.transpose(-2, -1)) * self.scale #(B, H, N, CH) @ (B, H, CH, N) -> (B, H, N, N)
        attn = torch.softmax(qk_scale_dot, dim=-1)
        attn = self.attn_drop(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, C) # (B, H, N, N) @ (B, H, N, CH) -> (B, H, N, CH) -> (B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out
    
class CrossAttention(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim_out = dim_out
        self.num_heads = num_heads
        head_dims = dim_out // num_heads
        self.qk_scale = qk_scale or head_dims ** -0.5
        
        self.q_map = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.k_map = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.v_map = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, q, v):
        # import pdb; pdb.set_trace()
        B, N, _ = q.size()
        C = self.dim_out
        k = v
        NK = k.size(1)
        
        q = self.q_map(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 
        k = self.k_map(k).reshape(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_map(v).reshape(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        qk_scale_dot = (q @ k.transpose(-2, -1)) * self.qk_scale # (B, H, N, CH) @ (B, H, CH, NK) -> (B, N, NK)
        attn = torch.softmax(qk_scale_dot, dim=-1)
        attn = self.attn_drop(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, C) # (B, H, N, NK) @ (B, H, NK, CH) -> (B, H, N, CH) -> (B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out
    
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
        x = self.fc2(x)
        x = self.drop(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        
        
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.knn_map = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.merge_map = nn.Linear(dim * 2, dim)
        
        mlp_hidden_features = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, hidden_features=mlp_hidden_features, act_layer=act_layer, drop=drop)
        
        
    def forward(self, x, knn_index=None):
        norm_x = self.norm1(x)
        x_1 = self.attn(norm_x)
        
        if knn_index is not None:
            knn_f = get_knn_feature(norm_x, knn_index=knn_index) # [B, k, np, 2*dim]
            knn_f = self.knn_map(knn_f) # [B, k, np, dim]
            #NOTE: max pooling the knn query feature
            knn_f = knn_f.max(dim=1, keepdim=False)[0] #[B, np, dim]
            #NOTE: concatenate the semantic attention feature vector and knn feature
            x_1 = torch.cat([x_1, knn_f], dim=-1) #[B, np, 2*dim]
            x_1 = self.merge_map(x_1) #[B, np, dim]
            
        #NOTE: the residual connector
        x = x + self.drop_path(x_1) # 前面所有层的residual connection
        x = x + self.drop_path(self.mlp(self.norm2(x))) # 最后mlp的residual connection
        return x
            
        
        
    
class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, dim_q=None, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )
        dim_q = dim_q or dim
        self.norm_q = norm_layer(dim_q)
        self.norm_v = norm_layer(dim)
        self.attn = CrossAttention(
            dim, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.knn_map = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.merge_map = nn.Linear(2 * dim, dim)
        
        self.cross_knn_map = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.cross_merge_map = nn.Linear(2 * dim, dim)
        
        self.norm2 = norm_layer(dim)
        middle_hidden_features = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, hidden_features=middle_hidden_features, act_layer=act_layer, drop=drop)
        
    def forward(self, q, v, knn_index=None, cross_knn_index=None):
        
        norm_q = self.norm1(q)
        q_1 = self.self_attn(norm_q)
        
        if knn_index is not None:
            knn_f = get_knn_feature(norm_q, knn_index)
            knn_f = self.knn_map(knn_f) # [B, k, np, dim]
            knn_f = knn_f.max(dim=1, keepdim=False)[0] #[B, np, dim]
            q_1 = torch.cat((q_1, knn_f), dim=-1)
            q_1 = self.merge_map(q_1)
        
        q = q + self.drop_path(q_1)
        
        norm_q = self.norm_q(q)
        norm_v = self.norm_v(v)
        # import pdb; pdb.set_trace()
        q_2 = self.attn(norm_q, norm_v)
        
        if cross_knn_index is not None:
            knn_f = get_knn_feature(norm_v, cross_knn_index, x_q=norm_q)
            knn_f = self.cross_knn_map(knn_f)
            knn_f = knn_f.max(dim=1, keepdim=False)[0]
            q_2 = torch.cat((q_2, knn_f), dim=-1)
            q_2 = self.cross_merge_map(q_2)
            
        q = q + self.drop_path(q_2)
        q = q + self.drop_path(self.mlp(self.norm2(q)))
        
        return q
    
class QueryGenerator(nn.Module):
    def __init__(self, embed_dim, feat_dim, num_query):
        super().__init__()
        self.num_query = num_query
        self.increase_dim = nn.Sequential(
            nn.Conv1d(embed_dim, feat_dim, 1),
            nn.BatchNorm1d(feat_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(feat_dim,feat_dim, 1)
        )
        self.coarse_pred = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, 3 * num_query)
        )
        self.mlp_query = nn.Sequential(
            nn.Conv1d(feat_dim + 3, feat_dim, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(feat_dim, feat_dim, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(feat_dim, embed_dim, 1)
        )
        
    def forward(self, x, coarse_points=None):
        bs = x.shape[0]
        glob_feat = self.increase_dim(x.transpose(1, 2))
        glob_feat = torch.max(glob_feat, dim=-1)[0] # [B, 1024]
        if coarse_points is None:
            coarse_points = self.coarse_pred(glob_feat).reshape(bs, -1, 3)
        else:
            #NOTE: iterative update每次的预测试相对上一次的偏移量 
            coarse_points = coarse_points + self.coarse_pred(glob_feat).reshape(bs, -1, 3)
        # import pdb; pdb.set_trace()
        query_feat = torch.cat([
            torch.unsqueeze(glob_feat, dim=1).expand(-1, self.num_query, -1),
            coarse_points], dim=-1)
        query_feat = self.mlp_query(query_feat.transpose(1, 2)).transpose(1, 2)
        
        return query_feat, coarse_points