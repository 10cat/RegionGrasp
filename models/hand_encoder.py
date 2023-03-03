# from select import KQ_NOTE_LINK
import sys

sys.path.append('.')
sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.PointMAE import (Block, Encoder, Grouper, MaskTransformer,
                             TransformerEncoder)
from models.pointnet_encoder import STN3d
from models.PointTr import (Attention, CrossAttention, get_knn_feature,
                            get_knn_index)
from utils.utils import size_splits


class HOIBlock(Block):
    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop=0, attn_drop=0, drop_path=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, knn_k=8):
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, act_layer, norm_layer)
        
        self.knn_k = knn_k
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
        
        self.cross_attn = CrossAttention(dim, dim, num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm_q = norm_layer(dim)
        self.norm_v = norm_layer(dim)
        
    def forward(self, q, v, knn_index=None, cross_knn_index=None):
        
        # self attention of query only
        norm_q = self.norm1(q)
        q_1 = self.attn(norm_q)
        
        # cross attention between q and v
        norm_q = self.norm_q(q)
        norm_v = self.norm_v(v)
        q_2 = self.cross_attn(norm_q, norm_v)
        
        if knn_index is not None:
            knn_f = get_knn_feature(norm_q, knn_index, k=self.knn_k)
            knn_f = self.knn_map(knn_f) # [B, k, np, dim]
            knn_f = knn_f.max(dim=1, keepdim=False)[0] #[B, np, dim]
            q_1 = torch.cat((q_1, knn_f), dim=-1)
            q_1 = self.merge_map(q_1)
        
        q = q + self.drop_path(q_1)
        
        if cross_knn_index is not None:
            knn_f = get_knn_feature(norm_v, cross_knn_index, x_q=norm_q, k=self.knn_k)
            knn_f = self.cross_knn_map(knn_f)
            knn_f = knn_f.max(dim=1, keepdim=False)[0]
            q_2 = torch.cat((q_2, knn_f), dim=-1)
            q_2 = self.cross_merge_map(q_2)
            
        q = q + self.drop_path(q_2)
        q = q + self.drop_path(self.mlp(self.norm2(q)))
        
        return q
        

class HOITransformerEncoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, knn_k =8, knn_layer_num=-1):
        super().__init__()
        self.blocks = nn.ModuleList([
            HOIBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                knn_k=knn_k
            )
            for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()
        self.knn_layer = knn_layer_num
        if self.knn_layer > 0:
            print("Finally using point patches encoder for hand AND HOIENC!")
        else:
            print("Using point patches encoder for hand only")
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def forward(self, q, v, center_q=None, center_v=None, cfg=None):
        # import pdb;pdb.set_trace()
        if cfg.knn_layer_num > 0:
            knn_index = get_knn_index(center_q) if center_q is not None else None
            cross_knn_index = get_knn_index(center_q, coor_k=center_v) if center_v is not None else None
        else:
            knn_index, cross_knn_index = None, None
        for i, block in enumerate(self.blocks):
            if i < self.knn_layer:
                q = block(q, v, knn_index, cross_knn_index)
            else:
                q = block(q, v)
        return q
        
        
class HandEncoder_group(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.region_size = config.region_size
        self.out_dim = config.out_dim
        
        trans_cfg = config.transformer # config = 全局cfg.model.cnet.kwargs
        self.trans_dim = trans_cfg.trans_dim
        self.depth = trans_cfg.depth
        self.num_heads = trans_cfg.num_heads
        
        self.encoder_dims = trans_cfg.encoder_dims

        self.group_divider = Grouper(self.num_group, self.group_size)
        
        self.MAE_encoder = MaskTransformer(trans_cfg)
        
        self.HOIencoder = HOITransformerEncoder(**config.hoienc) if config.get('hoienc') is not None else None
        
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, self.out_dim, 1),
            nn.BatchNorm1d(self.out_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.out_dim, self.out_dim, 1)
        )
        
        
    def forward(self, verts, feat_o=None, center_o=None, cfg=None):
        verts = verts.transpose(2, 1)
        B, _, _ = verts.shape
        neighborhood, center_h, p_idx = self.group_divider(verts, return_idx=True)
        embed_feat, pos = self.MAE_encoder(neighborhood, center_h, noaug=True, use_pos = True)
        embed_feat = embed_feat + pos
        
        if self.HOIencoder is not None:
            assert feat_o is not None, "Please input object local feature to the HOIenocder"
            embed_feat = self.HOIencoder(embed_feat, feat_o, center_q=center_h, center_v=center_o, cfg=cfg)
            embed_feat = embed_feat + pos
        
        embed_feat = embed_feat.transpose(1, 2)
        feat = self.increase_dim(embed_feat)
        glob_feat = torch.max(feat, dim=2)[0]
        return glob_feat, None, None
    
class PN_HOIEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        channel = config.channel
        loc_feat_dim = config.loc_feat_dim
        embed_dim = config.hoienc.embed_dim
        out_dim = config.out_dim
        self.feature_transform = config.feature_transform
        self.stn = STN3d(channel=channel)
        self.conv1 = nn.Conv1d(channel, loc_feat_dim, 1)
        self.bn1 = nn.BatchNorm1d(loc_feat_dim)
        # self.pn_map = nn.Sequential(
        #     nn.Conv1d(loc_feat_dim, embed_dim, 1),
        #     nn.BatchNorm1d(embed_dim),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.Conv1d(embed_dim, embed_dim, 1)
        # )
        
        # self.first_conv = nn.Sequential(
        #     nn.Conv1d(3, 128, 1),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(128, 256, 1)
        # )
        # self.second_conv = nn.Sequential(
        #     nn.Conv1d(512, 512, 1),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(512, embed_dim, 1)
        # )
        
        self.pos_embedding = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim),
        )
        
        self.HOIencoder = HOITransformerEncoder(**config.hoienc) if config.get('hoienc') is not None else None
        
        self.increase_dim = nn.Sequential(
            nn.Conv1d(embed_dim, out_dim, 1),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(out_dim, out_dim, 1)
        )
    
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
            
        return x, trans, trans_feat, N
    
    def forward(self, verts, feat_o=None, center_o=None, cfg=None):
        
        # B, _, N = verts.shape
        # feat = self.first_conv(verts)
        # feat_glb = torch.max(feat, dim=2, keepdim=True)[0]
        # feat = torch.cat([feat_glb.expand(-1, -1, N), feat], dim=1)
        # feat = self.second_conv(feat)
        # feat_glb = torch.max(feat, dim=2, keepdim=False)[0]
        # # import pdb; pdb.set_trace()
        # feat_h =  feat.reshape(B, N, -1)
        # import pdb; pdb.set_trace()
        
        x, trans, trans_feat, N = self.pre_pointfeat_forward(verts)
        feat_h = x
        # feat_h = self.pn_map(x)
        feat_h = feat_h.transpose(2, 1)
        # import pdb; pdb.set_trace()
        
        pos = self.pos_embedding(verts.transpose(2, 1))
        feat_h = feat_h + pos
        
        embed_feat = self.HOIencoder(feat_h, feat_o, center_q=verts.transpose(2, 1), center_v=center_o, cfg=cfg)
        embed_feat = embed_feat + pos
        
        embed_feat = embed_feat.transpose(1, 2)
        feat = self.increase_dim(embed_feat)
        glob_feat = torch.max(feat, dim=2)[0]
        
        return glob_feat, None, None
        