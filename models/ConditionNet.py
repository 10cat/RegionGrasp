import sys
sys.path.append('.')
sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.functional as F
from option import MyOptions as cfg
from models.pointnet_encoder import PointNetEncoder
from utils.utils import func_timer, region_masked_pointwise
from timm.models.layers import trunc_normal_
from models.PointTr import get_knn_index, PoseEmbedding, Attention, CrossAttention, EncoderBlock, DecoderBlock, QueryGenerator
from models.DGCNN import DGCNN_grouper

class ConditionTrans(nn.Module):
    def __init__(self, in_chans=3, embed_dim=132, num_heads=6, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., glob_feat_dim=1024, depth={'encoder':3, 'decoder':3}, num_query=32, knn_layer=-1):
        
        #CHECK: 不打算和PointTr一样使用fps将2048个点降为128个点，而直接使用最初的2048个点(N=2048, C=32), 所以embed_dim可以相应的降低维度
        super().__init__()
        
        self.num_features = self.embed_dim = embed_dim
        self.knn_layer = knn_layer
        self.num_query = num_query
        
        self.pc_embed = DGCNN_grouper()
        self.pos_embed = PoseEmbedding(in_chans=in_chans, embed_dim=embed_dim)
        self.input_proj = nn.Sequential(
            nn.Conv1d(128, embed_dim, 1),
            nn.BatchNorm1d(embed_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(embed_dim, embed_dim, 1)
        )
        
        self.encoder = nn.ModuleList([
            EncoderBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate)
            for i in range(depth['encoder'])])
        
        self.decoder = nn.ModuleList([
            DecoderBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate)
            for i in range(depth['decoder'])])
        
        self.query_generate = QueryGenerator(embed_dim=embed_dim, feat_dim=glob_feat_dim, num_query=num_query)
        
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight.data, gain=1)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)
            
    def get_input_embedding(self, input):
        # coor, f = self.grouper(input.transpose(1, 2).contiguous())
        coor, f = self.pc_embed(input.transpose(1, 2).contiguous(), fps=False)
        # import pdb; pdb.set_trace()
        knn_index = get_knn_index(coor)
        # import pdb; pdb.set_trace()
        pos = self.pos_embed(coor.transpose(1, 2).contiguous()).transpose(1, 2)
        x = self.input_proj(f).transpose(1, 2) # 
        
        return coor, x, pos, knn_index
            
    def get_new_knn_index(self, pred_points, coor_k):
        new_knn_index = get_knn_index(pred_points)
        cross_knn_index = get_knn_index(pred_points, coor_k=coor_k)
        return new_knn_index, cross_knn_index
    
    def forward(self, input):
        bs = input.size(0)
        
        coor, x, pos, knn_index = self.get_input_embedding(input)
        
        for i, blk in enumerate(self.encoder):
            if i < self.knn_layer:
                x = blk(x + pos, knn_index)
            else:
                x = blk(x + pos)
        
        q, pred_pc = self.query_generate(x)
        new_knn_index, cross_knn_index = self.get_new_knn_index(pred_pc, coor_k=coor)
        
        for i, blk in enumerate(self.decoder):
            if i < self.knn_layer:
                q = blk(q, x, new_knn_index, cross_knn_index)
            else:
                q = blk(q, x)
        return q, pred_pc
    
class ConditionBERT(ConditionTrans):
    def __init__(self, in_chans=3, embed_dim=132, num_heads=6, mlp_ratio=2, qkv_bias=False, qk_scale=None, drop_rate=0, attn_drop_rate=0, glob_feat_dim=1024, depth={ 'encoder': 3,'decoder': 3 }, num_query=32, knn_layer=-1):
        super().__init__(in_chans, embed_dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, glob_feat_dim, depth, num_query, knn_layer)
        
    def forward(self, input):
        bs = input.size(0)
         
        coor, x, pos, knn_index = self.get_input_embedding(input)
        
        assert len(self.encoder) == len(self.decoder), "The encoder and decoder should have same depth in Iterative mode"
        
        for i, blk in enumerate(self.encoder):
            if i == 0:
                pred_pc = torch.zeros([bs, self.num_query, 3]).to('cuda')
            if i > self.knn_layer:
                knn_index, new_knn_index, cross_knn_index = None, None, None
            x = blk(x + pos, knn_index)
            q, pred_pc = self.query_generate(x, pred_pc)
            new_knn_index, cross_knn_index = self.get_new_knn_index(pred_pc, coor_k=coor)
            q = self.decoder[i](q, x, new_knn_index, cross_knn_index)
        
        return q, pred_pc


class ConditionNet(nn.Module):
    def __init__(self, input_channel_obj, input_channel_hand):
        super(ConditionNet, self).__init__()
        self.in_channel_obj = input_channel_obj
        self.in_channel_hand = input_channel_hand

        self.obj_encoder = PointNetEncoder(global_feat=False, feature_transform=False, channel=self.in_channel_obj)
        self.obj_masked_encoder = PointNetEncoder(global_feat=False, feature_transform=False, channel=self.in_channel_obj)
        self.hand_encoder = PointNetEncoder(global_feat=False, feature_transform=False, channel=self.in_channel_hand)
        self.mapnet = SDmapNet(input_dim=cfg.SDmap_input_dim, layer_dims=cfg.SDmap_layer_dims, output_dim=cfg.SDmap_output_dim)
        self.convfuse = nn.Conv1d(3778, 3000, 1)
        self.convfuse_m1 = nn.Conv1d(6000, 3000, 1)
        # self.convfuse_m2 = nn.Conv1d(4096, 3000, 1)
        self.bnfuse = nn.BatchNorm1d(3000)
        self.bnfuse_m1 = nn.BatchNorm1d(3000)
        # self.bnfuse_m2 = nn.BatchNorm1d(3000)

    def mask_obj_pts(self, x, mask):
        # import pdb; pdb.set_trace()
        x = x * (1 - mask) # multiplied by negate mask
        return x

    # @func_timer
    def feat_hand_fusion(self, x, hand):
        x = torch.cat((x, hand), dim=2).permute(0, 2, 1).contiguous()
        x = F.relu(self.bnfuse(self.convfuse(x)))
        x = x.permute(0, 2, 1).contiguous()
        return x

    # @func_timer
    def feat_om_fusion(self, x, feat_om):
        x = torch.cat((x, feat_om), dim=2).permute(0, 2, 1).contiguous()
        x = F.relu(self.bnfuse_m1(self.convfuse_m1(x)))
        # x = F.relu(self.bnfuse_m2(self.convfuse_m2(x)))
        x = x.permute(0, 2, 1).contiguous()
        return x

    
    def forward(self, obj_pc, hand_xyz, region_mask):
        """
        :param obj_pc: [B, 3, N]
        :param region_mask: [B, 1, N]
        :param hand_xyz: [B, 3, 778]

        :return: predicted 
        """
        
        B = obj_pc.size(0)
        N = obj_pc.size(2)

        # mask the obj pointcloud
        obj_masked_pc = region_masked_pointwise(obj_pc, region_mask)
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
        self.activate = nn.LeakyReLU(negative_slope=cfg.SDmap_leaky_slope)

    # @func_timer
    def forward(self, x, N):
        """
        :param x: fused feature vector
        :param N: corresponds to the dim of obj verts

        :return 
        """
        # batch_size = cfg.batch_size
        batch_size = x.shape[0]
        x = self.activate(self.bn1(self.conv1(x)))
        x = self.activate(self.bn2(self.conv2(x)))
        x = self.activate(self.bn3(self.conv3(x)))
        x = self.conv4(x)

        x = x.transpose(2, 1).contiguous()
        # x = torch.sigmoid(x)  # w/ sigmoid -- sigmoid the annotated target
        x = x.view(batch_size, N) # w/o sigmoid -- target directly use annotated target

        return x
        

if __name__ == "__main__":
    import os
    from tqdm import tqdm
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.visible_device
    B = 8
    Ndata = 180000
    
    obj_pc = torch.randn(Ndata, B, 2048, 3) # N = 2048
    gt_mask_pc = torch.randn(Ndata, B, 32, 3) # M = 32
    
    
    # net = ConditionTrans(in_chans=3, embed_dim=132, num_heads=4, mlp_ratio=2.,glob_feat_dim=1024, depth={'encoder':3, 'decoder':3}, num_query=32, knn_layer=1).to('cuda')
    net = ConditionBERT(in_chans=3, embed_dim=132, num_heads=4, mlp_ratio=2.,glob_feat_dim=1024, depth={'encoder':3, 'decoder':3}, num_query=32, knn_layer=1).to('cuda')
    mseloss = nn.MSELoss().to('cuda')
    
    for epoch in range(10):
        for i in tqdm(range(Ndata), desc=f"epoch {epoch}"):
            sample = obj_pc[i]
            sample = sample.to('cuda')
            q, pred_mask_pc = net.forward(sample)
            loss = mseloss(pred_mask_pc, gt_mask_pc.to('cuda'))
            loss.backward()
            torch.cuda.empty_cache()
    
    
    # obj_pc = torch.randn(B, 3, 3000)
    ## region_mask = torch.zeros(B, 1, 3000)
    # hand_xyz = torch.randn(B, 3, 778)

    # # pointnet = PointNetEncoder(global_feat=False, feature_transform=False, channel=3)
    # model = ConditionNet(input_channel_obj=3, input_channel_hand=3)

    # feat_om, map1, map2 = model(obj_pc, region_mask, hand_xyz)


    # import pdb; pdb.set_trace()
    # feats, trans, trans_feat = pointnet(obj_pc)
    # import pdb; pdb.set_trace()
    # feats.shape
    
