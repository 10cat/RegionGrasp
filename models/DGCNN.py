import torch
import torch.nn as nn
from models.PointTr import knn_point
# from pointnet2_ops import pointnet2_utils
# from traineval_utils import pointnet_util
from pytorch3d.ops import sample_farthest_points

class DGCNN_grouper(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        K has to be 16
        '''
        self.input_trans = nn.Conv1d(3, 8, 1)

        self.layer1 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 32),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 64),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 64),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 128),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )
        
    # @staticmethod
    def fps_downsample(self, coor, x, num_group):
        
        xyz = coor.transpose(1, 2).contiguous() # b, n, 3
        # fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)
        # combined_x = torch.cat([coor, x], dim=1)
        # new_combined_x = (
        #     pointnet2_utils.gather_operation(
        #         combined_x, fps_idx
        #     )
        # )
        # fps_idx = pointnet_util.farthest_point_sample(xyz, num_group)
        _, fps_idx = sample_farthest_points(xyz, K=num_group)
        combined_x = torch.cat([coor, x], dim=1)
        # import pdb; pdb.set_trace()
        # SOLVED: 直接用torch.gather报错CUDA error: device-side assert triggered，而这个报错多是因为index out of bound
        # NOTE: 出这个bug的原因是torch.gather选择的dim选错了哈哈哈哈 
        new_combined_x = torch.gather(combined_x, dim=-1, index=fps_idx.unsqueeze(1).repeat(1, combined_x.shape[1], 1))
        
        new_coor = new_combined_x[:, :3]
        new_x = new_combined_x[:, 3:]
        # import pdb;pdb.set_trace()

        return new_coor, new_x
    
    
    # @staticmethod
    def get_graph_feature(self, coor_q, x_q, coor_k, x_k):

        # coor: bs, 3, np, x: bs, c, np

        k = 16
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
#             _, idx = knn(coor_k, coor_q)  # bs k np
            idx = knn_point(k, coor_k.transpose(-1, -2).contiguous(), coor_q.transpose(-1, -2).contiguous()) # B G M
            idx = idx.transpose(-1, -2).contiguous()
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature
    
    def forward(self, x, fps=True):
    
        # x: bs, 3, np

        # bs 3 N(128)   bs C(224)128 N(128)
        coor = x
        f = self.input_trans(x)
        
        # import pdb; pdb.set_trace()
        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer1(f)
        f = f.max(dim=-1, keepdim=False)[0]
        
        if not fps:
            coor_q, f_q = coor, f
        else:
            # import pdb; pdb.set_trace()
            coor_q, f_q = self.fps_downsample(coor, f, 512)
        f = self.get_graph_feature(coor_q, f_q, coor, f)
        f = self.layer2(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q

        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer3(f)
        f = f.max(dim=-1, keepdim=False)[0]

        if not fps:
            coor_q, f_q = coor, f
        else:
            # import pdb; pdb.set_trace()
            coor_q, f_q = self.fps_downsample(coor, f, 128)
        f = self.get_graph_feature(coor_q, f_q, coor, f)
        f = self.layer4(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q.transpose(1, 2).contiguous()

        return coor, f