from multiprocessing.sharedctypes import Value
import numpy as np
import torch
import logging
import time
from functools import wraps
import inspect
import random
import torch.nn.functional as F
# import chamfer_distance as chd
from chamfer_distance import ChamferDistance as ch_dist
from pytorch3d.structures import Meshes
import sys
sys.path.append('.')
sys.path.append('..')
from option import MyOptions as cfg

to_cpu = lambda tensor: tensor.detach().cpu().numpy() # 好！直接用lambda代入法一句话代替函数

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def func_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        print('[Function: {name} start...]'.format(name=function.__name__))
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print('[Function: {name} finished, spent time: {time:.8f}s]'.format(name=function.__name__, time=t1 - t0))
        return result

    return function_timer

def retrieve_name(var):
    local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in local_vars if var_val is var]


def to_tensor(array, dtype=torch.float32):
    if not torch.is_tensor(array):
        array = torch.tensor(array)
    return array.to(dtype)

def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)): #对稀疏矩阵
        array = np.array(array.todense(), dtype=dtype)
    elif torch.is_tensor(array):
        array = array.detach().cpu().numpy()
    return array

def makepath(desired_path, isfile=False):
    '''
    if the path does not exist make it
    :param desired_path: can be path to a file or a folder name
    :return:
    '''
    import os
    if isfile:
        if not os.path.exists(os.path.dirname(desired_path)): os.makedirs(os.path.dirname(desired_path))
    else:
        if not os.path.exists(desired_path): os.makedirs(desired_path)
    return desired_path

def get_std(log_vars):
    p_std = 0
    if cfg.std_type == 'softplus':
        p_std = F.softplus(log_vars)
    if cfg.std_type == 'exp':
        p_std = torch.exp(cfg.std_exp_beta * log_vars)

    return p_std



def euler(rots, order="xyz", units="deg"):
    '''
    convert euler angles to corresponding rotation matrix    
    '''
    rots = np.asarray(rots)
    single_dim = False if len(rots.shape) > 1 else True
    rots = rots.reshape(-1, 3)
    rotmats = []

    for xyz in rots:
        if units == 'deg':
            xyz = np.radians(xyz) # Convert angles from degrees to radians.
        r = np.eye(3)
        for theta, axis in zip(xyz, order):
            c = np.cos(theta)
            s = np.sin(theta)
            # 默认extrinsic -> 左乘
            if axis == 'x':
                r = np.dot(np.array([1,0,0], [0,c,-s], [0,s,c]), r) # x: x对应的第一行，第一列不变
            if axis == 'y':
                r = np.dot(np.array([c,0,s], [0,1,0], [-s,0,c]), r) # y: 相对于x下移-右移
            if axis == 'z':
                r = np.dot(np.array([c,-s,0], [s,c,0], [0,0,1]), r) # z: 相对于y下移-右移
        rotmats.append(r)
    rotmats = np.stack(rotmats).astype(np.float32)
    if single_dim:
        return rotmats[0]
    else:
        return rotmats

def size_splits(tensor, split_sizes, dim=0):
    """
    Splits the tensor according to the chunks of split_sizes.
    Arguments:
        tensor(Tensor): tensor to split
        size_splits(list(int)): sizes of chunks (unit lengths)
        dim(int): dimension along which to split the tensor
    """
    """
    [Coding notes]
    Applied functions:
    - torch.narrow: https://pytorch.org/docs/stable/generated/torch.narrow.html
    - torch.cumsum: https://pytorch.org/docs/stable/generated/torch.cumsum.html
    """

    if dim < 0:
        dim += tensor.dim()
    
    dim_size = tensor.size(dim)
    if dim_size != torch.sum(torch.Tensor(split_sizes)):
        raise KeyError("Sum of split sizes exceeds tensor dim")

    splits = torch.cumsum(torch.Tensor([0] + split_sizes), dim=0)[:-1] # 不包括最后一个：cumsum最后的结果就是原tensor.dim()

    return tuple(tensor.narrow(int(dim), int(start), int(length)) 
                 for start, length in zip(splits, split_sizes))

def region_masked_pointwise(obj_pc, mask):
    obj_pc_masked = obj_pc * (mask)
    return obj_pc_masked

def edges_for(x, vpe):
    return (x[:, vpe[:, 0]] - x[:, vpe[:, 1]])

def CRot2rotmat(pose):
    """
    Transform the rotation vector to rotation matrix
    (still don't quite know how it is computed)
    """

    reshaped_input = pose.view(-1, 3, 2) # [B*16, 3, 2]

    b1 = F.normalize(reshaped_input[:, :, 0], dim=1) # [B*16, 3]

    dot_product = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True) #[B*16, 1]
    b2 = F.normalize(reshaped_input[:, :, 1] - dot_product * b1, dim=-1) # [B*16, 3]
    b3 = torch.cross(b1, b2, dim=1)

    return torch.stack([b1, b2, b3], dim=-1)

def rotmat2aa(rotmat):
    """
    Convert rotation matrix to angle axis
    """
    batch_size = rotmat.size(0)

    homogen_matrot = F.pad(rotmat.view(-1, 3, 3), [0, 1])
    pose = rotation_matrix_to_angle_axis(homogen_matrot).view(batch_size, 1, -1, 3).contiguous()
    
    return pose

def rotation_matrix_to_angle_axis(rotation_matrix):
    """Convert 3x4 rotation matrix to Rodrigues vector
    Args:
        rotation_matrix (Tensor): rotation matrix.
    Returns:
        Tensor: Rodrigues vector transformation.
    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`
    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = rotation_matrix_to_angle_axis(input)  # Nx3
    """
    # todo add check that matrix is a valid rotation matrix
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    return quaternion_to_angle_axis(quaternion)

def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """Convert 3x4 rotation matrix to 4d quaternion vector
    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201
    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.
    Return:
        Tensor: the rotation in quaternion
    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`
    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * (~mask_d0_d1)
    mask_c2 = (~mask_d2) * mask_d0_nd1
    mask_c3 = (~mask_d2) * (~mask_d0_nd1)
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q


def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    
    """Convert quaternion vector to angle axis of rotation.
    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h
    Args:
        quaternion (torch.Tensor): tensor with quaternions.
    Return:
        torch.Tensor: tensor with angle axis of rotation.
    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`
    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = quaternion_to_angle_axis(quaternion)  # Nx3"""
   
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def point2point_signed(x, y, x_normals=None, y_normals=None):
    N, P1, D = x.shape
    P2 = y.shape[1]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape")

    # ch_dist = chd.ChamferDistance()

    x_near, y_near, xidx_near, yidx_near = ch_dist(x, y)

    xidx_near_expanded = xidx_near.view(N, P1, 1).expand(N, P1, D).to(torch.long)
    x_near = y.gather(1, xidx_near_expanded)

    yidx_near_expanded = yidx_near.view(N, P2, 1).expand(N, P2, D).to(torch.long)
    y_near = x.gather(1, yidx_near_expanded)

    x2y = x - x_near
    y2x = y - y_near

    if x_normals is not None:
        y_nn = x_normals.gather(1, yidx_near_expanded) #
        in_out = torch.bmm(y_nn.view(-1, 1, 3), y2x.view(-1, 3, 1)).view(N, -1).sign()
        y2x_signed = y2x.norm(dim=2) * in_out
    else:
        y2x_signed = y2x.norm(dim=2)

    if y_normals is not None:
        x_nn = y_normals.gather(1, xidx_near_expanded)
        in_out = torch.bmm(x_nn.view(-1, 1, 3), x2y.view(-1, 3, 1)).view(N, -1).sign()
        x2y_signed = x2y.norm(dim=2) * in_out
    else:
        x2y_signed = x2y.norm(dim=2)

    return y2x_signed, x2y_signed, yidx_near, xidx_near

def signed_distance_batch(device, rhand_vs, rh_f, obj_vs, object_faces=None):
    rh_normals = Meshes(verts=rhand_vs, faces=rh_f).to(device).verts_normals_packed().view(-1, cfg.num_rhand_verts, 3)
    if object_faces is not None:
        # obj_mesh_faces = torch.Tensor(obj_mesh_faces)
        # obj_vs = obj_vs.tolist()
        obj_vs_list = [obj_vs[i] for i in range(obj_vs.shape[0])] # need to be consistent length list with obj_mesh_faces
        obj_normals = Meshes(verts=obj_vs_list, faces=object_faces).to(device).verts_normals_packed().view(-1, cfg.num_obj_verts, 3)
    else:
        obj_normals = None
    o2h_signed, h2o_signed, o_nearest_ids, h_nearest_ids = point2point_signed(rhand_vs, obj_vs, rh_normals, obj_normals)
    
    return o2h_signed, h2o_signed, o_nearest_ids, h_nearest_ids

def dataset_object_faces_batch(sample_ids, dataset, device):
    sample_ids = sample_ids.reshape(-1).tolist()
    obj_names = dataset.frame_objs[sample_ids]
    obj_mesh_faces = [torch.Tensor(dataset.object_meshes[name].faces).to(device) for name in obj_names]
    return obj_mesh_faces
    
contact_ids={'Body': 1,
             'L_Thigh': 2,
             'R_Thigh': 3,
             'Spine': 4,
             'L_Calf': 5,
             'R_Calf': 6,
             'Spine1': 7,
             'L_Foot': 8,
             'R_Foot': 9,
             'Spine2': 10,
             'L_Toes': 11,
             'R_Toes': 12,
             'Neck': 13,
             'L_Shoulder': 14,
             'R_Shoulder': 15,
             'Head': 16,
             'L_UpperArm': 17,
             'R_UpperArm': 18,
             'L_ForeArm': 19,
             'R_ForeArm': 20,
             'L_Hand': 21,
             'R_Hand': 22,
             'Jaw': 23,
             'L_Eye': 24,
             'R_Eye': 25,
             'L_Index1': 26,
             'L_Index2': 27,
             'L_Index3': 28,
             'L_Middle1': 29,
             'L_Middle2': 30,
             'L_Middle3': 31,
             'L_Pinky1': 32,
             'L_Pinky2': 33,
             'L_Pinky3': 34,
             'L_Ring1': 35,
             'L_Ring2': 36,
             'L_Ring3': 37,
             'L_Thumb1': 38,
             'L_Thumb2': 39,
             'L_Thumb3': 40,
             'R_Index1': 41,
             'R_Index2': 42,
             'R_Index3': 43,
             'R_Middle1': 44,
             'R_Middle2': 45,
             'R_Middle3': 46,
             'R_Pinky1': 47,
             'R_Pinky2': 48,
             'R_Pinky3': 49,
             'R_Ring1': 50,
             'R_Ring2': 51,
             'R_Ring3': 52,
             'R_Thumb1': 53,
             'R_Thumb2': 54,
             'R_Thumb3': 55}


    
