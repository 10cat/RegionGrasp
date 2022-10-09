import numpy as np
import torch
import logging
import time
from functools import wraps
import inspect
import random

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


    
