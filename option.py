import os
import torch 
from torch import optim
from dataclasses import dataclass

# from pyrsistent import T
@dataclass
class MyOptions:

    """
    Base Configuration
    """
    w_wandb: bool = True
    exp_name: str = 'conditionnet_pretrain_1' # 1
    mode: str = 'train'
    num_mask: int = 1
    batch_size: int = 16
    start_epoch: int = 1
    num_epoch: int = 10
    fit_Condition: bool = True
    fit_cGrasp: bool = False

    learning_rate: float = 1e-3
    class optimizer_cond:
        type: str = 'adam'
    class optimizer_cgrasp:
        type: str = 'adam'


    """
    Model Hyperparams
    """
    # SDmapNet
    SDmap_input_dim: int = 1088
    SDmap_output_dim: int = 1
    SDmap_layer_dims = [512, 256, 128]
    SDmap_leaky_slope: float = 0.01
        

    """
    Loss Configuration
    """
    lambda_cond: float = 1.0 # = lambda_cond / lambda_vae
    lambda_om: float = 1
    lambda_feat: float = 1e-2
    
    """
    Root Path
    """
    output_root: str=f"/home/datassd/yilin/Outputs/ConditionHOI/"
    model_root: str= os.path.join(output_root, 'model', exp_name)
    check_interval: int = 2

if __name__=="__main__":
    config = MyOptions()
    import pdb; pdb.set_trace()
    config


