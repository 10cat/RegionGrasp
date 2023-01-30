from asyncio import FastChildWatcher
import os
from re import T
from cv2 import fastNlMeansDenoisingColored
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
    machine: int = '97'
    exp_name: str = 'thumb_cond_train0_1' # 1
    note: str = '解决thumb_cond_train0存在的bug: 计算signed_dist_loss时，rhand_vs_pred错与rhand_normals对应；rhand_vs错与rhand_normals_pred对应'
    run_type: str = 'train'
    batch_size: int = 32 # train: 32; test/val: 16
    test_part: bool = True
    select_k: float = 0.25 # 选取batch_size * select_k这么多
    
    use_cuda: bool = True
    visible_device: str="0"
    cuda_id: int = 0
    
    frame_names: str = 'frame_names_thumb.npz'
    obj_meshes: str = 'decimate_meshes'
    train_select_ids: bool = True
    num_mask: int = 1
    num_rhand_verts: int = 778
    num_obj_verts: int = 3000
    start_epoch: int = 1
    num_epoch: int = 40
    forward_Condition: bool = False
    forward_cGrasp: bool = True
    fit_Condition: bool = False
    fit_cGrasp: bool = True
    use_gtsdm: bool = False
    
    testmetrics: bool = True
    metrics_contact: bool = True
    metrics_inter: bool = True
    metrics_simul: bool = False
    metrics_cond: bool = False
    voxel_mode: str = 'voxels_hand'
    voxel_pitch: float = 0.01
    condition_dist: float = -0.005
    coverage_th: float = 0.6

    learning_rate: float = 1e-4
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
    SDmap_leaky_slope: float = 1

    #VAE
    VAE_encoder_sizes = [1024, 512, 256]
    VAE_enc_out_size: int = 64
    VAE_condition_size: int = 1024 # fusion layers output dim = 1024
    std_type: str = 'exp'
    std_exp_beta: float = 0.5

    """
    Loss Configuration
    """
    # ConditionNetLoss
    lambda_cond: float = 1.0 # = lambda_cond / lambda_vae
    lambda_om: float = 1
    lambda_feat: float = 1e-2

    # cGraspvaeLoss
    kl_coef: float = 0.005
    th_penet: float = -0.005
    th_contact: float = 0.01
    weight_penet: float = 1.5
    weight_contact: float = .1
    weight_region: float = 1.5
    lambda_dist_h: float = 35 * (1. - kl_coef)
    lambda_dist_o: float = 30 * (1. - kl_coef)
    lambda_mesh_rec_w: float = 30 * (1. - kl_coef)
    lambda_edge: float = 30 * (1. - kl_coef)
    vpe_path = "./config/verts_per_edge.npy"
    c_weights_path = "./config/rhand_weight.npy"


    """
    Metrics Configuration
    """
    use_h2osigned: bool = False
    penetrate_threshold: float = -0.005

    """
    Root Path
    """
    if machine == '97':
        dataset_dir: str = "/home/datassd/yilin/GrabNet"
        output_root: str = "/home/datassd/yilin/Outputs/ConditionHOI/"
        output_dir: str = "/home/datassd/yilin/Outputs/ConditionHOI/"+exp_name
        mano_rh_path: str = f"/home/datassd/yilin/Codes/_toolbox/mano/models/MANO_RIGHT.pkl"
    if machine == '41':
        dataset_dir: str = "/ssd_data/yilin/GrabNet"
        output_root: str = "/ssd_data/yilin/Outputs/ConditionHOI/"
        output_dir: str = "/ssd_data/yilin/Outputs/ConditionHOI/"+exp_name
        mano_rh_path: str = f"/home/yilin/smpl_models/mano/MANO_RIGHT.pkl"

    model_root: str= os.path.join(output_dir, 'model')
    check_interval: int = 2
    visual_interval_val: int = 100
    visual_sample_interval: int = 4

if __name__=="__main__":
    config = MyOptions()
    import pdb; pdb.set_trace()
    config