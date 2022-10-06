import os
from dataclasses import dataclass

from pyrsistent import T
@dataclass
class MyOptions:
    
    exp_name: str='exp_1' # 1
    mode: str = 'train'
    num_mask: int = 1
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-3
    fit_Condition: bool = True
    fit_cGrasp: bool = True

    """
    Model Hyperparams
    """
    # SDmapNet
    class SDmapNet:
        input_dim: int = 1088
        output_dim: int = 1
        layers: list = [512, 256, 128]

    output_root: str=f"/home/datassd/yilin/Outputs/ConditionHOI/"
    model_root: str= os.path.join(output_root, 'model', exp_name)

if __name__=="__main__":
    config = MyOptions()
    import pdb; pdb.set_trace()
    config


