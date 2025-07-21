# RegionGrasp: A Novel Task for Contact Region Controllable Hand Grasp Generation

[[Paper]](https://arxiv.org/abs/2410.07995)

Given as input a 3D object, together with its specific surface area selected as the intended contact region, to generate a diverse set of plausible hand grasps of the object, where the thumb finger tip touches the object surface on the contact region.


## Installation
1. First create a conda environment 

1. Activate the environment and install the dependencies using 
    ```
    bash install_pkgs.sh
    ```

## Dataset Preparation
Please first download the original dataset [GRAB](http://grab.is.tue.mpg.de) and [ObMan](https://www.di.ens.fr/willow/research/obman/data/).

To preprocess the dataset, please run the preprocessing script:
```
python dataset/obman_preprocessing.py
```
```
python dataset/grabnet_preprocessing.py
```

## Training
Before running the script, please configure your own data / mano / output directory in ```utils/cfgs.py``` and name your own machine with a customized number.


To train the model with our settings, refer to the config files in the ```cfgs``` folder. 

First, pretrain the object encoder by running:

```
python pretrain.py --cfgs pointmae_og_balance --exp_name XXX --mae --machine XX --cuda_id 0 --wandb
```

Then, train the entire model by running:
```
python traineval.py --cfgs thumb_mae_obman_og_hoienc --use_pos --mae --region_rand --exp_name XXX --machine XX --cuda_id 0 --wandb
```

## Inference

To obtain the generation result for your trained model, please change the run_mode as well as specify the checkpoint and corresponding config file:

```
python traineval.py --cfgs thumb_mae_obman_og_hoienc --run_mode test --exp_name [your_exp_name] --chkpt [your_checkpoint_name]
```

## Citation
If you find our code or paper helpful, please cite with the following bibtex and consider starring our repository:

```
@inproceedings{wang2024regiongrasp,
  title={RegionGrasp: A Novel Task for Contact Region Controllable Hand Grasp Generation},
  author={Wang, Yilin and Guo, Chuan and Cheng, Li and Jiang, Hai},
  booktitle={European Conference on Computer Vision (ECCV) Workshops},
  year={2024},
  note={arXiv preprint arXiv:2410.07995}
}
```








