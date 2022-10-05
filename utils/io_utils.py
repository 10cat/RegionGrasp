import os
import numpy as np
import torch

to_cpu = lambda tensor: tensor.detach().cpu().numpy()

def load_npy_file(file):
    array = np.load(file, allow_pickle=True)
    return array

def load_npz_file(file, one_key=None):
    # --key: 如果load的直接结果不是numpy array格式，需要用key向下索引一级
    npz = np.load(file)
    if one_key is not None:
        array = npz[one_key]
        return array
    else:
        return {k: npz[k].item() for k in npz.files}

def load_pt_file(file):
    tensor = torch.load(file)

    return tensor

def create_video(path, fps=30,name='movie'):
    import os
    import subprocess

    src = os.path.join(path,'%*.png')
    movie_path = os.path.join(path,'%s.mp4'%name)
    i = 0
    while os.path.isfile(movie_path):
        movie_path = os.path.join(path,'%s_%02d.mp4'%(name,i))
        i +=1

    cmd = 'ffmpeg -f image2 -r %d -i %s -b:v 6400k -pix_fmt yuv420p %s' % (fps, src, movie_path)

    subprocess.call(cmd.split(' '))
    while not os.path.exists(movie_path):
        continue


