import os
import numpy as np
import torch
import torch.nn as nn

def decode_hand_comp(dict, part):
    faces = dict[part][0]
    verts = dict[part][1]
    return faces, verts

"""
Some constant configuration settings
"""
# RGBA color
colors = {
    'pink': [1.00, 0.75, 0.80, 1.00],
    'skin': [0.96, 0.75, 0.69, 1.00],
    'purple': [0.63, 0.13, 0.94, 1.00],
    'red': [1.0, 0.0, 0.0, 1.00],
    'green': [.0, 1., .0, 1.00],
    'yellow': [1., 1., 0, 1.00],
    'brown': [1.00, 0.25, 0.25, 1.00],
    'blue': [.0, .0, 1., 1.00],
    'white': [1., 1., 1., 1.00],
    'orange': [1.00, 0.65, 0.00, 1.00],
    'grey': [0.75, 0.75, 0.75, 1.00],
    'black': [0., 0., 0., 1.00],
}

JOINTS_NUM = 15

hand_comp_npz = np.load('dataset/tools/hand_comp.npz', allow_pickle=True)
hand_comp_object = hand_comp_npz['arr_0']
hand_comp = hand_comp_object.tolist() # {'thumb':[faces, vertices], ...}
hand_comp_colors=['green', 'blue', 'purple', 'yellow', 'orange', 'red'] # 'thumb', 'index', 'middle', 'fourth', 'small'

thumb_center = [1295, 1296, 1297, 1298, 1299, 1300, 1301, 1302, 1303, 1304, 1305, 1306, 1307, 1308, 1311, 1312, 1313, 1314, 1315, 1316, 1317, 1318, 1319, 1320, 1321, 1322, 1323, 1324, 1325, 1326, 1329, 1330, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341, 1342, 1344, 1345, 1346, 1347, 1348, 1349, 1350, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 1358, 1359, 1360, 1361, 1362, 1363, 1364, 1365, 1366, 1367, 1368, 1369, 1370, 1371, 1372, 1373, 1374, 1375, 1376, 1377, 1378, 1379, 1380, 1381, 1382]

   

"""
Filepath settings
"""
machine = '97'

if machine == '97':

    DATASET_ROOT = "/home/datassd/yilin/GrabNet"
    GrabNet_ROOT = "/home/datassd/yilin/GrabNet"

    mano_dir = "/home/datassd/yilin/Codes/_toolbox/mano/models/MANO_RIGHT.pkl" # MANO right hand model path

    visual_root = "/home/datassd/yilin/Outputs"
    
    dataset_visual_dir = "/home/datassd/yilin/Outputs/GrabNet_visual"

    OUTPUT_ROOT = "/home/datassd/yilin/Outputs/ConditionHOI"

if machine == '41':
    DATASET_ROOT = "/home/yilin/GrabNet"

    mano_dir = "/home/yilin/smpl_models/mano/MANO_RIGHT.pkl" # MANO right hand model path
    
    visual_root = "/ssd_data/Outputs"
    
    dataset_visual_dir = "/ssd_data/Outputs/GrabNet_visual"

    OUTPUT_ROOT = "/home/yilin/Outputs/ConditionHOI"



dataset_dir = os.path.join(DATASET_ROOT, 'data')

obj_mesh_dir = os.path.join(DATASET_ROOT, 'tools/object_meshes/decimate_meshes')

"""
Dataset Generation configuration
"""
check = False

hand_sdf_th_alpha = -0.5

rtree_radius = 0.01

r_depth = None

gender_map = {
    "s1": 'male',
    "s2": 'male',
    "s3": 'female',
    "s4": 'female',
    "s5": 'female',
    "s6": 'female',
    "s7": 'female',
    "s8": 'male',
    "s9": 'male',
    "s10": 'male',
}

class Model(nn.Module):
    def __init__(self, opt):
        super().__init__()
        





