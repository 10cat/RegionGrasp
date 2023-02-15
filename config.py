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

force_resample = False
num_resample_points = 8192
num_mask_points = 32
expand_times = 1
ratio_lower = 0.05
ratio_upper = 0.30

"""
Filepath settings
"""
machine = '41'

if machine == '97':

    DATASET_ROOT = "/home/datassd/yilin/GrabNet"
    GrabNet_ROOT = "/home/datassd/yilin/GrabNet"
    
    OBMAN_ROOT = "/home/datassd/yilin/obman"
    
    SHAPENET_ROOT = "/home/datassd/yilin/ShapeNetCore.v2"
    
    mano_root = "/home/datassd/yilin/Codes/_toolbox/mano"

    mano_dir = "/home/datassd/yilin/Codes/_toolbox/mano/models/MANO_RIGHT.pkl" # MANO right hand model path

    visual_root = "/home/datassd/yilin/Outputs"
    
    dataset_visual_dir = "/home/datassd/yilin/Outputs/GrabNet_visual"
    obman_visual_dir = "/ssd_data/yilin/Outputs/ObMan_visual"
    
    OUTPUT_ROOT = "/home/datassd/yilin/Outputs/ConditionHOI"
    
elif machine == '208':

    DATASET_ROOT = "/home/shihao/yilin/GrabNet"
    GrabNet_ROOT = "/home/shihao/yilin/GrabNet"
    
    OBMAN_ROOT = "/home/shihao/yilin/obman"
    
    SHAPENET_ROOT = "/home/shihao/yilin/ShapeNetCore.v2"
    
    mano_root = "/home/shihao/yilin/Codes/_toolbox/mano"

    mano_dir = "/home/shihao/yilin/Codes/_toolbox/mano/models/MANO_RIGHT.pkl" # MANO right hand model path

    visual_root = "/home/shihao/yilin/Outputs"
    
    dataset_visual_dir = "/home/shihao/yilin/Outputs/GrabNet_visual"
    obman_visual_dir = "/home/shihao/yilin/Outputs/ObMan_visual"
    
    OUTPUT_ROOT = "/home/shihao/yilin/Outputs/ConditionHOI"
    
elif machine == '195':

    DATASET_ROOT = "/home/jupyter-yiling/GrabNet"
    GrabNet_ROOT = "/home/jupyter-yiling/GrabNet"
    
    OBMAN_ROOT = "/home/jupyter-yiling/obman"
    
    SHAPENET_ROOT = "/home/jupyter-yiling/ShapeNetCore.v2"
    
    mano_root = "/home/jupyter-yiling/Codes/_toolbox/mano"

    mano_dir = "/home/jupyter-yiling/Codes/_toolbox/mano/models/MANO_RIGHT.pkl" # MANO right hand model path

    visual_root = "/home/jupyter-yiling/Outputs"
    
    dataset_visual_dir = "/home/jupyter-yiling/Outputs/GrabNet_visual"
    obman_visual_dir = "/home/jupyter-yiling/Outputs/ObMan_visual"
    
    OUTPUT_ROOT = "/home/jupyter-yiling/Outputs/ConditionHOI"

elif machine == '41':
    DATASET_ROOT = "/ssd_data/yilin/GrabNet"
    
    OBMAN_ROOT = "/ssd_data/yilin/obman"
    
    SHAPENET_ROOT = "/ssd_data/yilin/ShapeNetCore.v2"
    
    mano_root = "/home/yilin/smpl_models/mano"

    mano_dir = "/home/yilin/smpl_models/mano/MANO_RIGHT.pkl" # MANO right hand model path
    
    visual_root = "/ssd_data/yilin/Outputs"
    
    dataset_visual_dir = "/ssd_data/yilin/Outputs/GrabNet_visual"
    obman_visual_dir = "/ssd_data/yilin/Outputs/ObMan_visual"

    OUTPUT_ROOT = "/ssd_data/yilin/Outputs/ConditionHOI"



dataset_dir = os.path.join(DATASET_ROOT, 'data')

obj_mesh_dir = os.path.join(DATASET_ROOT, 'tools/object_meshes/decimate_meshes')

"""
Dataset Generation configuration
"""
check = False

hand_sdf_th_alpha = -0.5

rtree_radius = 0.005

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
        







bigfinger_vertices = [697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707,
                      708, 709, 710, 711, 712, 713, 714,
                      715, 716, 717, 718, 719, 720, 721, 722, 723, 724,
                      725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737,
                      738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750,
                      751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763,
                      764, 765, 766, 767, 768]

indexfinger_vertices = [46, 47, 48, 49, 56, 57, 58, 59, 86, 87, 133, 134, 155,
                        156, 164, 165, 166, 167, 174, 175, 189, 194, 195, 212,
                        213, 221, 222, 223, 224, 225, 226, 237, 238, 272, 273,
                        280, 281, 282, 283, 294, 295, 296, 297, 298, 299, 300,
                        301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311,
                        312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322,
                        323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333,
                        334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345,
                        346, 347, 348, 349, 350, 351, 352, 353, 354, 355]

middlefinger_vertices = [356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367,
                         372, 373, 374, 375, 376, 377, 381,
                         382, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394,
                         395, 396, 397, 398, 400, 401, 402, 403, 404, 405, 406, 407,
                         408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420,
                         421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433,
                         434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446,
                         447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459,
                         460, 461, 462, 463, 464, 465, 466, 467]

fourthfinger_vertices = [468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479,
                         482, 483, 484, 485, 486, 487, 491, 492,
                         495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507,
                         508, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520,
                         521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533,
                         534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546,
                         547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559,
                         560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572,
                         573, 574, 575, 576, 577, 578]

smallfinger_vertices = [580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591,
                        598, 599, 600, 601, 602, 603,
                        609, 610, 613, 614, 615, 616, 617, 618, 619, 620,
                        621, 622, 623, 624, 625, 626, 628, 629, 630, 631, 632, 633,
                        634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646,
                        647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659,
                        660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672,
                        673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685,
                        686, 687, 688, 689, 690, 691, 692, 693, 694, 695]