import torch
import torch.nn as nn
from deeplkt.datasets.dataset import *
from torch.utils.data import DataLoader

from deeplkt.utils.util import dotdict
from deeplkt.tracker.lkt_tracker import LKTTracker
from deeplkt.config import *
from deeplkt.utils.model_utils import img_to_numpy, tensor_to_numpy
from deeplkt.utils.bbox import get_min_max_bbox, cxy_wh_2_rect, get_region_from_corner

#!/usr/bin/env python
# coding: utf-8


# from pytorch_practise import *


use_cuda = torch.cuda.is_available()
print(use_cuda)
device = torch.device("cuda") if use_cuda else torch.device("cpu")


vot_root_dir = '../../data/VOT/'
alov_root_dir = '../../data/ALOV/'


vot = VotDataset(os.path.join(vot_root_dir,
                       'VOT_images/'),
                 os.path.join(vot_root_dir,
                       'VOT_ann/'),
                 os.path.join(vot_root_dir,
                       'VOT_results/') 
                 )

alov = AlovDataset(os.path.join(alov_root_dir,
                       'ALOV_images/'),
                   os.path.join(alov_root_dir,
                       'ALOV_ann/'),
                   os.path.join(alov_root_dir,
                       'ALOV_results/')
                       )

x, y = vot.get_data_point(0, 1)
sx, sy = vot.get_train_data_point(0, 1)
bbox = x[2]
bbox = np.expand_dims(bbox, 0)
print("BBox = ", bbox)
bbox = get_min_max_bbox(bbox)
bbox = cxy_wh_2_rect(bbox)

center_pos = np.array([bbox[:, 0]+(bbox[:, 2])/2.0,
                            bbox[:, 1]+(bbox[:, 3])/2.0])
center_pos = center_pos.transpose()
print(center_pos)
size = np.array([bbox[:, 2], bbox[:, 3]])
size = size.transpose()
w_z = size[:, 0] + CONTEXT_AMOUNT * np.sum(size, 1)
h_z = size[:, 1] + CONTEXT_AMOUNT * np.sum(size, 1)
s_z = np.sqrt(w_z * h_z)
scale_z = NEW_EXEMPLAR_SIZE / s_z
# print(w_z, h_z, s_z, scale_z)
sy = np.expand_dims(sy, 0)
bbox = get_min_max_bbox(sy)
bbox[:, 0] -= (NEW_INSTANCE_SIZE / 2)
bbox[:, 1] -= (NEW_INSTANCE_SIZE / 2)
bbox[:, 2] -= (NEW_EXEMPLAR_SIZE)
bbox[:, 3] -= (NEW_EXEMPLAR_SIZE)

bbox = bbox / scale_z[:, np.newaxis]
cx = bbox[:, 0] + center_pos[:, 0]
cy = bbox[:, 1] + center_pos[:, 1]
width = size[:, 0] * (1 - TRANSITION_LR) + (size[:, 0] + bbox[:, 2]) * TRANSITION_LR
height = size[:, 1] * (1 - TRANSITION_LR) + (size[:, 1]+ bbox[:, 3]) * TRANSITION_LR
bbox = np.array([cx - width / 2,
                    cy - height / 2,
                    width,
                    height]).transpose()
bbox_temp = np.array([cx,
                    cy,
                    width,
                    height]).transpose()

# print("Bbox = ", bbox)
bbox = get_region_from_corner(bbox)
print(bbox, y)
            



