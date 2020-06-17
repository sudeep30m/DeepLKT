import torch
import torch.nn as nn
from deeplkt.datasets.dataset import *
from torch.utils.data import DataLoader
from deeplkt.models.pure_lkt import PureLKTNet
from deeplkt.models.lkt_attention import LKTAttention
from deeplkt.models.base_model import BaseModel

from deeplkt.utils.util import dotdict
from deeplkt.tracker.lkt_tracker import LKTTracker
from deeplkt.config import *
from deeplkt.datasets.imagenet import ImageNetDataset

#!/usr/bin/env python
# coding: utf-8


# from pytorch_practise import *


use_cuda = torch.cuda.is_available()
print(use_cuda)
device = torch.device("cuda") if use_cuda else torch.device("cpu")


# vot_root_dir = '../../data/VOT/'
# alov_root_dir = '../../data/ALOV/'


# vot = VotDataset(os.path.join(vot_root_dir,
#                        'VOT_images/'),
#                  os.path.join(vot_root_dir,
#                        'VOT_ann/'),
#                  os.path.join(vot_root_dir,
#                        'VOT_results/')
#                 )

# alov = AlovDataset(os.path.join(alov_root_dir,
#                        'ALOV_images/'),
#                    os.path.join(alov_root_dir,
#                        'ALOV_ann/'),
#                    os.path.join(alov_root_dir,
#                        'ALOV_results/')
#                     )

from deeplkt.config import *
params = dotdict({
    'mode' : MODE,
    'max_iterations' : MAX_LK_ITERATIONS,
    'epsilon' : EPSILON,
    'num_classes': NUM_CLASSES,
    'num_channels': NUM_CHANNELS,
    'info': "AttentionLKT"
})
# lr = 0.0005
# momentum = 0.5


net = LKTAttention(device, params)
tracker = LKTTracker(net)
train_params = dotdict({
    'batch_size' : BATCH_SIZE,
    'val_split' : VALIDATION_SPLIT,
    'train_examples':TRAIN_EXAMPLES,
    'shuffle_train': SHUFFLE_TRAIN,
    'lr': LR,
    'momentum': MOMENTUM,
    'l2': L2,
    'random_seed': RANDOM_SEED
})

model = BaseModel(tracker, 'checkpoint', 'logs', train_params)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Features parameters = ", count_parameters(net.attention.features))
print("classifier parameters = ", count_parameters(net.attention.classifier))
imagenet = pkl_load('imagenet.pkl')
print(len(imagenet))

model.train_model(imagenet)


# train_loader = DataLoader(alov, batch_size=1, shuffle=False)

