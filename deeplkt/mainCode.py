import torch
import torch.nn as nn
from deeplkt.datasets.dataset import *
from torch.utils.data import DataLoader
from deeplkt.models.pure_lkt import PureLKTNet
from deeplkt.models.lkt_alexsobel import LKTAlexSobelNet

from deeplkt.models.base_model import BaseModel

# from model_learnable_lkt import LearnableLKTNet
# from evaluate_pytorch import evaluate_video_index, plot_different_results
# from torch import optim
# from model_utils import save_model, update_model, save_results, load_model
# import matplotlib.pyplot as plt
# import time
from deeplkt.utils.util import dotdict
from deeplkt.tracker.lkt_tracker import LKTTracker
from deeplkt.config import *

#!/usr/bin/env python
# coding: utf-8


# from pytorch_practise import *

use_cuda = torch.cuda.is_available()
device = torch.device("cuda")

vot_root_dir = '../../data/VOT/'
alov_root_dir = '../../data/ALOV/'


vot = VotDataset(os.path.join(vot_root_dir,
                       'VOT_images/'),
                 os.path.join(vot_root_dir,
                       'VOT_ann/'),
                 os.path.join(vot_root_dir,
                       'VOT_results/'), 
                 device)

alov = AlovDataset(os.path.join(alov_root_dir,
                       'ALOV_images/'),
                   os.path.join(alov_root_dir,
                       'ALOV_ann/'),
                   os.path.join(alov_root_dir,
                       'ALOV_results/'), 
                       device)

from deeplkt.config import *
params = dotdict({
    'mode' : MODE,
    'max_iterations' : MAX_LK_ITERATIONS,
    'epsilon' : EPSILON,
    'info': "AlexSobel LKT"
})
# lr = 0.0005
# momentum = 0.5


nn = LKTAlexSobelNet(device, params)
tracker = LKTTracker(nn)
train_params = dotdict({
    'batch_size' : BATCH_SIZE,
    'val_split' : VALIDATION_SPLIT,
    'train_examples':TRAIN_EXAMPLES,
    'shuffle_train': SHUFFLE_TRAIN,
    'random_seed': RANDOM_SEED  
})

model = BaseModel(tracker, 'checkpoint', 'logs', train_params)
model.train_model(alov)


# train_loader = DataLoader(alov, batch_size=1, shuffle=False)

