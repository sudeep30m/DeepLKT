import torch
import torch.nn as nn
from deeplkt.datasets.dataset import *
from torch.utils.data import DataLoader
from deeplkt.models.pure_lkt import PureLKTNet
from deeplkt.models.lkt_alexsobel import LKTAlexSobelNet

from deeplkt.models.lkt_vggsobel import LKTVGGSobelNet
from deeplkt.models.lkt_vggimproved import LKTVGGImproved
from deeplkt.models.base_model import BaseModel

from deeplkt.utils.util import dotdict, pkl_load, pkl_save
from deeplkt.utils.visualise import plot_different_results, plot_bar_graph
from deeplkt.tracker.lkt_tracker import LKTTracker
from deeplkt.config import *#!/usr/bin/env python

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

# alov = AlovDataset(os.path.join(alov_root_dir,
#                        'ALOV_images/'),
#                    os.path.join(alov_root_dir,
#                        'ALOV_ann/'),
#                    os.path.join(alov_root_dir,
#                        'ALOV_results/')
#                        )



train_params = dotdict({
    'batch_size' : BATCH_SIZE,
    'val_split' : VALIDATION_SPLIT,
    'train_examples':TRAIN_EXAMPLES,
    'shuffle_train': SHUFFLE_TRAIN,
    'random_seed': RANDOM_SEED,
    'lr': LR,
    'momentum': MOMENTUM,
    'l2': L2

})


params = dotdict({
    'mode' : MODE,
    'max_iterations' : MAX_LK_ITERATIONS,
    'epsilon' : EPSILON,
    'num_classes': NUM_CLASSES,
    # 'info': "Learned LKT"
    'info': "VGGSobel LKT"
})

net = LKTVGGImproved(device, params)
tracker = LKTTracker(net)
learned_model = BaseModel(tracker, 'checkpoint', 'logs', train_params)
learned_model.load_checkpoint(169, best=True)
# print(count_parameters(net))

params = dotdict({
    'mode' : MODE,
    'max_iterations' : MAX_LK_ITERATIONS,
    'epsilon' : EPSILON,
    'info': "Pure LKT"
})
# lr = 0.0005
# momentum = 0.5

net = PureLKTNet(device, params)
tracker = LKTTracker(net)
pure_model = BaseModel(tracker, 'checkpoint', 'logs', train_params)
# from IPython import embed;embed()

base_model_name = "Pure LKT"
learned_model_name = "VGGSobel LKT"
seq = 19
pairWise = True
# pure_model.eval_model(vot, seq, pairWise=pairWise)
# learned_model.eval_model(vot, seq, pairWise=pairWise)


# from IPython import embed;embed()
vot.visualise_sobels(seq, base_model_name, learned_model_name)
# vot.visualise_images(seq, base_model_name, learned_model_name)
# vot.visualise_resized_images(seq, base_model_name, learned_model_name)
# vot.visualise_transitions(seq, base_model_name)
# vot.visualise_transitions(seq, learned_model_name)
# vot.visualise_transitions_resized(seq, base_model_name)
# vot.visualise_transitions_resized(seq, learned_model_name)



