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
from deeplkt.utils.model_utils import best_checkpoint
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
                       'VOT_results/'), 
                 device)

# alov = AlovDataset(os.path.join(alov_root_dir,
#                        'ALOV_images/'),
#                    os.path.join(alov_root_dir,
#                        'ALOV_ann/'),
#                    os.path.join(alov_root_dir,
#                        'ALOV_results/'), 
#                        device)



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
    'num_channels' : 1,    
    'info': "GreyLKT"
})

net = PureLKTNet(device, params)
tracker = LKTTracker(net)
learned_model = BaseModel(tracker, 'checkpoint', 'logs', train_params)
bc = best_checkpoint(learned_model.checkpoint_dir)
learned_model.load_checkpoint(bc, best=True)
print("Checkpoint loaded  = ", bc)
# print(count_parameters(net))

params = dotdict({
    'mode' : MODE,
    'max_iterations' : MAX_LK_ITERATIONS,
    'epsilon' : EPSILON,
    'num_channels' : 1,
    'info': "GreyPureLKT"
})
# lr = 0.0005
# momentum = 0.5

net = PureLKTNet(device, params)
tracker = LKTTracker(net)
pure_model = BaseModel(tracker, 'checkpoint', 'logs', train_params)

pure_lkt = []
learned_lkt = []

for i in range(vot.get_num_videos()):

    pure_lkt.append(pure_model.eval_model(vot, i, pairWise=True))
    bc = best_checkpoint(learned_model.checkpoint_dir, vid=i)
    learned_model.load_checkpoint(bc, best=True, vid=i)
    learned_lkt.append(learned_model.eval_model(vot, i, pairWise=True))

results = {}
results['grey_pure_lkt'] = pure_lkt
results['grey_learned_lkt'] = learned_lkt
pkl_save('grey-synth-results-pair.pkl', results)
# plot_bar_graph(results, 'results-pairwise.png')

pure_lkt = []
learned_lkt = []

for i in range(25):
    pure_lkt.append(pure_model.eval_model(vot, i, pairWise=False))
    bc = best_checkpoint(learned_model.checkpoint_dir, vid=i)
    learned_model.load_checkpoint(bc, best=True, vid=i)
    learned_lkt.append(learned_model.eval_model(vot, i, pairWise=False))


results = {}
results['grey_pure_lkt'] = pure_lkt
results['grey_learned_lkt'] = learned_lkt
pkl_save('grey-synth-results-seq.pkl', results)
# plot_bar_graph(results, 'results-seq.png')

# train_loader = DataLoader(alov, batch_size=1, shuffle=False)

