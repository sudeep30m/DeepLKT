#!/usr/bin/env python
# coding: utf-8


# from pytorch_practise import *
import torch
import torch.nn as nn
from deeplkt.datasets.dataset import *
from torch.utils.data import DataLoader
from deeplkt.models.pure_lkt import PureLKTNet
from deeplkt.models.base_model import BaseModel

# from model_learnable_lkt import LearnableLKTNet
# from evaluate_pytorch import evaluate_video_index, plot_different_results
# from torch import optim
# from model_utils import save_model, update_model, save_results, load_model
# import matplotlib.pyplot as plt
# import time
from deeplkt.utils.util import dotdict
from deeplkt.tracker.lkt_tracker import LKTTracker

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

# alov = AlovDataset(os.path.join(alov_root_dir,
#                        'ALOV_images/'),
#                    os.path.join(alov_root_dir,
#                        'ALOV_ann/'),
#                    os.path.join(alov_root_dir,
#                        'ALOV_results/'), 
#                        device)


# train_loader = DataLoader(alov, batch_size=1, shuffle=False)

params = dotdict({
    'mode' : 4,
    'max_iterations' : 20,
    'epsilon' : 0.05,
    'info': "Pure LKT"
})
# lr = 0.0005
# momentum = 0.5


nn = PureLKTNet(device, params)
tracker = LKTTracker(nn)
model = BaseModel(tracker, 'checkpoint', 'logs')
model.eval_model(vot, 0)
# model = BaseModel(nn, 'checkpoint', 'logs')


# model.load_checkpoint(6, folder='checkpoint/HeavySobel')
# iou = model.eval_model(vot, 0)

        

# for itr in range(10):
#     print(itr)
#     i = 0
#     bar = progressbar.ProgressBar(maxval=len(alov),     widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
#     bar.start()

#     for (x, y) in train_loader:
#         start_time = time.time()
#         x = [t.to(device) for t in x]
#         y = y.float().to(device)
#         optimizer.zero_grad()
#         y_pred = model(x)
#         loss = huber_loss(y_pred, y)
#         loss.backward()
#         optimizer.step()
#         i += 1
#         bar.update(i)
# #             print(time.time() - start_time) 

#         # except Exception as e:
#         #     print(e)
#         #     print(i)
#         #     torch.cuda.empty_cache()
#         #     continue
#     update_model(model, itr, optimizer.state_dict())


# plot_different_results(results2, 'results.png')

