#!/usr/bin/env python
# coding: utf-8


# from pytorch_practise import *
import torch
import torch.nn as nn
from deeplkt.datasets.dataset import *
from torch.utils.data import DataLoader
from deeplkt.models.pure_lkt import PureLKTNet
from deeplkt.models.base_model import BaseModel
import cv2
import glob
from os.path import join
# from model_learnable_lkt import LearnableLKTNet
# from evaluate_pytorch import evaluate_video_index, plot_different_results
# from torch import optim
# from model_utils import save_model, update_model, save_results, load_model
# import matplotlib.pyplot as plt
# import time
from deeplkt.utils.util import dotdict, make_dir
from deeplkt.utils.visualise import readDir, convertVideoToDir

from deeplkt.tracker.lkt_tracker import LKTTracker
from deeplkt.config import *

use_cuda = torch.cuda.is_available()
device = torch.device("cuda") if use_cuda  else torch.device("cpu") 

# vot_root_dir = '../../data/VOT/'
# alov_root_dir = '../../data/ALOV/'


# vot = VotDataset(os.path.join(vot_root_dir,
#                        'VOT_images/'),
#                  os.path.join(vot_root_dir,
#                        'VOT_ann/'),
#                  os.path.join(vot_root_dir,
#                        'VOT_results/'), 
#                  device)

# alov = AlovDataset(os.path.join(alov_root_dir,
#                        'ALOV_images/'),
#                    os.path.join(alov_root_dir,
#                        'ALOV_ann/'),
#                    os.path.join(alov_root_dir,
#                        'ALOV_results/'), 
#                        device)


# train_loader = DataLoader(alov, batch_size=1, shuffle=False)




params = dotdict({
    'mode' : MODE,
    'max_iterations' : MAX_LK_ITERATIONS,
    'epsilon' : EPSILON,
    'info': "Pure LKT"
})
# lr = 0.0005
# momentum = 0.5


nn = PureLKTNet(device, params)
tracker = LKTTracker(nn)

video_name = "../red_square.mp4"
dir_name = "../red_square"
outdir_name = "../red_square_results"

window_name = "ABC"
make_dir(dir_name)
make_dir(outdir_name)

convertVideoToDir(video_name, dir_name)
frames = readDir(dir_name)
first_frame = True
cnt = 0
for frame in frames:
    frame = np.expand_dims(frame, 0)
    if first_frame:
        file_pth = join(dir_name, "first_frame.npy")
        if not (os.path.isfile(file_pth)):

            # try:
            cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            init_rect = cv2.selectROI(window_name, frame[0], False, False)
            # init_rect = [63, 63, 127, 127]
            init_rect = np.expand_dims(np.array(init_rect), 0)
            np.save(file_pth, init_rect)
            # except:
            # print("Bakchodi")
            # exit()

        init_rect = np.load(file_pth)
        print(init_rect)
        
        print(frame.shape)
        # init_rect = 
        tracker.init(frame, init_rect)
        first_frame = False
    else:
        outputs, _, _, _ = tracker.track(frame)
        # if 'polygon' in outputs:
        #     polygon = np.array(outputs['polygon']).astype(np.int32)
        #     cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
        #                     True, (0, 255, 0), 3)
        #     mask = ((outputs['mask'] > 0.30 * 255))
        #     mask = mask.astype(np.uint8)
        #     mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
        #     frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
        # else:
        print(outputs)
        bbox = list(map(int, outputs[0]))
        cv2.rectangle(frame[0], (bbox[0], bbox[1]),
                        (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                        (0, 255, 0), 1)
        out_pth = join(outdir_name, str(format(cnt, '08d')) + '.jpg')
        print(out_pth)
        cv2.imwrite(out_pth, frame[0])
        cnt += 1
        if(cnt > 200):
            break
        # cv2.waitKey(40)

# model = BaseModel(tracker, 'checkpoint', 'logs')
# model.eval_model(vot, 0)
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

