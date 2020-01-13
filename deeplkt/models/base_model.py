import os
from os.path import join
from deeplkt.utils.logger import Logger
from torch.optim import SGD
import torch.nn as nn
from deeplkt.utils.util import make_dir, write_to_output_file
from deeplkt.utils.visualise import outputBboxes
from deeplkt.config import *
from deeplkt.utils.model_utils import splitData, calc_iou
from deeplkt.utils.bbox import get_min_max_bbox, cxy_wh_2_rect, get_region_from_corner
import time
import torch
import numpy as np

class BaseModel():


    def __init__(self, model, checkpoint_dir, logs_dir):
        super().__init__()

        # self.nn = PureLKTNet(device, params).to(device)
        self.nn = model
        self.checkpoint_dir = join(checkpoint_dir, self.nn.params.info)
        make_dir(logs_dir)
        logs_dir = join(logs_dir, self.nn.params.info)
        make_dir(logs_dir)
        self.logs_dir = logs_dir
        self.writer = Logger(logs_dir)
        self.optimizer = SGD(self.nn.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=L2)
        self.loss = nn.SmoothL1Loss()


    def train_model(self, dataset):
        trainLoader, validLoader = splitData(dataset)
        print("Train dataset size = ", len(trainLoader.dataset))
        print("Valid dataset size = ", len(validLoader.dataset))
        curr_max = -1
        for file in os.listdir(self.checkpoint_dir):
            if(file[0] != '.'):
                files = file.split('-')
                curr_max = max(curr_max, int(files[0]))

        self.load_checkpoint(curr_max, folder=self.checkpoint_dir)
        print("Checkpoint loaded = {}".format(curr_max))

        for epoch in range(curr_max + 1, NUM_EPOCHS):
            print("EPOCH = ", epoch)
            # bar = progressbar.ProgressBar(maxval=len(dataset), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            # bar.start()
            self.nn = self.nn.train()
            train_loss = 0.0
            i = 0
            print("Training for epoch:{}".format(epoch))
            start_time = time.time()

            for (x, y) in trainLoader:
                x = [t.to(self.nn.device) for t in x]
                y = y.float().to(self.nn.device)
                self.optimizer.zero_grad()
                try:
                    y_pred = self.nn(x)
                except Exception as e:
                    print(e)
                    continue
                loss = self.loss(y_pred, y)
                train_loss += loss
                loss.backward()
                self.optimizer.step()
                i += 1
                # bar.update(i)
            train_loss /= i
            print("Total training examples = ", i)
            print("Training time for {} epoch = {}".format(epoch, time.time() - start_time))

            self.save_checkpoint(epoch, folder=self.checkpoint_dir)

            print("Validation for epoch:{}".format(epoch))
            self.nn = self.nn.eval()
            valid_loss = 0.0
            i = 0
            start_time = time.time()
            with torch.no_grad():
                for (x, y) in validLoader:
                    x = [t.to(self.nn.device) for t in x]
                    y = y.float().to(self.nn.device)
                    try:
                        y_pred = self.nn(x)
                    except Exception as e:
                        print(e)
                        torch.cuda.empty_cache()
                        continue
                    loss = self.loss(y_pred, y)
                    valid_loss += loss
                    i += 1
            valid_loss /= i
            print("Total validation examples = ", i)
            print("Validation time for {} epoch = {}".format(epoch, time.time() - start_time))


            info = {'train_loss': train_loss, 
                    'valid_loss': valid_loss}
            for tag, value in info.items():
                self.writer.scalar_summary(tag, value, epoch + 1)


    def eval_model(self, dataset, vid):
        num_img_pair = dataset.get_num_images(vid)
        quads = []
        iou_list = []
        in_video = dataset.get_in_video_path(vid)
        # print()
        # print(in_video)
        out_video = dataset.get_out_video_path(vid)
        print("Evaluating dataset for video ", vid)
        # make_dir(out_video + "/kernel_visualisations/")
        data_x, quad_old = dataset.get_data_point(vid, 0)
        # print(data_x[0].shape)
        print(quad_old)
        bbox = get_min_max_bbox(quad_old)
        print(bbox)
        bbox = cxy_wh_2_rect(bbox)
        print(bbox)

        self.nn.init(data_x[0], bbox)
        self.nn.cnt = 0
        start_t = time.time()
        with torch.no_grad():
            for img_pair in range(1, 10):
                data_x, quad_old = dataset.get_data_point(vid, img_pair)
                # try:
                # print(quad_old)
                # print("---------------")
                quad = self.nn.track(data_x[1])
                # print(quad)
                quad_num = get_region_from_corner(quad)
                # print(quad_num)
                # print("************************")
                # except Exception as e: 
                #     print(e)
                #     break
                # print(img_pair)
                # quad_num = quad.detach().cpu().numpy()
                # quad_old_num = quad_old.cpu().numpy()
                quads.append(quad_num)
                try:
                    iou = calc_iou(quad_num, quad_old)
                    iou_list.append(iou)
                except Exception as e: 
                    print(e)
                    break

        end_t = time.time()
        print("Time taken = ", end_t - start_t)

        mean_iou = np.sum(iou_list) / num_img_pair
        
        write_to_output_file(quads, out_video + "/results.txt")
        outputBboxes(in_video +"/", out_video + "/images/", out_video + "/results.txt")

        # plt.plot(iou_list)
        # plt.savefig(out_video + "/iou_plot.png")
        # plt.close()
        return mean_iou




    def save_checkpoint(self, epoch, folder='checkpoint', filename='checkpoint.pth'):

        if not os.path.exists(folder):
            os.mkdir(folder)
        filepath = join(folder, str(epoch) + '-' + filename)

        torch.save({ 'state_dict' : self.nn.state_dict()}, filepath)
    
    def load_checkpoint(self, epoch, folder='checkpoint', filename='checkpoint.pth'):
        # folder = join(folder, self.nn.params.info)
        filepath = join(folder, str(epoch) + '-' + filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))            
        checkpoint = torch.load(filepath)
        self.nn.load_state_dict(checkpoint['state_dict'])
