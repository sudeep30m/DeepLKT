import os
from os.path import join
from deeplkt.utils.logger import Logger
from torch.optim import SGD, Adam
import torch.nn as nn
from deeplkt.utils.util import make_dir, write_to_output_file
from deeplkt.utils.visualise import outputBboxes, writeImagesToFolder
from deeplkt.config import *
from deeplkt.utils.model_utils import img_to_numpy, tensor_to_numpy
from deeplkt.utils.model_utils import splitData, calc_iou, last_checkpoint, get_batch
from deeplkt.utils.bbox import get_min_max_bbox, cxy_wh_2_rect, get_region_from_corner
import time
import torch
import numpy as np
import cv2
from scipy.special import huber

class BaseModel():


    def __init__(self, model, checkpoint_dir, logs_dir, params):
        super().__init__()

        # self.nn = PureLKTNet(device, params).to(device)
        self.nn = model
        self.checkpoint_dir = join(checkpoint_dir, self.nn.params.info)
        make_dir(self.checkpoint_dir)
        logs_dir = join(logs_dir, self.nn.params.info)
        make_dir(logs_dir)
        self.logs_dir = logs_dir
        self.writer = Logger(logs_dir)
        self.optimizer = Adam(self.nn.model.parameters())
        self.loss = nn.SmoothL1Loss()
        self.params = params
        self.best = -1


    def train_model(self, dataset):
        self.nn.model = self.nn.model.train()

        trainLoader, validLoader = splitData(dataset, self.params)
        total = min(self.params.train_examples, len(dataset))

        train_total = int(total * (1.0 - self.params.val_split))
        val_total = total - train_total
        print("Train dataset size = ", train_total)
        print("Valid dataset size = ", val_total)

        lc = -1
        # lc = last_checkpoint(self.checkpoint_dir)
        # if(lc != -1):
        #     self.load_checkpoint(lc)
        #     print("Checkpoint loaded = {}".format(lc))
        best_val = float("inf")
        for epoch in range(lc + 1, NUM_EPOCHS):
            print("EPOCH = ", epoch)
            train_loss = 0.0
            i = 0
            print("Training for epoch:{}".format(epoch))
            start_time = time.time()
            print("Total training batches = ", len(trainLoader))
            for bind, batch in enumerate(trainLoader):
                # print(bind)
                x, ynp = get_batch(dataset, batch)
                y = torch.tensor(ynp, device=self.nn.model.device).float()
                self.optimizer.zero_grad()
                self.nn.init(x[0], x[2])
                # print(len(self.nn.train(x[1])))
                outputs = self.nn.train(x[1])
                y_pred = outputs[0]
                scale_z = outputs[-1]
                scale_z = torch.from_numpy(scale_z).to(self.nn.model.device).float()
                scale_z = scale_z.view(scale_z.shape[0], 1)               
                loss = self.loss(y_pred / scale_z, y / scale_z)
                train_loss += loss
                params = [x for x in self.nn.model.parameters() if x.requires_grad]
                loss.backward()
                # print(self.nn.model.vgg.sobelx.grad[0, pind[0], 0, :, :])
                # print(self.nn.model.vgg.sobelx.grad[0, pind[1], 0, :, :])
                # print(self.nn.model.vgg.sobelx.grad[0, pind[2], 0, :, :])
                # print(self.nn.model.vgg.sobelx.grad[0, pind[3], 0, :, :])

                self.optimizer.step()
                # print(i)
                i += 1

            train_loss /= i
            print("Training time for {} epoch = {}".format(epoch, time.time() - start_time))
            print("Training loss for {} epoch = {}".format(epoch, train_loss))

            self.save_checkpoint(epoch)
            if(epoch > NUM_CHECKPOINTS):
                self.delete_checkpoint(epoch - NUM_CHECKPOINTS)
            print("Validation for epoch:{}".format(epoch))
            self.nn.model = self.nn.model.eval()
            valid_loss = 0.0
            i = 0
            start_time = time.time()

            print("Total validation batches = ", len(validLoader))

            with torch.no_grad():

                for batch in validLoader:
                    x, y = get_batch(dataset, batch)
                    y = torch.tensor(y, device=self.nn.model.device).float()
                    self.nn.init(x[0], x[2])
                    outputs = self.nn.train(x[1])
                    y_pred = outputs[0]
                    scale_z = outputs[-1]
                    scale_z = torch.from_numpy(scale_z).to(self.nn.model.device).float()
                    scale_z = scale_z.view(scale_z.shape[0], 1)               
                    loss = self.loss(y_pred / scale_z, y / scale_z)
                    valid_loss += loss
                    i += 1
            valid_loss /= i
            print("Validation time for {} epoch = {}".format(epoch, time.time() - start_time))
            print("Validation loss for {} epoch = {}".format(epoch, valid_loss))

            if(valid_loss < best_val):
                best_val = valid_loss
                print("Epoch = ", epoch)
                print("Best validation loss = ", best_val)
                self.save_checkpoint(epoch, best=True)
                self.best = epoch
            # print("Total validation batches = ", i)
            # print("Validation time for {} epoch = {}".format(epoch, time.time() - start_time))


            info = {'train_loss': train_loss, 
                    'valid_loss': valid_loss}
            for tag, value in info.items():
                self.writer.scalar_summary(tag, value, epoch + 1)


    def eval_model(self, dataset, vid, pairWise):
        self.nn.model = self.nn.model.eval()

        num_img_pair = dataset.get_num_images(vid)
        quads = []
        iou_list = []
        sobel_x = []
        sobel_y = []
        imgs = []
        in_video = dataset.get_in_video_path(vid)
        out_video = dataset.get_out_video_path(vid)
        info = self.nn.model.params.info
        imgs_out_dir = join(out_video, "img_tcr")        
        model_out_dir = join(out_video, info)
        make_dir(imgs_out_dir)
        make_dir(model_out_dir)
        print("Evaluating dataset for video ", vid)
        data_x, quad_gt = dataset.get_data_point(vid, 0)
        quads.append(data_x[2])
        # data_x[0] = data_x[0][np.newaxis, :, :, :]
        # bbox = data_x[2][np.newaxis, :]
        # self.nn.init(data_x[0], bbox)
        self.nn.cnt = 0
        start_t = time.time()
        loss = 0
        sz_loss = 0
        with torch.no_grad():
            for img_pair in range(num_img_pair):
                # print(img_pair)
                data_x, quad_gt = dataset.get_data_point(vid, img_pair)
                _, quad_pip_gt = dataset.get_train_data_point(vid, img_pair)
                data_x[0] = data_x[0][np.newaxis, :, :, :]
                bbox = data_x[2][np.newaxis, :]
                data_x[1] = data_x[1][np.newaxis, :, :, :]

                if(img_pair == 0 and not pairWise):
                    quad = bbox
                    self.nn.init(data_x[0], quad)
                elif(pairWise):
                    quad = bbox
                    self.nn.init(data_x[0], quad)
                # else:

                # try:
                outputs = self.nn.track(data_x[1])
                # except:
                #     print("Error!!!!!!")
                #     break
                if(len(outputs) == 9):
                    quad_new, sx, sy, img_pip_tcr, sx_ker, \
                        sy_ker, img_pip_i, quad_pip, scale_z = outputs
                    
                    sx_ker = tensor_to_numpy(sx_ker[0])
                    sy_ker = tensor_to_numpy(sy_ker[0])  
                    np.save(join(model_out_dir, str(img_pair) + "-sx.npy"),\
                            sx_ker)
                    np.save(join(model_out_dir, str(img_pair) + "-sy.npy"),\
                            sy_ker)
 
                elif(len(outputs) == 7):
                    quad_new, sx, sy, img_pip_tcr, img_pip_i,\
                        quad_pip, scale_z = outputs
                # print(quad_pip, quad_pip_gt)
                sz_loss +=  huber(100, quad_pip - quad_pip_gt).mean()
                loss += (scale_z[0]) * (scale_z[0]) * huber(100, quad_new - quad_gt).mean()
                img_pip_tcr = img_to_numpy(img_pip_tcr[0])
                try:
                    iou = calc_iou(quad_new[0], quad_gt)
                    iou_list.append(iou)
                except Exception as e: 
                    print(e)
                    break
                quads.append(quad_new[0])

                quad = quad_new

        end_t = time.time()
        loss /= num_img_pair
        sz_loss /= num_img_pair
        mean_iou = np.sum(iou_list) / num_img_pair
        write_to_output_file(quads, out_video + "/results.txt")
        
        outputBboxes(in_video +"/", out_video + "/images/", out_video + "/results.txt")
        print("Resized loss = ", sz_loss)
        print("Actual loss = ", loss)
        print("Total time taken = ", end_t - start_t)
        print("Mean IOU = ", mean_iou)

        # plt.plot(iou_list)
        # plt.savefig(out_video + "/iou_plot.png")
        # plt.close()
        return mean_iou




    def save_checkpoint(self, epoch, filename='checkpoint.pth', best=False, vid=-1):
        folder = self.checkpoint_dir
        if not os.path.exists(folder):
            os.mkdir(folder)
        filestr = str(epoch) + '-' + filename
        if(best):
            if (self.best != -1):
                self.delete_checkpoint(self.best, best=True, vid=vid)
            filestr = 'best-' + filestr
        if(vid != -1):
            filestr = 'v' + str(vid) + '-' + filestr
        # else:
        filepath = join(folder, filestr)

        torch.save({ 'state_dict' : self.nn.model.state_dict(),\
                    'optimizer' : self.optimizer.state_dict()}, filepath)
    
    def load_checkpoint(self, epoch, filename='checkpoint.pth', best=False, vid=-1):
        folder = self.checkpoint_dir

        filestr = str(epoch) + '-' + filename
        if(best):
            filestr = 'best-' + filestr
        if(vid != -1):
            filestr = 'v' + str(vid) + '-' + filestr
        filepath = join(folder, filestr)
        # print(filepath)

        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))            
        checkpoint = torch.load(filepath)

        self.nn.model.load_state_dict(checkpoint['state_dict'])
        if 'optimizer' in checkpoint:
           self.optimizer.load_state_dict(checkpoint['optimizer'])

    def delete_checkpoint(self, epoch, filename='checkpoint.pth', best=False, vid=-1):
        folder = self.checkpoint_dir
        if not os.path.exists(folder):
            os.mkdir(folder)
        filestr = str(epoch) + '-' + filename
        if(best):
            filestr = 'best-' + filestr
        if(vid != -1):
            filestr = 'v' + str(vid) + '-' + filestr
        filepath = join(folder, filestr)
        if(os.path.exists(filepath)):
            os.remove(filepath)
        # torch.save({ 'state_dict' : self.nn.model.state_dict()}, filepath)
