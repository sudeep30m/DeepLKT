import os
from os.path import join
from deeplkt.utils.logger import Logger
from torch.optim import SGD
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
        self.optimizer = SGD(self.nn.model.parameters(),\
                                lr=params.lr,\
                                momentum=params.momentum,\
                                weight_decay=params.l2)
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

        lc = last_checkpoint(self.checkpoint_dir)
        if(lc != -1):
            self.load_checkpoint(lc)
            print("Checkpoint loaded = {}".format(lc))
        best_val = float("inf")
        for epoch in range(lc + 1, NUM_EPOCHS):
            print("EPOCH = ", epoch)
            train_loss = 0.0
            i = 0
            print("Training for epoch:{}".format(epoch))
            start_time = time.time()
            print("Total training batches = ", len(trainLoader))
            for batch in trainLoader:
                # print(batch)
                x, ynp = get_batch(dataset, batch)
                y = torch.tensor(ynp, device=self.nn.model.device).float()
                self.optimizer.zero_grad()
                self.nn.init(x[0], x[2])
                y_pred, _, _, _ = self.nn.train(x[1])
                # print(probs.shape)
                # pmx, pind = probs.max(1)
                # pmx = pmx[:, 0, 0, 0]
                # pind = pind[:, 0, 0, 0]
                # print(pmx, pind)
                # print(y)
                # print(y_pred)
                loss = self.loss(y_pred, y)
                # print(loss)
                train_loss += loss
                # for p in self.nn.model.parameters():
                #     if(p.requires_grad):
                #         print(p.grad)
                params = [x for x in self.nn.model.parameters() if x.requires_grad]
                # print(len(params))
                # grads = torch.autograd.grad(loss,\
                #                             params,\
                #                             retain_graph=True,\
                #                             create_graph=True, allow_unused=True)
                # print(grads)
                loss.backward()
                # print(self.nn.model.vgg.sobelx.grad[0, pind[0], 0, :, :])
                # print(self.nn.model.vgg.sobelx.grad[0, pind[1], 0, :, :])
                # print(self.nn.model.vgg.sobelx.grad[0, pind[2], 0, :, :])
                # print(self.nn.model.vgg.sobelx.grad[0, pind[3], 0, :, :])

                self.optimizer.step()
                print(i)
                i += 1

            train_loss /= i
            print("Training time for {} epoch = {}".format(epoch, time.time() - start_time))
            print("Training loss for {} epoch = {}".format(epoch, train_loss))

            self.save_checkpoint(epoch)
            if(epoch > 15):
                self.delete_checkpoint(epoch - 15)
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
                    y_pred, _, _, _ = self.nn.train(x[1])
                    loss = self.loss(y_pred, y)
                    valid_loss += loss
                    i += 1
            valid_loss /= i
            if(valid_loss < best_val):
                best_val = valid_loss
                print("Epoch = ", epoch)
                print("Best validation loss = ", best_val)
                self.save_checkpoint(epoch, best=True)
                self.best = epoch
            # print("Total validation batches = ", i)
            print("Validation time for {} epoch = {}".format(epoch, time.time() - start_time))


            info = {'train_loss': train_loss, 
                    'valid_loss': valid_loss}
            for tag, value in info.items():
                self.writer.scalar_summary(tag, value, epoch + 1)


    def eval_model(self, dataset, vid):
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
        sobel_out_dir = join(out_video, info)
        make_dir(imgs_out_dir)
        make_dir(sobel_out_dir)
        print("Evaluating dataset for video ", vid)
        data_x, quad_old = dataset.get_data_point(vid, 0)
        quads.append(quad_old)
        # data_x[0] = data_x[0][np.newaxis, :, :, :]
        # bbox = data_x[2][np.newaxis, :]
        # self.nn.init(data_x[0], bbox)
        self.nn.cnt = 0
        start_t = time.time()
        
        with torch.no_grad():
            for img_pair in range(num_img_pair):
                data_x, quad_old = dataset.get_data_point(vid, img_pair)
                data_x[0] = data_x[0][np.newaxis, :, :, :]
                bbox = data_x[2][np.newaxis, :]
                data_x[1] = data_x[1][np.newaxis, :, :, :]

                if(img_pair == 0):
                    self.nn.init(data_x[0], bbox)
                
                outputs = self.nn.track(data_x[1])
                if(len(outputs) == 8):
                    quad, sx, sy, img_tcr, sx_ker, sy_ker,\
                        img_i, quad_uns = outputs
                    sx_ker = tensor_to_numpy(sx_ker[0])
                    sy_ker = tensor_to_numpy(sy_ker[0])
                    # print(sx_ker.shape)                
                    np.save(join(sobel_out_dir, str(img_pair) + "-sx.npy"),\
                            sx_ker)
                    np.save(join(sobel_out_dir, str(img_pair) + "-sy.npy"),\
                            sy_ker)
 
                elif(len(outputs) == 6):
                    quad, sx, sy, img_tcr, img_i, quad_uns = outputs


                img_tcr = img_to_numpy(img_tcr[0])
                img_i = img_to_numpy(img_i[0])

                cv2.imwrite(join(imgs_out_dir,\
                    str(img_pair) +".jpeg"), img_tcr)
                cv2.imwrite(join(imgs_out_dir,\
                    str(img_pair) +"_i.jpeg"), img_i)

                np.save(join(sobel_out_dir, str(img_pair) + "-quad.npy"),\
                        quad_uns[0, :])

                sx = img_to_numpy(sx[0])
                sy = img_to_numpy(sy[0])

                for i in range(3):
                    cv2.imwrite(join(sobel_out_dir,\
                        str(img_pair) + "-x-" +str(i) +".jpeg"), sx[:, :, i])
                    cv2.imwrite(join(sobel_out_dir,\
                        str(img_pair) + "-y-" +str(i) +".jpeg"), sy[:, :, i])

                try:
                    iou = calc_iou(quad[0], quad_old)
                    iou_list.append(iou)
                except Exception as e: 
                    print(e)
                    break

        end_t = time.time()

        mean_iou = np.sum(iou_list) / num_img_pair
        write_to_output_file(quads, out_video + "/results.txt")
        
        outputBboxes(in_video +"/", out_video + "/images/", out_video + "/results.txt")

        print("Total time taken = ", end_t - start_t)
        print("Mean IOU = ", mean_iou)

        # plt.plot(iou_list)
        # plt.savefig(out_video + "/iou_plot.png")
        # plt.close()
        return mean_iou




    def save_checkpoint(self, epoch, filename='checkpoint.pth', best=False):
        folder = self.checkpoint_dir
        if not os.path.exists(folder):
            os.mkdir(folder)

        if(best):
            if (self.best != -1):
                self.delete_checkpoint(self.best, best=True)
            filepath = join(folder, 'best-' + str(epoch) + '-' + filename)
        else:
            filepath = join(folder, str(epoch) + '-' + filename)

        torch.save({ 'state_dict' : self.nn.model.state_dict()}, filepath)
    
    def load_checkpoint(self, epoch, filename='checkpoint.pth', best=False):
        folder = self.checkpoint_dir
        if(best):
            filepath = join(folder, 'best-' + str(epoch) + '-' + filename)
        else:
            filepath = join(folder, str(epoch) + '-' + filename)
        print(filepath)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))            
        checkpoint = torch.load(filepath)

        # own_state = self.nn.model.state_dict()
        # for i, p in enumerate(checkpoint['state_dict'].items()):
        #     name, param = p
        #     # if name not in own_state:
        #     #      continue
        #     # if isinstance(param, Parameter):
        #     #     # backwards compatibility for serialized parameters
        #     #     param = param.data
        #     if(i < 4):
        #         own_state[name].copy_(param)
        self.nn.model.load_state_dict(checkpoint['state_dict'])

    def delete_checkpoint(self, epoch, filename='checkpoint.pth', best=False):
        folder = self.checkpoint_dir
        if not os.path.exists(folder):
            os.mkdir(folder)

        if(best):
            filepath = join(folder, 'best-' + str(epoch) + '-' + filename)
        else:
            filepath = join(folder, str(epoch) + '-' + filename)
        if(os.path.exists(filepath)):
            os.remove(filepath)
        # torch.save({ 'state_dict' : self.nn.model.state_dict()}, filepath)
