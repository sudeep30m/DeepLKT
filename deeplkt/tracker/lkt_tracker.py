import numpy as np
import torch.nn.functional as F

from deeplkt.config import *
from deeplkt.utils.model_utils import img_to_numpy, tensor_to_numpy
from deeplkt.utils.bbox import get_min_max_bbox
from deeplkt.tracker.base_tracker import SiameseTracker
import cv2
import os

class LKTTracker(SiameseTracker):
    def __init__(self, model):
        super(LKTTracker, self).__init__()
        # self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
        #     cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        # self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        # hanning = np.hanning(self.score_size)
        # window = np.outer(hanning, hanning)
        # self.window = np.tile(window.flatten(), self.anchor_num)
        # self.anchors = self.generate_anchor(self.score_size)
        self.model = model
        # self.model.eval()
        self.params = self.model.params
        self.parameters = self.model.parameters

    # def generate_anchor(self, score_size):
    #     anchors = Anchors(cfg.ANCHOR.STRIDE,
    #                       cfg.ANCHOR.RATIOS,
    #                       cfg.ANCHOR.SCALES)
    #     anchor = anchors.anchors
    #     x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
    #     anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
    #     total_stride = anchors.stride
    #     anchor_num = anchor.shape[0]
    #     anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    #     ori = - (score_size // 2) * total_stride
    #     xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
    #                          [ori + total_stride * dy for dy in range(score_size)])
    #     xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
    #         np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    #     anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    #     return anchor

    # def _convert_bbox(self, delta, anchor):
    #     delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
    #     delta = delta.data.cpu().numpy()

    #     delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
    #     delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
    #     delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
    #     delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
    #     return delta

    # def _convert_score(self, score):
    #     score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
    #     score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
    #     return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        print("Bounding box = ", bbox)
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))
        print("Squared average size of s_z = ", s_z)
        # calculate channel average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        # cv2.imwrite('/home/sudeep/Desktop/img1.jpg', img)
        # print(img.shape)
        print("Init centre = ", self.center_pos)
        z_crop = self.get_subwindow(img, self.center_pos,
                                    EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        # print(z_crop)
        # temp = img_to_numpy(z_crop[0])
        # print(temp.shape)
        # cv2.imwrite("temp.jpeg", temp)
        self.model.template(z_crop)
        self.cnt = 0
        # print(z_crop.shape)
        # cv2.imwrite('/home/sudeep/Desktop/img2.jpg', z_crop.cpu().detach().numpy().transpose(0,2,3,1)[0])
        # self.model.template(z_crop)

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = EXEMPLAR_SIZE / s_z
        s_x = s_z * (INSTANCE_SIZE / EXEMPLAR_SIZE)

        print("Track centre = ", self.center_pos)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    INSTANCE_SIZE,
                                    round(s_x), self.channel_average)
        # z_crop = self.get_subwindow(img, self.center_pos,
        #                             EXEMPLAR_SIZE,
        #                             s_z, self.channel_average)

        # self.model.template(z_crop)

        # print(x_crop.shape)
        # print(z_crop.shape)
        # from IPython import embed; embed()
        self.cnt += 1
        # print()
        # print(x_crop)
        # print("$$$$$$$$$$$$$$$$$$$")
        # print(self.model.bbox.shape)
        # print
        temp = img_to_numpy(x_crop[0])
        temp2 = img_to_numpy((self.model.bbox)[0])
        
        # print(temp.shape)
        cv2.imwrite("cropped/" + str(self.cnt) + ".jpeg", temp)
        cv2.imwrite("cropped/" + str(self.cnt) + "_1.jpeg", temp2)

        outputs = self.model(x_crop)
        outputs = tensor_to_numpy(outputs)
        # print(outputs)
        # print("Pure LKT output - ", outputs[0])
        bbox = get_min_max_bbox(outputs[0])

        print("Min max output - ", bbox)

        # print("{{{{{{{{{{{{{{", bbox)
        
        bbox[0] -= int(INSTANCE_SIZE / 2)
        bbox[1] -= int(INSTANCE_SIZE / 2)
        bbox[2] -= int(EXEMPLAR_SIZE)
        bbox[3] -= int(EXEMPLAR_SIZE)
        
        
        bbox = bbox / scale_z
        # print("!!!!!!!!!!!!!!", bbox)
        

        # lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]
        # bbox2 = [x.detach() for x in bbox]
        # # smooth bbox
        # print(self.size, bbox)
        try:
            width = self.size[0] * (1 - TRANSITION_LR) + (self.size[0] + bbox[2]) * TRANSITION_LR
            height = self.size[1] * (1 - TRANSITION_LR) + (self.size[1]+ bbox[3]) * TRANSITION_LR
        except:
            from IPython import embed;embed()
        # # print(width, height)
        # print("^^^^^^^^^^^^^^^^^^^^^^^^")
        # # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # # udpate state
        # print(self.center_pos, self.size)
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])
        # print(self.center_pos, self.size)

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        # best_score = score[best_idx]
        # from IPython import embed; embed()
        return bbox
        # return {
        #         'bbox': bbox,
        #         'best_score': best_score
        #        }

 
 
 
 
        # score = self._convert_score(outputs['cls'])
        # pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

        # def change(r):
        #     return np.maximum(r, 1. / r)

        # def sz(w, h):
        #     pad = (w + h) * 0.5
        #     return np.sqrt((w + pad) * (h + pad))

        # # scale penalty
        # s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
        #              (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # # aspect ratio penalty
        # r_c = change((self.size[0]/self.size[1]) /
        #              (pred_bbox[2, :]/pred_bbox[3, :]))
        # penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        # pscore = penalty * score

        # # window penalty
        # pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
        #     self.window * cfg.TRACK.WINDOW_INFLUENCE
        # best_idx = np.argmax(pscore)

 