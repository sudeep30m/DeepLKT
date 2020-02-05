import numpy as np
import torch.nn.functional as F
import torch
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
        self.model = model.to(model.device)
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

    def _min(self, arr):
        return np.min(np.array(arr), 0)

    def _max(self, arr):
        return np.max(np.array(arr), 0)


    def _bbox_clip(self, cx, cy, width, height, boundary):

        B = cx.shape[0]
        cx = self._max([np.zeros(B), self._min([cx, boundary[:, 1]]) ])
        cy = self._max([np.zeros(B), self._min([cy, boundary[:, 0]]) ])
        width = self._max([np.zeros(B) + 10, self._min([width, boundary[:, 1]]) ])
        height = self._max([np.zeros(B) + 10, self._min([height, boundary[:, 0]]) ])

        # width = max(10, min(width, boundary[1]))
        # height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, imgs, bbox):
        """
        args:
            imgs(np.ndarray): batch of BGR image
            batch of bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[:, 0]+(bbox[:, 2]-1)/2.0,
                                    bbox[:, 1]+(bbox[:, 3]-1)/2.0])
        self.center_pos = self.center_pos.transpose()
        self.size = np.array([bbox[:, 2], bbox[:, 3]])
        self.size = self.size.transpose()

        w_z = self.size[:, 0] + CONTEXT_AMOUNT * np.sum(self.size, 1)
        h_z = self.size[:, 1] + CONTEXT_AMOUNT * np.sum(self.size, 1)
        s_z = np.round(np.sqrt(w_z * h_z))
        self.channel_average = []
        for img in imgs:
            self.channel_average.append(np.mean(img, axis=(0, 1)))
        self.channel_average = np.array(self.channel_average)
        z_crop = []
        for i, img in enumerate(imgs):
            z_crop.append(self.get_subwindow(img, self.center_pos[i],
                                    EXEMPLAR_SIZE,
                                    s_z[i], 
                                    self.channel_average[i], 
                                    ind=0))
        z_crop = torch.cat(z_crop)        # print(z_crop)
        self.model.template(z_crop)
        self.cnt = 0

    def track(self, imgs):
        """
        args:
            img(np.ndarray): Batch of BGR image
        return:
            bbox(list):[x, y, width, height]
        """

        w_z = self.size[:, 0] + CONTEXT_AMOUNT * np.sum(self.size, 1)
        h_z = self.size[:, 1] + CONTEXT_AMOUNT * np.sum(self.size, 1)
        s_z = np.sqrt(w_z * h_z)

        scale_z = EXEMPLAR_SIZE / s_z
        s_x = s_z * (INSTANCE_SIZE / EXEMPLAR_SIZE)

        x_crop = []
        for i, img in enumerate(imgs):
            x_crop.append(self.get_subwindow(img, self.center_pos[i],
                                    INSTANCE_SIZE,
                                    np.round(s_x)[i], 
                                    self.channel_average[i],
                                    ind=1))
        x_crop = torch.cat(x_crop)
        self.cnt += 1

        outputs, sx, sy, img_tcr = self.model(x_crop)
        outputs = tensor_to_numpy(outputs)

        bbox = get_min_max_bbox(outputs)
        bbox[:, 0] -= (INSTANCE_SIZE / 2)
        bbox[:, 1] -= (INSTANCE_SIZE / 2)
        bbox[:, 2] -= (EXEMPLAR_SIZE)
        bbox[:, 3] -= (EXEMPLAR_SIZE)
        
        bbox = bbox / scale_z[:, np.newaxis]

        cx = bbox[:, 0] + self.center_pos[:, 0]
        cy = bbox[:, 1] + self.center_pos[:, 1]
        width = self.size[:, 0] * (1 - TRANSITION_LR) + (self.size[:, 0] + bbox[:, 2]) * TRANSITION_LR
        height = self.size[:, 1] * (1 - TRANSITION_LR) + (self.size[:, 1]+ bbox[:, 3]) * TRANSITION_LR
        shapes = []
        for img in imgs:
            shapes.append(img.shape[:2])
        shapes = np.array(shapes)
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, shapes)

        self.center_pos = np.array([cx, cy]).transpose()
        self.size = np.array([width, height]).transpose()

        bbox = np.array([cx - width / 2,
                cy - height / 2,
                width,
                height]).transpose()
        return bbox,sx,sy, img_tcr
        
    def train(self, imgs):
        """
        args:
            img(np.ndarray): Batch of BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        # print(self.size.shape)
        # print(self.center_pos.shape)

        w_z = self.size[:, 0] + CONTEXT_AMOUNT * np.sum(self.size, 1)
        h_z = self.size[:, 1] + CONTEXT_AMOUNT * np.sum(self.size, 1)
        s_z = np.sqrt(w_z * h_z)

        scale_z = EXEMPLAR_SIZE / s_z
        s_x = s_z * (INSTANCE_SIZE / EXEMPLAR_SIZE)

        # print("Track centre = ", self.center_pos)
        x_crop = []
        for i, img in enumerate(imgs):
            x_crop.append(self.get_subwindow(img, self.center_pos[i],
                                    INSTANCE_SIZE,
                                    np.round(s_x)[i], self.channel_average[i]))
        x_crop = torch.cat(x_crop)

        outputs = self.model(x_crop)
        return outputs
 
    # def train(self, data_ba):
 
 
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

 