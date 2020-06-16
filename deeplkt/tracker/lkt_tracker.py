import numpy as np
import torch.nn.functional as F
import torch
from deeplkt.config import *
from deeplkt.utils.model_utils import img_to_numpy, tensor_to_numpy
from deeplkt.utils.bbox import get_min_max_bbox, cxy_wh_2_rect, get_region_from_corner
from deeplkt.utils.visualise import draw_bbox
from deeplkt.tracker.base_tracker import SiameseTracker
import cv2
import os

class LKTTracker(SiameseTracker):
    
    def __init__(self, model):
        super(LKTTracker, self).__init__()
        self.model = model.to(model.device)
        self.params = self.model.params
        self.parameters = self.model.parameters

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
        return cx, cy, width, height

    def init(self, imgs, bbox):
        """
        args:
            imgs(np.ndarray): batch of BGR image
            batch of bbox: (x, y, w, h) bbox
        """
        bbox = get_min_max_bbox(bbox)
        bbox = cxy_wh_2_rect(bbox)

        self.center_pos = np.array([bbox[:, 0]+(bbox[:, 2])/2.0,
                                    bbox[:, 1]+(bbox[:, 3])/2.0])
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

        outputs = self.model(x_crop)
        # print(len(outputs))
        scale_z = NEW_EXEMPLAR_SIZE / s_z
        x_crop = img_to_numpy(x_crop[0])
        bbox_lkt = []
        bbox_rescaled = []
        # for i in range(len(outputs[0])):
        bbox1 = tensor_to_numpy(outputs[0])
        bbox = get_min_max_bbox(bbox1)
        bbox[:, 0] -= (NEW_INSTANCE_SIZE / 2)
        bbox[:, 1] -= (NEW_INSTANCE_SIZE / 2)
        bbox[:, 2] -= (NEW_EXEMPLAR_SIZE)
        bbox[:, 3] -= (NEW_EXEMPLAR_SIZE)
        
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


        bbox = np.array([cx - width / 2,
                cy - height / 2,
                width,
                height]).transpose()
        bbox = get_region_from_corner(bbox)
        # bbox_rescaled.append(bbox) 
        # if(i == len(outputs[0]) - 1):
        self.center_pos = np.array([cx, cy]).transpose()
        self.size = np.array([width, height]).transpose()

        return (bbox,) + outputs[1:] + (x_crop, bbox1, scale_z)
        
    def train(self, imgs):
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
                                    np.round(s_x)[i], self.channel_average[i]))
        x_crop = torch.cat(x_crop)

        outputs = self.model(x_crop)
        scale_z = NEW_EXEMPLAR_SIZE / s_z
        return outputs + (scale_z,)
 
 