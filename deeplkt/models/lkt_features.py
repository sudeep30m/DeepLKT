import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from deeplkt.models.lkt_layers import LKTLayers
import torch.nn.functional as F
from deeplkt.models.base_model import BaseModel
from deeplkt.utils.model_utils import img_to_numpy
import numpy as np
from deeplkt.config import *
import cv2
import os
from deeplkt.models.alexnet import AlexNet

class LKTFeaturesNet(LKTLayers):


    def __init__(self, device, params):
        super().__init__(device)
        self.params = params
        self.pad = nn.ReflectionPad2d(1)
        self.device = device
        self.alex = AlexNet()
        # self.alex.load_pretrained()
        self.conv1, self.conv2 = self.sobel_kernels(self.params.num_channels)
        self.instance_size = INSTANCE_SIZE
        self.exemplar_size = EXEMPLAR_SIZE

    def template(self, bbox):
        self.bbox = bbox

    def sobel_kernels(self, C):
        conv1 = nn.Conv2d(C, C, kernel_size=3, stride=1, bias=False, groups=C)
        conv2 = nn.Conv2d(C, C, kernel_size=3, stride=1, bias=False, groups=C)
        conv_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float')
        conv_x = np.tile(np.expand_dims(np.expand_dims(conv_x, 0), 0), (C, 1, 1, 1))
        conv_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype='float') 
        conv_y = np.tile(np.expand_dims(np.expand_dims(conv_y, 0), 0), (C, 1, 1, 1))
        conv1.weight = nn.Parameter(torch.from_numpy(conv_x).\
                            to(self.device).float(), requires_grad=False)
        conv2.weight = nn.Parameter(torch.from_numpy(conv_y).\
                            to(self.device).float(), requires_grad=False)
        return conv1, conv2


    def forward(self, img_i):
        img_tcr = self.bbox
        # print("Before Img_tcr shape = ", img_tcr.shape)
        # print("Before Img_i shape = ", img_i.shape)
        img_i = self.alex.features(img_i)
        img_tcr = self.alex.features(img_tcr)
        # print("After Img_tcr shape = ", img_tcr.shape)
        # print("After Img_i shape = ", img_i.shape)
        B, C, h, w = img_tcr.shape
        self.exemplar_size = img_tcr.shape[2]
        self.instance_size = img_i.shape[2]

        p_init = torch.zeros((B, 6), device=self.device)
        sz = self.exemplar_size
        sx = self.instance_size
        centre = torch.Tensor([(sx / 2.0), (sx / 2.0)], device=self.device)
        # print("Center = ", centre)                
        xmin = centre[0] - sz / 2.0
        xmax = centre[0] + sz / 2.0
        
        coords = torch.tensor([xmin, xmin, xmax, xmax], device=self.device)  #exclusive
        # print("Coords = ", coords)                

        img_quad = torch.tensor([xmin, xmax, xmin, xmin, xmax, xmin, xmax, xmax], device=self.device) #inclusive
        # print("Img quad = ", img_quad)
        img_quad = img_quad.unsqueeze(0).repeat(B, 1)
        quads = []
        quad = img_quad
        quads.append(quad)
        omega_t = self.form_omega_t(coords, B)
        N = omega_t.shape[1]

        sobel_tx, sobel_ty = self.sobel_gradients(img_tcr, self.conv1, self.conv2)
        J = self.J_matrix(omega_t, sobel_tx, sobel_ty, self.params.mode)
        J_pinv = self.J_pinv(J, self.params.mode)
        itr = 1
        p = p_init
        W = self.warp_matrix(p_init, self.params.mode)
        N = omega_t.shape[1]
        omega_t = torch.cat((omega_t, torch.ones((B, N, 1), device=self.device)), 2)  # (B x N x 3)
        while(self.params.max_iterations > 0):

            omega_warp = omega_t.bmm(W)
            # print(img_i.shape)
            # print(omega_warp.shape)
            
            warped_i = self.sample_layer(img_i, omega_warp).permute(0, 2, 1) # (B x C x N)
            warped_i = warped_i.view(img_tcr.shape)

            r = (warped_i - img_tcr)
            r = r.permute(0, 2, 3, 1)            
            r = r.contiguous().view(B, C * h * w, 1)

            delta_p = (J_pinv.bmm(r)).squeeze(2)
            dp = self.warp_inv(delta_p)
            p_new = self.composition(p, dp)
            W = self.warp_matrix(p_new, self.params.mode)
            quad_new = self.quad_layer(img_quad, W, img_i.shape)

            if (itr >= self.params.max_iterations):
                quad = quad_new
                quads.append(quad)
                break
            itr += 1
            p = p_new
            quad = quad_new
            # quads.append(quad)

        # print("--------------------------------------------------------------------------------")
        # print(itr)
        sx_ker = self.conv1.weight.repeat(B, 1, 1, 1, 1)
        sy_ker = self.conv2.weight.repeat(B, 1, 1, 1, 1)
        return quad, sobel_tx, sobel_ty, img_tcr, sx_ker, sy_ker

# if __name__ == '__main__':
#     device = torch.device("cuda")
#     model = LKT