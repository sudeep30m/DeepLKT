import torch
import torch.nn as nn
from deeplkt.models.lkt_layers import LKTLayers
from deeplkt.models.alexsobel import AlexSobel
from deeplkt.models.base_model import BaseModel
from deeplkt.utils.model_utils import img_to_numpy
import numpy as np
from deeplkt.configParams import *
import cv2
import torch.nn.functional as F


class LKTAlexSobelNet(LKTLayers):


    def __init__(self, device, params):
        super().__init__(device)
        self.params = params
        self.sobel = AlexSobel().to(self.device)
        # self.conv1, self.conv2 = self.sobel_kernels(3)

    def template(self, bbox):
        self.bbox = bbox

    def sobel_layer(self, x):
        sx, sy = self.sobel(x)
        out_x = []
        out_y = []

        for i in range(x.shape[0]):
            out_x.append(F.conv2d(x[i:i+1, :, :, :], sx[i, :, :, :, :], \
                stride=1, padding=1, groups=self.sobel.num_channels))
            out_y.append(F.conv2d(x[i:i+1, :, :, :], sy[i, :, :, :, :], \
                stride=1, padding=1, groups=self.sobel.num_channels))
        out_x = torch.cat(out_x)
        out_y = torch.cat(out_y)
        return out_x, out_y


    def forward(self, img_i):
        # img_i = img_i.unsqueeze(0)
        img_tcr = self.bbox
        # print("Image i = ", img_i)
        # img_quad = x[2]
        B, C, h, w = img_tcr.shape

        p_init = torch.zeros((B, 6), device=self.device)
        # print(img_i.shape)
        # print(img_tcr.shape)
        # print(p_init.shape)
        # img_tcr, coords = self.crop_function(img_t, img_quad)
        sz = EXEMPLAR_SIZE
        sx = INSTANCE_SIZE
        centre = torch.Tensor([(sx / 2.0), (sx / 2.0)], device=self.device)
                
        xmin = centre[0] - (sz / 2.0)
        xmax = centre[0] + (sz / 2.0)
        
        coords = torch.tensor([xmin, xmin, xmax, xmax], device=self.device)  #exclusive
        # coords = coords.unsqueeze(0)
        # coords = coords.repeat(BATCH_SIZE, 1)

        img_quad = torch.tensor([xmin, xmax, xmin, xmin, xmax, xmin, xmax, xmax], device=self.device) #inclusive
        img_quad = img_quad.unsqueeze(0)
        img_quad = img_quad.repeat(B, 1)

        quad = img_quad
        omega_t = self.form_omega_t(coords, B)
        sobel_tx, sobel_ty = self.sobel_layer(img_tcr)
        J = self.J_matrix(omega_t, sobel_tx, sobel_ty, self.params.mode)
        J_pinv = self.J_pinv(J, self.params.mode)
        itr = 1
        p = p_init
        W = self.warp_matrix(p_init, self.params.mode)
        N = omega_t.shape[1]
        omega_t = torch.cat((omega_t, torch.ones((B, N, 1), device=self.device)), 2)  # (B x N x 3)
        
        while(self.params.max_iterations > 0):

            omega_warp = omega_t.bmm(W)
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
            if (itr >= self.params.max_iterations or \
            (quad_new - quad).norm() <= self.params.epsilon):
                quad = quad_new
                break
            itr += 1
            p = p_new
            quad = quad_new
        # print("--------------------------------------------------------------------------------")
        # print(itr)

        # img_tcr, _ = self.crop_function(img_i, quad_new)
        # self.bbox = img_tcr
        # self.p = p_new
        # print(img_tcr.shape)
        return quad

# if __name__ == '__main__':
#     device = torch.device("cuda")
#     model = LKT