import torch
import torch.nn as nn
from deeplkt.models.lkt_layers import LKTLayers
from deeplkt.models.base_model import BaseModel
from deeplkt.utils.model_utils import img_to_numpy
import numpy as np
from deeplkt.config import *
import cv2


class PureLKTNet(LKTLayers):


    def __init__(self, device, params):
        super().__init__(device)
        self.params = params
        self.conv1, self.conv2 = self.sobel_kernels(3)

    def template(self, bbox):
        self.bbox = bbox

    def sobel_kernels(self, C):
        conv1 = nn.Conv2d(C, C, kernel_size=3, stride=1, bias=False, groups=C)
        conv2 = nn.Conv2d(C, C, kernel_size=3, stride=1, bias=False, groups=C)
        conv_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float')
        conv_x = np.tile(np.expand_dims(np.expand_dims(conv_x, 0), 0), (C, 1, 1, 1))
        conv_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype='float') 
        conv_y = np.tile(np.expand_dims(np.expand_dims(conv_y, 0), 0), (C, 1, 1, 1))
        conv1.weight = nn.Parameter(torch.from_numpy(conv_x).to(self.device).float())
        conv2.weight = nn.Parameter(torch.from_numpy(conv_y).to(self.device).float())
        return conv1, conv2


    def forward(self, img_i):
        # img_i = img_i.unsqueeze(0)
        img_tcr = self.bbox
        # img_quad = x[2]
        p_init = torch.zeros((BATCH_SIZE, 6)).to(self.device)

        # print(img_i.shape)
        # print(img_tcr.shape)
        # print(p_init.shape)
        # img_tcr, coords = self.crop_function(img_t, img_quad)
        B, C, h, w = img_tcr.shape
        sz = EXEMPLAR_SIZE
        sx = INSTANCE_SIZE
        centre = torch.Tensor([int(sx / 2.0), int(sx / 2.0)])
        
        xmin = centre[0] - int(sz / 2.0)
        xmax = centre[0] + int(sz / 2.0)
        
        coords = torch.tensor([xmin, xmin, xmax + 1, xmax + 1]).to(self.device)  #exclusive
        img_quad = torch.tensor([xmin, xmax, xmin, xmin, xmax, xmin, xmax, xmax]).to(self.device) #inclusive
        img_quad = img_quad.unsqueeze(0)
        quad = img_quad
        omega_t = self.form_omega_t(coords, B)
        sobel_tx, sobel_ty = self.sobel_gradients(img_tcr, self.conv1, self.conv2)
        # sx_crop, _ = self.crop_function(sobel_tx, img_quad)
        # sy_crop, _ = self.crop_function(sobel_ty, img_quad)
        
        J = self.J_matrix(omega_t, sobel_tx, sobel_ty, self.params.mode)
        J_pinv = self.J_pinv(J, self.params.mode)
        # print(J_pinv.shape)
        # analyse_output(J_pinv)
        itr = 1
        # quad = img_quad
        p = p_init
        # print("$$$$(((((()))))) ", p)
        W = self.warp_matrix(p_init, self.params.mode)
        N = omega_t.shape[1]
        # omega_t = omega_t + centre.repeat(B*N).view(B, N, 2)
        omega_t = torch.cat((omega_t, torch.ones((B, N, 1), device=self.device)), 2)  # (B x N x 3)
        # print(self.params['epsilon'])
        # print(self.params['max_iterations'])
        
        while(self.params.max_iterations > 0):

            omega_warp = omega_t.bmm(W)
            warped_i = self.sample_layer(img_i, omega_warp).permute(0, 2, 1) # (B x C x N)
            warped_i = warped_i.view(img_tcr.shape)

            # print(warped_i.shape)
            # print()
            # print()
            # print(img_tcr.shape)
            # ti1 = img_to_numpy(warped_i[0])
            # ti2 = img_to_numpy(img_tcr[0])
            # cv2.imwrite("t1.jpeg", ti1)
            # cv2.imwrite("t2.jpeg", ti2)
            
            # from IPython import embed;embed()
            r = (warped_i - img_tcr)
            # print(r)
            r = r.permute(0, 2, 3, 1)            
            r = r.contiguous().view(B, C * h * w, 1)
            # analyse_output(r)
            # print(r.norm())
            delta_p = (J_pinv.bmm(r)).squeeze(2)
            dp = self.warp_inv(delta_p)
            p_new = self.composition(p, dp)
            W = self.warp_matrix(p_new, self.params.mode)
            # print(itr, img_quad)
            # print(p, p_new)
            quad_new = self.quad_layer(img_quad, W, img_i.shape)
            # print(quad_new)
            # print("::::::::::::::::")
            # analyse_output(quad_new)

            if (itr >= self.params.max_iterations or \
            (quad_new - quad).norm() <= self.params.epsilon):
                break
            itr += 1
            p = p_new
            quad = quad_new
        print("--------------------------------------------------------------------------------")
        print(itr)

        # img_tcr, _ = self.crop_function(img_i, quad_new)
        # self.bbox = img_tcr
        # self.p = p_new
        # print(img_tcr.shape)
        return quad

# if __name__ == '__main__':
#     device = torch.device("cuda")
#     model = LKT