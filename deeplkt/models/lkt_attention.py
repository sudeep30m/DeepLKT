import torch
import torch.nn as nn
from deeplkt.models.lkt_layers import LKTLayers
from deeplkt.models.attention import Attention
from deeplkt.models.segmentation import AttentionModule

from deeplkt.models.base_model import BaseModel
from deeplkt.utils.model_utils import img_to_numpy
import numpy as np
from deeplkt.configParams import *
import cv2
import nvgpu
import torch.nn.functional as F


class LKTAttention(LKTLayers):


    def __init__(self, device, params):
        super().__init__(device)
        self.params = params
        self.attention = AttentionModule(device).to(self.device)
        self.conv1, self.conv2 = self.sobel_kernels(3)

    def template(self, bbox):
        self.bbox = bbox

    def imgList(self, img_list):
        # print([x.shape for x in img_list])
        self.img_list = img_list
        

    def sobel_kernels(self, C):
        conv1 = nn.Conv2d(C, C, kernel_size=3, stride=1, bias=False, groups=C)
        conv2 = nn.Conv2d(C, C, kernel_size=3, stride=1, bias=False, groups=C)
        conv_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float')
        conv_x = np.tile(np.expand_dims(np.expand_dims(conv_x, 0), 0), (C, 1, 1, 1))
        conv_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype='float') 
        conv_y = np.tile(np.expand_dims(np.expand_dims(conv_y, 0), 0), (C, 1, 1, 1))
        conv1.weight = nn.Parameter(torch.from_numpy(conv_x).to(
                                    self.device).float(), requires_grad=False)
        conv2.weight = nn.Parameter(torch.from_numpy(conv_y).to(
                                    self.device).float(), requires_grad=False)
        return conv1, conv2


    # def sobel_layer(self, x):
    #     sx, sy, p = self.vgg(x)
    #     pad = nn.ZeroPad2d(1)

    #     out_x = []
    #     out_y = []

    #     for i in range(x.shape[0]):
    #         out_x.append(F.conv2d(pad(x[i:i+1, :, :, :]), sx[i, :, :, :, :], \
    #             stride=1,  groups=self.vgg.num_channels))
    #         out_y.append(F.conv2d(pad(x[i:i+1, :, :, :]), sy[i, :, :, :, :], \
    #             stride=1,  groups=self.vgg.num_channels))
    #     out_x = torch.cat(out_x)
    #     out_y = torch.cat(out_y)
    #     return out_x, out_y, sx, sy, p


    def forward(self, img_i):
        img_tcr = self.bbox
        B, C, h, w = img_tcr.shape

        p_init = torch.zeros((B, 6), device=self.device)
        sz = EXEMPLAR_SIZE
        sx = INSTANCE_SIZE
        centre = torch.Tensor([(sx / 2.0), (sx / 2.0)], device=self.device)
                
        xmin = centre[0] - (sz / 2.0)
        xmax = centre[0] + (sz / 2.0)
        
        coords = torch.tensor([xmin, xmin, xmax, xmax], device=self.device)  #exclusive

        img_quad = torch.tensor([xmin, xmax, xmin, xmin, xmax, xmin, xmax, xmax], device=self.device) #inclusive
        img_quad = img_quad.unsqueeze(0)
        img_quad = img_quad.repeat(B, 1)

        quads = []
        quad = img_quad
        # quads.append(quad)
        omega_t = self.form_omega_t(coords, B)
        # print(omega_t.shape)
        N = omega_t.shape[1]
        # sobel_tx, sobel_ty, sx, sy, probs = self.sobel_layer(img_tcr)
        sobel_tx, sobel_ty = self.sobel_gradients(img_tcr, self.conv1, self.conv2)

        # print(sobel_tx.shape)
        J = self.J_matrix(omega_t, sobel_tx, sobel_ty, self.params.mode)
        # print(J.shape)
        
        J = J.view(B, N, C, J.shape[2])
        # print("Before attention = ", nvgpu.gpu_info()[0]['mem_used'])

        P = self.attention(self.img_list, (EXEMPLAR_SIZE, EXEMPLAR_SIZE))
        # print(P.shape)
        # try:
        J_pinv = self.J_pinv(J, P, self.params.mode)
        # except:
        #     from IPython import embed;embed()

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
            if (itr >= self.params.max_iterations):
                quad = quad_new
                break
            itr += 1
            p = p_new
            quad = quad_new
            # quads.append(quad)

        # print("iterations = ", itr)
        return quad_new, sobel_tx, sobel_ty, img_tcr

    def J_pinv(self, J, P, mode):
        """ Computes inverse of Jacobian matrix

        Keyword arguments:
            J -- B x N x C x 6
            P -- B x N

        Returns:
            J_inv -- B x 6 x (N * C)
        """

        B, N, C, Wp = J.shape
        P = P.view(B, N, 1, 1)
        Jw = J * P
        J = J.view(B, N*C, Wp)
        Jw = Jw.view(B, N*C, Wp)

        Jt = J.permute(0, 2, 1)
        Jwt = Jw.permute(0, 2, 1)

        Jtj = Jt.bmm(Jw)
        # diagonal = torch.diagonal(Jtj, dim1=-2, dim2=-1)
        # diag = torch.zeros(Jtj.shape, device=self.device)
        # for i in  range(B):
        #     diag[i] = torch.diag(diagonal[i])
        # Jtj += LAMBDA * diag
        # try:
        Jtji = torch.stack([self.matrixInverse(m) for m in Jtj])
        # except:
        #     from IPython import embed;embed()
        
        J_pinv = Jtji.bmm(Jwt)
        M = J_pinv.shape[2]
        if(mode == 4):

            J_pinv = torch.stack([J_pinv[:, 0, :],
                                  torch.zeros((B, M), device=self.device),
                                  torch.zeros((B, M), device=self.device),
                                  J_pinv[:, 1, :],
                                  J_pinv[:, 2, :],
                                  J_pinv[:, 3, :]], dim=1)
        elif(mode == 5):
            J_pinv = torch.stack([J_pinv[:, 0, :],
                                  J_pinv[:, 1, :],
                                  torch.zeros((B, M), device=self.device),
                                  torch.zeros((B, M), device=self.device),
                                  J_pinv[:, 2, :],
                                  J_pinv[:, 3, :]], dim=1)

        elif(mode != 0):
            raise NotImplementedError
        return J_pinv




