import torch
import torch.nn as nn
from model_pytorch import PureLKTNet
import numpy as np

class LearnableLKTNet(PureLKTNet):

    def conv_network(self):
        return nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(3, 4, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(4),
        
        nn.ReflectionPad2d(1),
        nn.Conv2d(4, 8, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(8),

        nn.ReflectionPad2d(1),
        nn.Conv2d(8, 16, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(16), 

        nn.ReflectionPad2d(1),
        nn.Conv2d(16, 3, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(3) )


    def __init__(self, device, params):
        super(LearnableLKTNet, self).__init__(device, params)
        self.conv1, self.conv2 = self.sobel_kernels(16)
        self.cnn1 = self.conv_network()

    def forward(self, x):
        img_t = x[0]
        img_i = x[1]
        img_t = self.cnn1(img_t)
        img_i = self.cnn1(img_i)
        img_quad = x[2]
        p_init = x[3]

        img_tcr, omega_t = self.crop_function(img_t, img_quad)
        B, C, h, w = img_tcr.shape
        sobel_tx, sobel_ty = self.sobel_gradients(img_t, self.conv1, self.conv2)
        J = self.J_matrix(omega_t, sobel_tx, sobel_ty, self.params['mode'])
        J_pinv = self.J_pinv(J, self.params['mode'])
        itr = 1
        quad = img_quad
        p = p_init

        W = self.warp_matrix(p_init, self.params['mode'])
        N = omega_t.shape[1]
        omega_t = torch.cat((omega_t, torch.ones((B, N, 1), device=self.device)), 2)  # (B x N x 3)

        while(True):

            omega_warp = omega_t.bmm(W)
            warped_i = self.sample_layer(img_i, omega_warp).permute(0, 2, 1) # (B x C x N)
            warped_i = warped_i.view(img_tcr.shape)
            r = (warped_i - img_tcr)
            r = r.permute(0, 2, 3, 1)            
            r = r.contiguous().view(B, C * h * w, 1)
            delta_p = (J_pinv.bmm(r)).squeeze(2)
            dp = self.warp_inv(delta_p)
            p_new = self.composition(p, dp)
            W = self.warp_matrix(p_new, self.params['mode'])
            quad_new = self.quad_layer(img_quad, W, img_i.shape)

            if (itr >= self.params['max_iterations'] or \
            (quad_new - quad).norm() <= self.params['epsilon']):
                break
            itr += 1
            p = p_new
            quad = quad_new
        return quad_new, p_new, sobel_tx, sobel_ty


