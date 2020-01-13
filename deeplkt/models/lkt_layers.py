import torch
import torch.nn as nn
import numpy as np
import cv2
import time
from deeplkt.models.base_model import BaseModel

class LKTLayers(nn.Module):



    def __init__(self, device):
        super().__init__()
        self.device = device
        # super().__init__(device, checkpoint_dir, log_dir)
        self.pad = nn.ReflectionPad2d(1)


    def _forward_(self):
        pass

    def warp_tensor(self, x, y, W):
        """ Warps input tensors

        Keyword arguments:
        x, y -- int, int
        p -- B x 6
        Returns
        Transformed x, y -- [B , B]
        """

        B = W.shape[0]
        if(self.device.type == 'cuda'):
            C = torch.cuda.FloatTensor([x, y, 1.0]).unsqueeze(0).unsqueeze(0)
        else:
            C = torch.FloatTensor([x, y, 1.0]).unsqueeze(0).unsqueeze(0)

        C = C.repeat(B, 1, 1)
        Cn = C.bmm(W)
        return Cn[:, 0, 0], Cn[:, 0, 1] 


    def sample_layer(self, img, omega_warp):
        """ Finds image intensity values for coords. in omega_warp

        Keyword arguments:
            img -- B x C x H x W
            omega_warp -- B x N x 2
        Returns:
            Interpolated image pixel values -- B x N x C
        """
        samples_x = omega_warp[:, :, 0]
        samples_y = omega_warp[:, :, 1]

        H = img.shape[2]
        W = img.shape[3]
        samples_x = samples_x.unsqueeze(2)
        samples_x = samples_x.unsqueeze(3)
        samples_y = samples_y.unsqueeze(2)
        samples_y = samples_y.unsqueeze(3)
        samples = torch.cat([samples_x, samples_y], 3)  # shape = B x N x 1 x 2
        samples[:, :, :, 0] = (samples[:, :, :, 0]/(W-1))
        samples[:, :, :, 1] = (samples[:, :, :, 1]/(H-1))
        samples = samples*2-1                       # normalize to between -1 and 1
        return torch.nn.functional.grid_sample(img, samples) \
            .squeeze(3) \
            .permute(0, 2, 1)   # shape = B x N x C

    def form_omega_t(self, coords, b):
        """ Form a meshgrid of coordinates.

        Keyword arguments:
            coords -- x1, y1, x2, y2
            b -- batch size
        Returns:
            List of coordinates -- B x N x 2
        """
        x_range = torch.arange(coords[0], coords[2], device=self.device)
        y_range = torch.arange(coords[1], coords[3], device=self.device)
        
        Y, X = torch.meshgrid([y_range, x_range])
        h = Y.shape[0]
        w = Y.shape[1]

        omega_t = torch.stack([X, Y], dim=2)
        omega_t = omega_t.view(1, h * w, 2)
        omega_t = omega_t.repeat(b, 1, 1)
        return omega_t.float()

    def warp_inv(self, p):
        """ Finds warp_inv of warp parameters p

        Keyword arguments:
            p -- B x 6
        Returns:
            p_inv -- B x 6
        """

        b = p.shape[0]
        val = (torch.ones((b,), device=self.device) + p[:, 0]) * \
              (torch.ones((b,), device=self.device) +
               p[:, 3]) - (p[:, 1] * p[:, 2])

        inverse_output = torch.stack([(-p[:, 0] - p[:, 0] * p[:, 3] + p[:, 1] * p[:, 2]) / val,
                          (-p[:, 1]) / val,
                          (-p[:, 2]) / val,
                          (-p[:, 3] - p[:, 0] * p[:, 3] +
                           p[:, 1] * p[:, 2]) / val,
                          (-p[:, 4] - p[:, 3] * p[:, 4] +
                           p[:, 2] * p[:, 5]) / val,
                          (-p[:, 5] - p[:, 0] * p[:, 5] + p[:, 1] * p[:, 4]) / val])

        inverse_output = inverse_output.permute(1, 0)
        return inverse_output

    def crop_function(self, img_t, bb):
        """ Crops image to a given bounding box

        Keyword arguments:
            img_t -- B x C x H x W
            bb -- B x 8
        Returns:
            img_tcr = Cropped image -- B x C x h x w
            coords = [x1, y1, x2, y2]
        """

        quad = bb[0]

        b = img_t.shape[0]
        if(self.device.type == 'cuda'):

            rect = torch.cuda.FloatTensor([torch.min(quad[2], quad[0]),
                                           torch.min(quad[3], quad[5]),
                                           torch.max(quad[6], quad[4]),
                                           torch.max(quad[7], quad[1])])
            x1 = rect[0].type(torch.cuda.IntTensor)
            y1 = rect[1].type(torch.cuda.IntTensor)
            x2 = rect[2].type(torch.cuda.IntTensor)
            y2 = rect[3].type(torch.cuda.IntTensor)
        else:
            rect = torch.FloatTensor([torch.min(quad[2], quad[0]),
                                      torch.min(quad[3], quad[5]),
                                      torch.max(quad[6], quad[4]),
                                      torch.max(quad[7], quad[1])])

            x1 = rect[0].type(torch.IntTensor)
            y1 = rect[1].type(torch.IntTensor)
            x2 = rect[2].type(torch.IntTensor)
            y2 = rect[3].type(torch.IntTensor)

        img_tcr = img_t[:, :, y1:y2, x1:x2]
        coords = [x1, y1, x2, y2]

        return img_tcr, coords

    def sobel_gradients(self, img, conv1, conv2):
        """ Computes sobel gradients for an image

        Keyword arguments:
            img -- B x C x H x W
        Returns:
            sobel_x, sobel_y -- B x C x H x W
        """
        pad = nn.ReflectionPad2d(1)
        G_x = conv1(pad(img))
        G_y = conv2(pad(img))
        # C = G_x.shape[1]
        # exit()
        return [G_x, G_y]

    def J_matrix(self, omega_t, sobel_tx, sobel_ty, mode):
        """ Computes Jacobian matrix for image t

        Keyword arguments:
            omega_t -- B x N x 2
            sobel_tx -- B x C x H x W 
            sobel_ty -- B x C x H x W 
            mode -- int

        Returns:
            J -- B x (N * C) x 6
        """
        # return

        # start_t = time.process_time()
        B = omega_t.shape[0]
        N = omega_t.shape[1]
        C = sobel_tx.shape[1]
        H = sobel_tx.shape[2]
        W = sobel_tx.shape[3]
        print(H, W, N)
        assert(N == H * W)

        # print(time.process_time() - start_t)

        # sobel_tx_new = sobel_tx_new.
        sobel_tx = sobel_tx.contiguous().view(B, C, N)
        sobel_ty = sobel_ty.contiguous().view(B, C, N)
        
        # sobel_tx_new = self.sample_layer(sobel_tx, omega_t)  # (B, N, C)
        # sobel_ty_new = self.sample_layer(sobel_ty, omega_t)  # (B, N, C)
        
        # sobel_tx_new = sobel_tx[:, :, coords[1]:coords[3], coords[0]:coords[2] ]
        # sobel_tx_new = sobel_tx_new.contiguous().view(B, C, sobel_tx_new.shape[2] * sobel_tx_new.shape[3]).permute(0, 2, 1)
        # sobel_ty_new = sobel_ty[:, :, coords[1]:coords[3], coords[0]:coords[2] ]
        # sobel_ty_new = sobel_ty_new.contiguous().view(B, C, sobel_ty_new.shape[2] * sobel_ty_new.shape[3]).permute(0, 2, 1)

        # return
        sobel_tstk = torch.stack(
            [sobel_tx, sobel_ty], dim=3)  # (B, N, C, 2)
        # del sobel_tx_new, sobel_ty_new
        # return
        grad_zeros = torch.zeros((B, N), device=self.device)
        grad_ones = torch.ones((B, N), device=self.device)

        if (mode == 0):
            grad_wx = torch.stack([omega_t[:, :, 0],
                                   torch.zeros((B, N), device=self.device),
                                   omega_t[:, :, 1],
                                   torch.zeros((B, N), device=self.device),
                                   torch.ones((B, N), device=self.device),
                                   torch.zeros((B, N), device=self.device)], dim=2)

            grad_wy = torch.stack([torch.zeros((B, N), device=self.device),
                                   omega_t[:, :, 0],
                                   torch.zeros((B, N), device=self.device),
                                   omega_t[:, :, 1],
                                   torch.zeros((B, N), device=self.device),
                                   torch.ones((B, N), device=self.device)], dim=2)

        elif (mode == 4):
            grad_wx = torch.stack([omega_t[:, :, 0],
                                   torch.zeros((B, N), device=self.device),
                                   torch.zeros((B, N), device=self.device),
                                   torch.zeros((B, N), device=self.device),
                                   grad_ones,
                                   torch.zeros((B, N), device=self.device)], dim=2)

            grad_wy = torch.stack([torch.zeros((B, N), device=self.device),
                                   torch.zeros((B, N), device=self.device),
                                   torch.zeros((B, N), device=self.device),
                                   omega_t[:, :, 1],
                                   torch.zeros((B, N), device=self.device),
                                   torch.ones((B, N), device=self.device)], dim=2)

        elif (mode == 5):
            grad_wx = torch.stack([omega_t[:, :, 0],
                                   -omega_t[:, :, 1],
                                   torch.zeros((B, N), device=self.device),
                                   torch.zeros((B, N), device=self.device),
                                   torch.ones((B, N), device=self.device),
                                   torch.zeros((B, N), device=self.device)], dim=2)

            grad_wy = torch.stack([omega_t[:, :, 1],
                                   omega_t[:, :, 0],
                                   torch.zeros((B, N), device=self.device),
                                   torch.zeros((B, N), device=self.device),
                                   torch.zeros((B, N), device=self.device),
                                   torch.ones((B, N), device=self.device)], dim=2)



        else:
            raise NotImplementedError

        grad_w = torch.stack([grad_wx, grad_wy], dim=2)  # (B, N, 2, 6)
        # analyse_output(grad_w)
        # del grad_wx, grad_wy
        sobel_tstk = sobel_tstk.view(B * N, C, 2)
        grad_w = grad_w.view(B * N, 2, 6)

        J = sobel_tstk.bmm(grad_w)
        J = J.view(B, N * C, 6)
        del(grad_w)
        if(mode == 4):
            J = torch.stack(
                [J[:, :, 0], J[:, :, 3], J[:, :, 4], J[:, :, 5]], dim=2)
        elif(mode == 5):
            J = torch.stack(
                [J[:, :, 0], J[:, :, 1], J[:, :, 4], J[:, :, 5]], dim=2)

        elif(mode != 0):
            raise NotImplementedError
        # print(J.shape)
        return J

    def J_pinv(self, J, mode):
        """ Computes inverse of Jacobian matrix

        Keyword arguments:
            J -- B x (N * C) x 6

        Returns:
            J_inv -- B x 6 x (N * C)
        """

        B = J.shape[0]
        # start_t = time.process_time()
        # print(type(J))
        Jt = J.permute(0, 2, 1)

        Jtj = Jt.bmm(J)

        # print(Jtj)
        # print(Jtj[0].pinverse().shape)
        # print(Jtj)
        # return
        Jtji = torch.stack([m.inverse() for m in Jtj])
#
        # del Jtj
        # print(Jtji.shape)
        # print(Jt.shape)
        # start_t = time.process_time()
        J_pinv = Jtji.bmm(Jt)

        # print(time.process_time() - start_t)
        # return
        # start_t = time.process_time()

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
        # print(time.process_time() - start_t)
        return J_pinv
        # (K.batch_dot(Jt, J)), Jt)

    def warp_matrix(self, p_init, mode):
        """ Warps all omega_t coordinates acc. to p

        Keyword arguments:
            p_init -- B x 6
            mode -- int

        Returns:
            warp_matrix -- B x 3 x 2
        """
        B = p_init.shape[0]
        # print("p_init_shape = ", p_init.shape)
        # if(mode == 0):
        W = torch.stack([torch.stack([torch.ones(B, device=self.device) + p_init[:, 0],
                p_init[:, 2],
                p_init[:, 4]]),
                torch.stack([p_init[:, 1],
                torch.ones(B, device=self.device) + p_init[:, 3],
                p_init[:, 5]])])

        # elif(mode == 4):
        #     # print(p_init)
        #     W = torch.stack([torch.stack([torch.ones(B, device=self.device) + p_init[:, 0],
        #           torch.zeros(B, device=self.device),
        #           p_init[:, 4]]),
        #           torch.stack([torch.zeros(B, device=self.device),
        #           torch.ones(B, device=self.device) + p_init[:, 3],
        #           p_init[:, 5]])])

        # elif(mode == 5):
        #     # print(p_init)
        #     W = torch.stack([torch.stack([torch.ones(B, device=self.device) + p_init[:, 0],
        #           -p_init[:, 1],
        #           p_init[:, 4]]),
        #           torch.stack([p_init[:, 1],
        #           torch.ones(B, device=self.device) + p_init[:, 0],
        #           p_init[:, 5]])])


        # else:
        #     raise NotImplementedError
        # W = [torch.stack(e) for e in W]
        # W = torch.stack(W)  # 2 x 3 x B
        W = W.permute(2, 1, 0)  # B x 3 x 2
        return W

    def compute_omega_warp(self, omega_t, p_init, mode):
        B = omega_t.shape[0]
        N = omega_t.shape[1]
        # p_init = self.center_transform(p_init)
        omega_t = torch.cat([omega_t, torch.ones(
            (B, N, 1), device=self.device)], 2)  # (B x N x 3)

        W = self.warp_matrix(p_init, mode)
        return omega_t.bmm(W)  # B x N x 2

    def composition(self, p, dp):
        """ Compositon of p and dp

        Keyword arguments:
            p -- B x 6
            dp -- B x 6

        Returns:
            p_new -- B x 6
        """
        # print(p)
        # print(dp)
        pn = torch.stack([p[:, 0] + dp[:, 0] + p[:, 0] * dp[:, 0] + p[:, 2] * dp[:, 1],
              p[:, 1] + dp[:, 1] + p[:, 1] * dp[:, 0] + p[:, 3] * dp[:, 1],
              p[:, 2] + dp[:, 2] + p[:, 0] * dp[:, 2] + p[:, 2] * dp[:, 3],
              p[:, 3] + dp[:, 3] + p[:, 1] * dp[:, 2] + p[:, 3] * dp[:, 3],
              p[:, 4] + dp[:, 4] + p[:, 0] * dp[:, 4] + p[:, 2] * dp[:, 5],
              p[:, 5] + dp[:, 5] + p[:, 1] * dp[:, 4] + p[:, 3] * dp[:, 5]])
        return pn.permute(1, 0)

    def center_transform(self, quad, p):
        """ Translating to image center, applying warp
            and translating back.

        Keyword arguments:
            quad -- B * 8
            p -- B x 6
        Returns:
            p1 -- B x 6
        """

        B = p.shape[0]
        xc = quad[0].clone()
        yc = quad[1].clone()
        for j in range(1, 4):
            xc += quad[2 * j]
            yc += quad[2 * j + 1]
        xc /= 4.0
        yc /= 4.0
        T = torch.eye(3, device=self.device).unsqueeze(0).repeat(B, 1, 1)
        Tb = torch.eye(3, device=self.device).unsqueeze(0).repeat(B, 1, 1)

        T[:, 0, 2] = -xc
        T[:, 1, 2] = -yc
        Tb[:, 0, 2] = xc
        Tb[:, 1, 2] = yc

        p = p.view(B, 3, 2).permute(0, 2, 1)
        p = torch.cat([p, torch.zeros((B, 1, 3), device=self.device)], dim=1)
        p[:, 0, 0] += 1.0
        p[:, 1, 1] += 1.0
        p[:, 2, 2] = 1.0
        # print(T)
        # print(p.bmm(T))
        p1 = Tb.bmm(p.bmm(T))
        p1[:, 0, 0] -= 1.0
        p1[:, 1, 1] -= 1.0
        p1 = p1[:, 0:2, :].permute(0, 2, 1)
        p1 = p1.contiguous().view(B, 6)
        return p1

    def quad_layer(self, bb, W, img_i_shape):
        """ Getting new quadrlateral using new warp coordinates

        Keyword arguments:
            quad -- B * 8
            p_new -- B x 6
            img_i_shape -- (B x C x H x W
        Returns:
            quad_new -- B x 8
        """

        quad = bb[0]
        B = img_i_shape[0]

        xbl_p, ybl_p = self.warp_tensor(quad[0], quad[1], W)
        # print(xbl_p, ybl_p)
        x1_p, y1_p = self.warp_tensor(quad[2], quad[3], W)
        xtr_p, ytr_p = self.warp_tensor(quad[4], quad[5], W)
        x2_p, y2_p = self.warp_tensor(quad[6], quad[7], W)

        # print(img_i_shape)
        H = img_i_shape[2]
        W = img_i_shape[3]

        x1_p = torch.max(torch.zeros(B, device=self.device), x1_p)
        y1_p = torch.max(torch.zeros(B, device=self.device), y1_p)
        x2_p = torch.min(torch.zeros(B, device=self.device) + W, x2_p)
        y2_p = torch.min(torch.zeros(B, device=self.device) + H, y2_p)

        xbl_p = torch.max(torch.zeros(B, device=self.device), xbl_p)
        ybl_p = torch.min(torch.zeros(B, device=self.device) + H, ybl_p)
        xtr_p = torch.min(torch.zeros(B, device=self.device) + W, xtr_p)
        ytr_p = torch.max(torch.zeros(B, device=self.device), ytr_p)

        wc = torch.stack([xbl_p, ybl_p, x1_p, y1_p, xtr_p, ytr_p, x2_p, y2_p])
        return wc.view(B, 8)
