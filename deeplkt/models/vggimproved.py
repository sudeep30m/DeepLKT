
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
from deeplkt.config import *
from time import time


class VGGImproved(nn.Module):

    def __init__(self, device, num_channels=3, num_classes=1000):
        super(VGGImproved, self).__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.device = device
        # self.convx = nn.Conv2d(num_channels, num_channels*num_classes,\
        #      kernel_size=3, stride=1, bias=False, groups=num_channels)
        # self.convy = nn.Conv2d(num_channels, num_channels*num_classes,\
        #      kernel_size=3, stride=1, bias=False, groups=num_channels)
        # print("Wx shape = ", self.convx.weight.shape)
        self.vgg = models.vgg16(pretrained=True)
        self.soft = nn.Softmax(dim=1)
        conv_1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float')
        self.convx = np.tile(np.expand_dims(np.expand_dims(conv_1, 0), 0), \
            (num_channels * num_classes, 1, 1, 1))
        conv_2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype='float') 
        self.convy = np.tile(np.expand_dims(np.expand_dims(conv_2, 0), 0), \
            (num_channels * num_classes, 1, 1, 1))

        self.convx = np.expand_dims(self.convx, 0) #(1, CH*CL, 1, 3, 3)
        self.convy = np.expand_dims(self.convy, 0) #(1, CH*CL, 1, 3, 3)

        self.convx = torch.tensor(self.convx, device=self.device)
        self.convy = torch.tensor(self.convy, device=self.device)
        self.convx = nn.Parameter(self.convx, requires_grad=True)
        self.convy = nn.Parameter(self.convy, requires_grad=True)
        
        self.register_parameter(name='sobelx', param=self.convx)
        self.register_parameter(name='sobely', param=self.convy)

        # print("conv1 shape = ", conv_1.shape)
        # conv_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float')
        # conv_x = np.tile(np.expand_dims(np.expand_dims(conv_x, 0), 0),\
        #                                         (nu, 1, 1, 1))
        # conv_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype='float') 
        # conv_y = np.tile(np.expand_dims(np.expand_dims(conv_y, 0), 0), (C, 1, 1, 1))

        # self.convx.weight = nn.Parameter(torch.from_numpy(conv_1).float())
        # self.convy.weight = nn.Parameter(torch.from_numpy(conv_2).float())

        for i, param in enumerate(self.vgg.parameters()):
            param.requires_grad = False
        self.transform = transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225] )
        


    def forward(self, x):

        # print("X shape = ", x.shape)
        # outx = self.convx(self.pad(x))
        # outy = self.convy(self.pad(x))
        # print(outx.shape)
        # print(outy.shape)
        
        x = F.interpolate(x, size=(VGG_SIZE, VGG_SIZE), mode='bilinear') / 255.0
        # print(x)
        B = x.shape[0]
        img = []
        for i in range(B):
            img.append(self.transform(x[i, :, :, :]))
        img = torch.stack(img, dim=0)
        # print("Image shape = ", img.shape)
        # print(img)
        p = self.vgg(img)
        p = self.soft(p)
        p = p.unsqueeze(2).unsqueeze(3).unsqueeze(4).double() # (B, num_classes, 1, 1, 1)
        sobelx = []
        sobely = []
        B = p.shape[0]
        for i in range(self.num_channels):
            #(1, CH*CL, 1, 3, 3)
            sx = torch.sum(self.convx[:, 
                    i * self.num_classes : (i + 1) * self.num_classes, :, :, :] \
                    * p, dim=1)
            sy = torch.sum(self.convy[:, 
                i * self.num_classes : (i + 1) * self.num_classes, :, :] \
                    * p, dim=1)
            sobelx.append(sx)
            sobely.append(sy)            
        sobelx = torch.stack(sobelx, dim=1).float()
        sobely = torch.stack(sobely, dim=1).float()

        return sobelx, sobely, p


# def alexsobels():
# def alexnet(pretrained=False, progress=True, **kwargs):
#     r"""AlexNet model architecture from the
#     `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     model = AlexNet(**kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls['alexnet'],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model

if __name__ == '__main__':
    device = torch.device('cuda')
    model = VGGImproved(device).to(device)
    start_t = time()
    x = torch.ones(5, 3, 127, 127).to(device)
    sx, sy = model(x)
    end_t = time()
    print(end_t - start_t)
    print(sx, sy)
    