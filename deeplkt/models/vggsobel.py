
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
from deeplkt.config import *


class VGGSobel(nn.Module):

    def __init__(self, num_channels=3, num_classes=1000):
        super(VGGSobel, self).__init__()
        self.pad = nn.ReflectionPad2d(1)

        self.convx = nn.Conv2d(num_channels, num_channels*num_classes,\
             kernel_size=3, stride=1, bias=False, groups=num_channels)
        self.convy = nn.Conv2d(num_channels, num_channels*num_classes,\
             kernel_size=3, stride=1, bias=False, groups=num_channels)
        # print("Wx shape = ", self.convx.weight.shape)
        self.vgg = models.vgg16(pretrained=True)
        self.soft = nn.Softmax(dim=1)
        conv_1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float')
        conv_1 = np.tile(np.expand_dims(np.expand_dims(conv_1, 0), 0), \
            (num_channels * num_classes, 1, 1, 1))
        conv_2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype='float') 
        conv_2 = np.tile(np.expand_dims(np.expand_dims(conv_2, 0), 0), \
            (num_channels * num_classes, 1, 1, 1))
        # print("conv1 shape = ", conv_1.shape)

        self.convx.weight = nn.Parameter(torch.from_numpy(conv_1).float())
        self.convy.weight = nn.Parameter(torch.from_numpy(conv_2).float())

        for i, param in enumerate(self.vgg.parameters()):
            param.requires_grad = False
        self.transform = transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225] )
        


    def forward(self, x):

        # print("X shape = ", x.shape)
        outx = self.convx(self.pad(x))
        outy = self.convy(self.pad(x))
        print(outx.shape)
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
        p = p.unsqueeze(2).unsqueeze(3)
        sobelx = []
        sobely = []
        
        for i in range(3):
            sx = torch.sum(outx[:, i*1000:(i + 1)*1000, :, :] * p, dim=1)
            sy = torch.sum(outy[:, i*1000:(i + 1)*1000, :, :] * p, dim=1)
            sobelx.append(sx)
            sobely.append(sy)            
        sobelx = torch.stack(sobelx, dim=1)
        sobely = torch.stack(sobely, dim=1)
        # print("Sobel_x shape = ", img.shape)
        # print("Sobel_y shape = ", img.shape)

        return sobelx, sobely


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
    device = torch.device('cpu')
    model = VGGSobel().to(device)
    x = torch.ones(5, 3, 127, 127).to(device)
    sx, sy = model(x)
    print(sx.shape, sy.shape)
    