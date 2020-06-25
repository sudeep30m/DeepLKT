
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
from deeplkt.configParams import *
from time import time


class VGGImproved(nn.Module):

    def __init__(self, device, num_channels=3, num_classes=200):
        super(VGGImproved, self).__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.device = device
        self.vgg = models.vgg16(pretrained=True)
        for i, param in enumerate(self.vgg.parameters()):
            param.requires_grad = False

        new_classifier = torch.nn.Sequential(*(list(self.vgg.classifier.children())[:-1]))
        new_classifier.add_module('out_layer', torch.nn.Linear(4096, num_classes))
        self.vgg.classifier = new_classifier
        # print(self.vgg)

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
        
        self.transform = transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225] )
        


    def forward(self, x):

        x = F.interpolate(x, size=(VGG_SIZE, VGG_SIZE), mode='bilinear') / 255.0
        B = x.shape[0]
        img = []
        for i in range(B):
            img.append(self.transform(x[i, :, :, :]))
        img = torch.stack(img, dim=0)

        p = self.vgg(img)
        p = self.soft(p)
        p = p.unsqueeze(2).unsqueeze(3).unsqueeze(4).double() # (B, num_classes, 1, 1, 1)
        sobelx = []
        sobely = []
        B = p.shape[0]
        for i in range(self.num_channels):
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


if __name__ == '__main__':
    device = torch.device('cuda')
    model = VGGImproved(device).to(device)
    start_t = time()
    x = torch.ones(5, 3, 224, 224).to(device)
    sx, sy, p = model(x)
    end_t = time()
    print(end_t - start_t)
    # print(sx, sy)
    