
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
from deeplkt.config import *
from time import time
from collections import OrderedDict


# class Attention(nn.Module):

#     def __init__(self, device, num_channels=3):
#         num_classes = EXEMPLAR_SIZE * EXEMPLAR_SIZE
#         super(Attention, self).__init__()
#         self.pad = nn.ReflectionPad2d(1)
#         self.num_channels = num_channels
#         self.num_classes = num_classes
#         self.device = device
#         self.vgg = models.vgg16(pretrained=True)
#         l = [module for module in self.vgg.features.modules() if type(module) != nn.Sequential]
#         for i, ele in enumerate(l):
#             print(i, ele)        
#         print("Parameters = ")
#         for i, param in enumerate(self.vgg.features.parameters()):
#             print(i, param.shape)
#             param.requires_grad = False

#         new_classifier = torch.nn.Sequential(*(list(self.vgg.classifier.children())[:-1]))
#         fc = torch.nn.Linear(4096, num_classes)
#         # fc.weight = nn.Parameter(torch.ones((num_classes, 4096), requires_grad=True).to(self.device).float())
#         # fc.bias = nn.Parameter(torch.zeros((num_classes), requires_grad=True).to(self.device).float())

#         new_classifier.add_module('out_layer', fc)
#         self.vgg.classifier = new_classifier
#         self.soft = nn.Softmax(dim=1)        
#         self.transform = transforms.Normalize(
#                                 mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225] )
    
        


#     def forward(self, x):

#         x = F.interpolate(x, size=(VGG_SIZE, VGG_SIZE), mode='bilinear') / 255.0
#         B = x.shape[0]
#         img = []
#         for i in range(B):
#             img.append(self.transform(x[i, :, :, :]))
#         img = torch.stack(img, dim=0)

#         p = self.vgg(img)
#         p = self.soft(p)
#         # print(p.shape)
#         return p


# if __name__ == '__main__':
#     device = torch.device('cuda')
#     model = Attention(device).to(device)
#     start_t = time()
#     x = torch.ones(5, 3, EXEMPLAR_SIZE, EXEMPLAR_SIZE).to(device)
#     p = model(x)
#     end_t = time()
#     print(end_t - start_t)
#     print(p.shape)
    

class Attention(nn.Module):

    def __init__(self, device, model_path='pretrained/alexnet/model.pth'):
        # configs = list(map(lambda x: 3 if x == 3 else
        #                int(x*width_mult), AlexNetLegacy.configs))

        configs = [3, 96, 256, 384, 384, 256]
        self.model_path = model_path
        self.device = device
        super(Attention, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(configs[0], configs[1], kernel_size=11, stride=2),
            nn.BatchNorm2d(configs[1]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[1], configs[2], kernel_size=5),
            nn.BatchNorm2d(configs[2]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[2], configs[3], kernel_size=3),
            nn.BatchNorm2d(configs[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[3], configs[4], kernel_size=3),
            nn.BatchNorm2d(configs[4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[4], configs[5], kernel_size=3),
            nn.BatchNorm2d(configs[5]),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True)
        )
        num_classes = EXEMPLAR_SIZE*EXEMPLAR_SIZE
        self.num_classes = num_classes
        self.out_layer = nn.Linear(2048, num_classes)
        self.out_layer.weight = nn.Parameter(torch.zeros((num_classes, 2048), requires_grad=True))
        self.out_layer.bias = nn.Parameter(torch.zeros((num_classes), requires_grad=True))
        # self.out_layer = fc
        self.softmax = nn.Softmax(dim=1)
        self.feature_size = configs[5]

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        x = self.out_layer(x) + 1.0
        x = self.softmax(x)
        x = x * self.num_classes
        x = x.view(x.size(0), EXEMPLAR_SIZE, EXEMPLAR_SIZE)
        return x

    def load_pretrained(self):
        pretrained = torch.load(self.model_path)
        self.features.load_state_dict(torch.load(self.model_path))


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = Attention(device).to(device)
    model.load_pretrained()
    l = [module for module in model.features.modules() if type(module) != nn.Sequential]
    # for i, ele in enumerate(l):
    #     print(i, ele)
    a = torch.ones(4, 3)
    print(model.softmax(a) * 3)
    a = torch.ones((5, 3, 127, 127)).to(device)
    # print(a)
    model(a)
    # print(i, ele)