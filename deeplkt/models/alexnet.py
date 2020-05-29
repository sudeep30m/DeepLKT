import torch.nn as nn
import torch
class AlexNet(nn.Module):

    def __init__(self, model_path='pretrained/alexnet/model.pth'):
        # configs = list(map(lambda x: 3 if x == 3 else
        #                int(x*width_mult), AlexNetLegacy.configs))
        configs = [3, 96, 256, 384, 384, 256]
        self.model_path = model_path
        super(AlexNet, self).__init__()
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
        self.feature_size = configs[5]

    def forward(self, x):
        x = self.features(x)
        return x

    def load_pretrained(self):
        self.load_state_dict(torch.load(self.model_path))


if __name__ == '__main__':
    model = AlexNet(model_path='pretrained/alexnet/model.pth')
    model.load_state_dict(torch.load(model.model_path))
    l = [module for module in model.features.modules() if type(module) != nn.Sequential]
    for i, ele in enumerate(l):
        print(i, ele)
