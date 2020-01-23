import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F




class AlexSobel(nn.Module):

    def __init__(self, num_channels=3):
        super(AlexSobel, self).__init__()
        self.num_channels = num_channels
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.sobelx = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 9 * num_channels),
        )
        self.sobely = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 9 * num_channels),
        )
        # for i, param in enumerate(self.features.parameters()):
        #     print(i, param.requires_grad)

        # def count_parameters(mod):
        #     return sum(p.numel() for p in mod.parameters() if p.requires_grad)
        # print(count_parameters(self.sobelx) * 2)
        # print(count_parameters(self.features))

        st_dict = models.alexnet(pretrained=True).features.state_dict()
        self.features.load_state_dict(st_dict)
        for i, param in enumerate(self.features.parameters()):
            # print(i, param.requires_grad)
            if(i < 8):
                param.requires_grad = False


        # print(count_parameters(self.features))



    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # print(x.shape)
        sx = self.sobelx(x)
        sx = sx.view(sx.shape[0], self.num_channels, 1, 3, 3)
        sy = self.sobely(x)
        sy = sy.view(sy.shape[0], self.num_channels, 1, 3, 3)
        return sx, sy


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
    model = AlexSobel()
    x = torch.ones(5, 3, 127, 127)
    y = torch.ones(5, 3, 127, 127)
    out_x = []
    sx, sy = model(x)
    for i in range(5):
        out_x.append(F.conv2d(y[i:i+1, :, :, :], sx[i, :, :, :, :], \
            stride=1, padding=1, groups=model.num_channels))
    out_x = torch.cat(out_x)
    print(out_x.shape)
    