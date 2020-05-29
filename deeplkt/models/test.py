import torch
import torch.nn as nn
import torchvision.models as models

model = models.alexnet(pretrained=True)
new_features = torch.nn.Sequential(*(list(self.vgg.classifier.children())[:-2]))
eles = list(model.features.children())
for e in eles:
    print(e)
# l = [module for module in model.features.modules() if type(module) != nn.Sequential]
# for i, ele in enumerate(l):
#     print(i, ele)

# for i,f in enumerate(model.features.parameters()):
#     print(i, f.shape)