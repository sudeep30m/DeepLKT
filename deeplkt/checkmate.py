import torch
import torch.nn as nn
import numpy as np 
from scipy.special import huber


a = np.array([[0.7, 0.6]])
b = np.array([[1.0, 0.0]])
print(huber(1, a - b).mean())


a = torch.from_numpy(a)
b = torch.from_numpy(b)

loss = nn.SmoothL1Loss()
print(loss(a, b))

