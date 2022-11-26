import torch
from SSD.utiles.box import Box
import numpy as np


X = torch.rand(size=(1, 3, 10, 10))
# 这个随机数组干啥的范围是0-1
# print(X)
Y = Box.multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
print(Y)

