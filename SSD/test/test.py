import torch
from SSD.utiles.box import Box
import numpy as np


X = torch.rand(size=(1, 3, 10, 10))
# 这个随机数组干啥的范围是0-1
# print(X)
Y = Box.multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
print(Y.size())
x = torch.randn(1,1,2,3)
print(x)
print(x.permute(0,3,2,1))# 1 1 2 3
x=torch.randn(1,2)
y=torch.randn(1,2)
print(torch.cat((x,y),dim=1))
pred = [None]*5
pred[0] = torch.rand(1,8,32,32)
pred[1] = torch.rand(1,8,16,16)
pred[2] = torch.rand(1,2,3,2)
print(pred[2])
print(torch.flatten(pred[2].permute(0,2,3,1),start_dim= 1 ))
