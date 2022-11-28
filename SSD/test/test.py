import torch
from SSD.utiles.box import Box
import numpy as np
import cv2
import matplotlib.pyplot as plt
"""
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
print(torch.flatten(pred[2].permute(0,2,3,1),start_dim= 1 ))"""

img = cv2.imread("../banana.png")
# img = transforms.ToPILImage(img)#用于transforms方法使用
plt.figure()
fig = plt.imshow(img)
h,w = img.shape[:2]
print(h,w)
###实现功能：给出框坐标给框出来
# bbox是边界框的英文缩写

cat_box = [0.1556*256, 0.5877*256, 0.3749*256, 0.8087*256]
boxes = torch.tensor([cat_box])
# print(boxes.size())
# print(Box.box_center_to_corner(Box.box_corner_to_center(boxes)) == boxes)
rect = Box.rect_(cat_box, 'purple', 3)
fig.axes.text(750 + 10, 100 + 50, "my_cat 0.9", bbox={'facecolor': 'blue', 'alpha': 0.5})
# 添加标签
fig.axes.add_patch(rect)
# 添加框框
plt.show()


"""
img_test = cv2.imread("../banana.png")
plt.imshow(img_test)
plt.show()
0.1556, 0.5877, 0.3749, 0.8087
"""
"""
print(img_test.shape)
img_test = torch.from_numpy(img_test).permute(2, 0, 1)
plt.imshow(img_test[1,:,:])
plt.show()
X = img_test.unsqueeze(0).float()
print(X.size())
img = X.squeeze(0).permute(1, 2, 0).long()
print(img.size())"""


















