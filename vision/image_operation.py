import torch
import torch.nn as nn
import  torchvision
import cv2
import torchvision.transforms as transforms
# 实际上这玩意有20多种操作，要用的时候再看吧，增强数据集就当
from d2l import torch as d2l
import matplotlib.pyplot as plt
class Operation:
    def __init__(self):
        pass


def split_3(img):
    b = cv2.split(img)[0]  # B通道
    g = cv2.split(img)[1]  # G通道
    r = cv2.split(img)[2]  # R通道
    # 转换为numpy的array格式
    return b,g,r


if __name__ == '__main__':
    img = cv2.imread("my_cat.jpg")
    topil=transforms.ToPILImage()
    # 读进来的图像都要经过这样的操作，这才是torch体系可以使用的
    img = topil(img)
    ## 翻转操作有两类，一个是左右翻转，一个是垂直翻转，这个是水平翻转
    flip_random = transforms.RandomHorizontalFlip()
    ## 这个是垂直翻转
    flip_random_horize = transforms.RandomVerticalFlip()
    # b,g,r=split_3(img)
    # print(type(b))
    imgs = [flip_random(img) for _ in range(2*2)]
    figures1 = plt.figure()
    for i in range(4):
        ax = figures1.add_subplot(2,2,i+1)
        ax.imshow(imgs[i])
    ## 随机剪裁
    shape_oper_norm = transforms.RandomResizedCrop((500,500),
                                                   scale=(0.1,1),
                                                   ratio=(0.5,2))
    imgs = [shape_oper_norm(img) for _ in range(2 * 2)]
    figures2 = plt.figure()

    for i in range(4):
        ax = figures2.add_subplot(2,2,i+1)
        ax.imshow(imgs[i])
    ## 改变颜色，图像颜色的四种描述：亮度，饱和度，对比度，色调
    ## 具体用的时候进行更改
    # 亮度（brightness）、对比度（contrast）、饱和度（saturation）和色调（hue）
    color_random_op = transforms.ColorJitter(brightness= 0.5,
                                             contrast= 0.6,
                                             saturation= 0,
                                             hue= 0)
    imgs = [color_random_op(img) for _ in range(2*2)]
    figures3 = plt.figure()
    for i in range(4):
        ax = figures3.add_subplot(2, 2, i + 1)
        ax.imshow(imgs[i])
    ## 多个结合：几个操作方式的融合
    contact = transforms.Compose([flip_random,
                                  flip_random_horize,
                                  color_random_op,
                                  shape_oper_norm

    ])
    imgs = [contact(img) for _ in range(2 * 2)]
    figures4 = plt.figure()
    for i in range(4):
        ax = figures4.add_subplot(2, 2, i + 1)
        ax.imshow(imgs[i])


    plt.show()

