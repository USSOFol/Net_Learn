import os
import pandas as pd
import torch
import torchvision
import cv2
import matplotlib.pyplot as plt
from d2l import torch as d2l

data_dir="../data/banana-detection"
is_train=True
"""读取香蕉检测数据集中的图像和标签"""


csv_fname = os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val', 'label.csv')
print(csv_fname)
# 如果istrain为true，拼接为，
csv_data = pd.read_csv(csv_fname)
#     img_name  label  xmin  ymin  xmax  ymax
#print(csv_data)
csv_data = csv_data.set_index('img_name')
# 将图像名称作为索引
images, targets = [], []
for img_name, target in csv_data.iterrows():
    # iterrows以行作为迭代器，输出index 和index后的目标
    path_name_img=os.path.join(data_dir, 'bananas_train' if is_train else
        'bananas_val', 'images', f'{img_name}')
    # print(path_name_img)
    img = cv2.imread(path_name_img)
    # print(img.shape)
    img = torch.from_numpy(img)
    # print(img.shape)
    images.append(img)
    # image_banana = torchvision.io.read_image(path_name_img)
    targets.append(list(target))



"""
os.path.join(data_dir, 'bananas_train' if is_train else
        'bananas_val', 'images', f'{img_name}'))
"""
"""
for img_name, target in csv_data.iterrows():
    images.append(torchvision.io.read_image(
        os.path.join(data_dir, 'bananas_train' if is_train else
        'bananas_val', 'images', f'{img_name}')))
    # 这里的target包含（类别，左上角x，左上角y，右下角x，右下角y），
    # 其中所有图像都具有相同的香蕉类（索引为0）
    targets.append(list(target))
"""