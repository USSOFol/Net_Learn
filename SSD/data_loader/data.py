import os
import pandas as pd
import torch
import cv2
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from d2l import torch as d2l
#@save
"""
d2l.DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')
"""
def read_data_bananas(data_dir = "data/banana-detection" , is_train=True):
    """读取香蕉检测数据集中的图像和标签"""

    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val', 'label.csv')
    print(csv_fname)
    # 如果istrain为true，拼接为，
    csv_data = pd.read_csv(csv_fname)
    #     img_name  label  xmin  ymin  xmax  ymax
    # print(csv_data)
    csv_data = csv_data.set_index('img_name')
    # 将图像名称作为索引
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        # iterrows以行作为迭代器，输出index 和index后的目标
        path_name_img = os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val', 'images', f'{img_name}')
        # print(path_name_img)
        img = cv2.imread(path_name_img)
        # 注意cv2读出来的是三个通道
        # print(img.shape)
        img = torch.from_numpy(np.array([img[:, :, 0], img[:, :, 1], img[:, :, 2]]))
        # img = torch.from_numpy(img)
        # print(img.shape)
        images.append(img)
        # image_banana = torchvision.io.read_image(path_name_img)
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256
#@save
class BananasDataset(torch.utils.data.Dataset):
    """一个用于加载香蕉检测数据集的自定义数据集"""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train=is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)
#@save
def load_data_bananas(batch_size):
    """加载香蕉检测数据集"""
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
                                             batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
                                           batch_size)
    return train_iter, val_iter
if __name__ == '__main__':
    batch_size, edge_size = 100, 256
    train_iter, val_iter = load_data_bananas(batch_size)
    batch = next(iter(train_iter))
    #print(batch[0][1].shape, batch[1].shape)
    print(batch[0][1].shape)