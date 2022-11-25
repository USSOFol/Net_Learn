import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from SSD.utiles.box import Box


def cls_predictor(num_inputs, num_anchors, num_classes):
    """
    这是一个类别预测层
    :param num_inputs:
    :param num_anchors:
    :param num_classes:
    :return:
    """
    # 输入为？，输出为每个中线点上的锚框数量num_anchors * （类比加1），加一是因为有个背景
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),kernel_size=3, padding=1)

def bbox_predictor(num_inputs, num_anchors):
    """
    边界框预测层
    :param num_inputs:
    :param num_anchors:
    :return: 为每一个锚框预测四个偏移量
    """
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)

def forward(x, block):
    """
    用来连接多尺度特征图
    :param x:输入值
    :param block: 处理模块
    :return:
    """
    return block(x)
def down_sample_blk(in_channels, out_channels):
    """
    :param in_channels:输入通道数量
    :param out_channels: 输出通道数量
    :return: 改变通道数量，并用maxpool进行减半采样
    """
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)
def base_net():
    """
    基本块
    :return:
    """
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)

def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)
##############################完整多尺度模型###############################################################
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
        # 减半采样
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:
        blk = down_sample_blk(128, 128)
    return blk

def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    """

    :param X:
    :param blk:
    :param size:
    :param ratio:
    :param cls_predictor:
    :param bbox_predictor:
    :return:
    """
    Y = blk(X)
    anchors = Box.multibox_prior(Y, sizes=size, ratios=ratio)
    #anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)
class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # 即赋值语句self.blk_i=get_blk(i)
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i],
                                                    num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i],
                                                      num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # getattr(self,'blk_%d'%i)即访问self.blk_i
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds

if __name__ == '__main__':
    Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
    Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
    print(Y1.shape, Y2.shape)
    print(forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape)
    print(forward(torch.zeros((2, 3, 256, 256)), base_net()).shape)
    sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
             [0.88, 0.961]]
    ratios = [[1, 2, 0.5]] * 5
    num_anchors = len(sizes[0]) + len(ratios[0]) - 1
    net = TinySSD(num_classes=1)
    X = torch.zeros((32, 3, 256, 256))
    anchors, cls_preds, bbox_preds = net(X)

    print('output anchors:', anchors.shape)
    print('output class preds:', cls_preds.shape)
    print('output bbox preds:', bbox_preds.shape)

