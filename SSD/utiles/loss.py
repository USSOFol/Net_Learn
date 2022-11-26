"""
这里不能简单的应用原来函数默认的loss函数，要使用范数loss
定义评价函数
"""
import torch
import torch.nn as nn
from SSD.utiles.box import Box

def calc_loss(cls_preds,cls_labels,bbox_preds,bbox_labels,bbox_masks):
    """

    :param cls_preds:
    :param cls_labels:
    :param bbox_preds:
    :param bbox_labels:
    :param bbox_masks:
    :return:损失是锚框类别和便宜量的集合
    """
    cls_loss = nn.CrossEntropyLoss(reduction="none")
    # 类别预测
    bbox_loss = nn.L1Loss(reduction="none")
    batch_size,num_classes = cls_preds.shape[0],cls_preds.shape[2]
    cls = cls_loss(
        cls_preds.reshape(-1,num_classes),
        cls_labels.reshape(-1)
    ).reshape(batch_size,-1).mean(dim = 1)
    bbox = bbox_loss(
        bbox_preds * bbox_masks,
        bbox_labels * bbox_labels
    ).mean(dim=1)
    return cls + bbox

def cls_eval(cls_preds , cls_labels):
    return float((cls_preds.argmax(dim=-1).type(cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds,bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())



