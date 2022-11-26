import torch
import torch.nn as nn
from SSD.utiles.box import Box



class TinySSD(nn.Module):
    """
    本网络用于多框检测，检测一个模型中出现的类，这里将会对这个模型进行一个详细的讲解
    """
    def __init__(self,num_classes,sizes,ratios,**kwargs):
        """
        输入初始化
        :param num_class:要识别的类
        :param sizes: 锚框高宽缩放比
        :param ratios: 锚框高宽比
        :param kwargs:
        """
        num_anchors = len(sizes[0])+len(ratios[0])-1
        self.sizes = sizes
        self.ratios =ratios
        # 每个点上的锚框个数
        super(TinySSD,self).__init__(**kwargs)
        self.num_class = num_classes
        idx_to_in_channels = [64,128,128,128,128]
        for i in range(5):
            setattr(self,f'blk_{i}',self.get_blk(i))
            # 输入为对象,字符串,属性值,设置属性值，但是该属性不一定存在，返回就是self.blk_i = blk
            setattr(self,f"cls_{i}",self.cls_preictor(idx_to_in_channels[i],num_anchors,self.num_class))
            #
            setattr(self,f"bbox_{i}",self.bbox_predictor(idx_to_in_channels[i],num_anchors))
            #
############# 1 ######################o .]']

    def get_blk(self,i):
        # 选择层
        if i == 0 :
            blk = self.base_net()
        elif i == 1:
            blk = self.down_sample_blk(64,128)
        elif i==4:
            blk = nn.AdaptiveAvgPool2d((1,1))
            # 自适应平均池化层,输出个数为N C 1 1
        else:
            blk = self.down_sample_blk(128,128)
        return blk


    def base_net(self):
        blk = []
        num_filter = [3,16,32,64]
        for i in range(len(num_filter)-1):
            blk.append(self.down_sample_blk(num_filter[i],num_filter[i+1]))
        return nn.Sequential(*blk)



    def down_sample_blk(self,in_channels,out_channels):
        """
        降采样卷积层
        :param in_channels:
        :param out_channels:
        :return:
        """
        blk = []
        for _ in range(2):
            blk.append(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1))
            blk.append(nn.BatchNorm2d(out_channels))
            blk.append(nn.ReLU())
            in_channels = out_channels
        blk.append(nn.MaxPool2d(2))
        return nn.Sequential(*blk)
    ########### 2 ###########################
    def cls_preictor(self,num_inputs,num_anchors,num_classes):
        """
        每个锚框内的物体的预测，也就是类别预测
        :param num_inputs:
        :param num_anchors:
        :param num_classes:
        :return:
        """
        return nn.Conv2d(num_inputs,num_anchors*(num_classes+1),kernel_size=3,padding=1)
    ########### 3 ###############################
    def bbox_predictor(self,num_inputs,num_anchors):
        """
        锚框偏移量预测，众所周知，偏移量是四个值
        :param num_inputs:输入值
        :param num_anchors:锚框数量
        :return:
        """
        return nn.Conv2d(num_inputs,num_anchors*4,kernel_size=3,padding=1)
    ########### 4 #################################
    def blk_forward(self,x,blk,size,ratio,cls_prdictor,bbox_predictor):
        """

        :param x: 输入的训练值
        :param blk:
        :param size:
        :param ratio:
        :param cls_prdictor:
        :param bbox_predictor:
        :return:
        """
        y =blk(x)
        anchors = Box.multibox_prior(y,sizes=size,ratios = ratio)
        # 返回锚框的对角值
        cls_preds = cls_prdictor(y)
        bbox_preds = bbox_predictor(y)
        return (y , anchors, cls_preds, bbox_preds)
    ######### 5 #########################################
    def flatten_pred(self,pred):
        return torch.flatten(pred.permute(0,2,3,1),start_dim= 1 )
    def concat_pred(self, preds):
        return torch.cat([self.flatten_pred(p) for p in preds],dim = 1)
    def forward(self,x):
        anchors, cls_pred, bbox_pred = [None]*5,[None]*5,[None]*5
        for i in range(5):
            # 这里为5是因为锚框大小共有五个等级，每个等级的中心点产生四个大小的锚框，锚框依次变大
            # getattr(self,"blk_{i}"),返回self.blk_i
            x, anchors[i], cls_pred[i], bbox_pred[i] = self.blk_forward(x,
                                                                        getattr(self,f"blk_{i}"),
                                                                        self.sizes[i],
                                                                        self.ratios[i],
                                                                        getattr(self,f"cls_{i}"),
                                                                        getattr(self,f"bbox_{i}"))
        anchors = torch.cat(anchors, dim =1)
        cls_pred = self.concat_pred(cls_pred)
        cls_pred = cls_pred.reshape(cls_pred.shape[0],-1,self.num_class+1)
        bbox_pred =self.concat_pred(bbox_pred)
        return anchors,cls_pred,bbox_pred



if __name__ =="__main__":
    m = nn.AdaptiveAvgPool2d((1,1))
    n = torch.randn(1,64,8,9)
    print(m(n).size())
    sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
             [0.88, 0.961]]
    #print(sizes[0])
    ratios = [[1, 2, 0.5]] * 5
    ssd = TinySSD(1,sizes,ratios)
    print(ssd)
    x = torch.randn(1,3,256,256)
    print(ssd(x))






