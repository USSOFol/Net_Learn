import torch
import torch.nn as nn
import torch.nn.functional as F
"""
本代码用于实现注意力机制
"""
class ProjectorBlock(nn.Module):
    #
    def __init__(self,in_features,out_features):
        #这里进行一个卷积
        super(ProjectorBlock,self).__init__()
        self.op = nn.Conv2d(in_channels=in_features,out_channels=out_features,kernel_size=1,padding=0,bias=False)

    def forwad(self,x):
        x = self.op(x)
        return x

class SpatialAtten(nn.Module):
    def __init__(self,in_features,normalize_att = True):
        super(SpatialAtten,self).__init__()
        self.in_features = in_features
        self.normalize_att=normalize_att
        self.op = nn.Conv2d(in_channels=in_features, out_channels=1, kernel_size=1, padding=0, bias=False)

    def forward(self,l,g):
        # l是什么,g是什么
        N,C,H,W = l.size()
        c = self.op(l+g)
        # 为什么这里要过卷积核
        if self.normalize_att:
            a=F.softmax(c.view(N,1,-1),dim=2).view(N,1,H,W)
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l),l)
        if self.normalize_att:
            g = g.view(N, C, -1).sum(dim=2)  # (batch_size,C)
        else:
            g = F.adaptive_avg_pool2d(g, (1, 1)).view(N, C)
        return c.view(N, 1, H, W), g
"""
Temporal attention block
Reference: https://github.com/philipperemy/keras-attention-mechanism
"""
class TemporalAttn(nn.Module):
    def __init__(self,hidden_size):
        super(TemporalAttn,self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.hidden_size,self.hidden_size,bias=False)
        self.fc2 = nn.Linear(self.hidden_size*2,self.hidden_size,bias=False)

    def forward(self,hidden_states):
        # 输入隐藏层
        score_first_part = self.fc1(hidden_states)



















