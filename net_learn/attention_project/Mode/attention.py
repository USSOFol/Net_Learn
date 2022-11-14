import torch
import torch.nn as nn
import torch.nn.functional as F
class PBlocker(nn.Module):
    def __init__(self,in_features,out_features):
        super(PBlocker,self).__init__()
        self.op = nn.Conv2d(
            in_channels= in_features,
            out_channels= out_features,
            kernel_size=1,
            padding= 0,
            bias=False
        )
    def forward(self,x):
        return self.op(x)

class Spatial_Attention(nn.Module):
    # 空间注意力模型
    def __init__(self,in_features,normalize_Flag=True):
        super(Spatial_Attention,self).__init__()
        self.normalize_Flag = normalize_Flag
        self.op = nn.Conv2d(
            in_channels=in_features,
            out_channels=1,
            kernel_size=1,
            padding=0,
            bias=False
        )
    def forward(self,l,g):
        N,C,H,W = l.size()
        c = self.op(l+g)
        # c:N 1 H W
        if self.normalize_Flag:
            a=F.softmax(c.view(N,1,-1),dim = 2).view(N,1,H,W)
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l),l)
        if self.normalize_Flag:
            g = g.view(N, C, -1).sum(dim=2)  # (batch_size,C)
            # 降维整形为N C HW 到N C
        else:
            g = F.adaptive_avg_pool2d(g, (1, 1)).view(N, C)
        return c.view(N, 1, H, W), g


