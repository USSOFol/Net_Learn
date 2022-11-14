import torch
import torch.nn as nn
import torch.nn.functional as F
from Mode.attention import PBlocker,Spatial_Attention

class AttenVGG(nn.Module):
    def __init__(self,sample_size,num_classes,attention_flag = True,normalize_att=True):
        super(AttenVGG,self).__init__()
        #convblocks
        self.attention_flag = attention_flag
        self.num_classes = num_classes
        self.normalize = normalize_att
        self.conv1 = self._make_layer(3, 64, 2)
        self.conv2 = self._make_layer(64, 128, 2)
        self.conv3 = self._make_layer(128, 256, 3)
        self.conv4 = self._make_layer(256, 512, 3)
        self.conv5 = self._make_layer(512, 512, 3)
        self.conv6 = self._make_layer(512, 512, 2, pool=True)
        self.dense = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=int(sample_size / 32), padding=0,
                               bias=True)
        # attention blocks
        if self.attention_flag:
            self.projector = PBlocker(256,512)
            # 用于升维度
            self.attn = Spatial_Attention(in_features=512,normalize_Flag = normalize_att)
            # 空间注意力计算模块
            # 最终的分类层
        if self.attention_flag:
            self.classify = nn.Linear(in_features=512 * 3, out_features=num_classes, bias=True)
        else:
            self.classify = nn.Linear(in_features=512, out_features=num_classes, bias=True)
            # if init_weights:
            #     self._initialize_weights()

    def _make_layer(self,in_features,out_features,blocks,pool = False):
        layers = []
        for i in range(blocks):
            cov2d = nn.Conv2d(in_channels= in_features,
                              out_channels= out_features,
                              kernel_size=3,
                              padding=1,
                              bias=False)
            layers += [cov2d,nn.BatchNorm2d(out_features),nn.ReLU(inplace=True)]
            in_features = out_features
            if pool:
                layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
        return nn.Sequential(*layers)
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        l1 = self.conv3(x)
        # 256x图片尺寸
        # l1进入attention中,注意，这里的l1无法直接放入注意力模块当中，需要进行升维之后放入
        x = F.max_pool2d(l1, kernel_size=2, stride=2, padding=0)
        # 图片缩小
        l2 = self.conv4(x)
        # 512x半个图片尺寸
        # l2进入attention中
        x = F.max_pool2d(l2, kernel_size=2, stride=2, padding=0)
        # 继续缩小一半
        l3 = self.conv5(x)
        # 512x图片尺寸/4
        # l3 进入attention中
        x = F.max_pool2d(l3, kernel_size=2, stride=2, padding=0)
        # 512x图片/8
        x = self.conv6(x)
        # 512x图片/32
        g = self.dense(x)  # batch_sizex512x1x1
        # 512x1
        # attention
        if self.attention_flag:
            l1 = self.projector(l1)
            c1,g1 = self.attn(l1,g)
            c2,g2 = self.attn(l2,g)
            c3,g3 = self.attn(l3,g)
            # 这里的g就是查询量，寻找每一层最相关的量进行输出
            g = torch.cat((g1, g2, g3), dim=1)  # batch_sizex3C
            # 按照列进行拼接，扩充列
            # 最终分类使用g
            x = self.classify(g)  # batch_sizexnum_classes
        else:
            c1, c2, c3 = None, None, None
            x = self.classify(torch.squeeze(g))
        return x, c1, c2, c3
if __name__ == '__main__':
    a = AttenVGG(sample_size=32,num_classes=100)
    img = torch.randn(100,3,32,32)
    pre,_,_,_ =a(img)
    print(pre.size())


