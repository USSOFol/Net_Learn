import torch.nn as nn
# import torch.utils.model_zoo as model_zoo
import math

class VGG(nn.Module):
    def __init__(self,cfg,batch_norm=False,classes = 1000):
        super(VGG,self).__init__()
        self.cfg = cfg
        self.batch_norm = batch_norm
        self.features = self._make_layers()
        self.classifier = nn.Linear(512,classes)
        self._initialize_weights()

    def _initialize_weights(self):
        #初始化权重
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                # 为什么Vgg要做这些操作？
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0,math.sqrt(2./n))
                # 这里对weights做什么操作没看懂
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m,nn.BatchNorm2d):
                #这里不懂
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m,nn.Linear):
                #全连接层做权值初始化为高斯分布
                #偏置为0
                n = m.weight.size(1)
                m.weight.data.normal_(0,0.01)
                m.bias.data.zero_()

    def _make_layers(self):
        #创建训练网络
        layers =[]
        # 创建空列表用于存放网络
        in_channels = 3
        for v in self.cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
                # 最大汇聚，将图像降采样为原来的1/2尺寸
            else:
                conv2d = nn.Conv2d(in_channels,v,kernel_size=3,padding=1)
                if self.batch_norm:
                    # 如果添加归一化
                    layers += [conv2d,nn.BatchNorm2d(v),nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d,nn.ReLU(inplace=True)]
                in_channels = v
        return  nn.Sequential(*layers)

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
def vgg11(**kwargs):
    model =VGG(cfg = cfg['A'],**kwargs )
    return model
def vgg16(**kwargs):
    model =VGG(cfg = cfg['D'],**kwargs )
    return model
def vgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = VGG(cfg=cfg['A'], batch_norm=True, **kwargs)
    return model

if __name__ == '__main__':
    vg16 = vgg11_bn()
    print(vg16)