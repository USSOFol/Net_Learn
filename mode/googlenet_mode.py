import torch.nn as nn
import torch
import torch.nn.functional as F
"""
googlenet最大的不同就在于其存在许多分离的卷积层，inception模块
"""
class Inception(nn.Module):
    def __init__(self,in_planes,n1x1,n3x3red,n3x3,n5x5red,n5x5,pool_planes):
        """
        :param in_planes: 上一层给我的列数
        :param n1x1:
        :param n3x3red:
        :param n3x3:
        :param n5x5red:
        :param n5x5:
        :param pool_planes:
        """
        super(Inception, self).__init__()
        """一个inception模块拥有四个分支，一个1x1,1x1=>3x3,1x1=>5x5,3x3=>1x1四个卷积分支
        注意最后一个为3x3最大汇聚到1x1卷积分支"""
        self.b1=nn.Sequential(
            nn.Conv2d(in_planes,n1x1,kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(),
        )
        self.b2=nn.Sequential(
            nn.Conv2d(in_planes,n3x3red,kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(),
            nn.Conv2d(n3x3red,n3x3,kernel_size=3,padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(),
        )
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(),
        )
    def forward(self,x):
        y1=self.b1(x)
        y2=self.b2(x)
        y3=self.b3(x)
        y4=self.b4(x)
        return torch.cat([y1,y2,y3,y4],1)
        # concatenate： 连接，拼接的意思
        # 1横着拼接，0竖着拼接，这里横着拼接，输出就是列数之和

class GoogLeNet(nn.Module):
    """
    输入图片为224*224*3尺寸
      GoogLenet拥有9个Inception层
      5个汇聚层
      其他全连接层
      共计22层网络
    """
    def __init__(self,n_classes=10):
        super(GoogLeNet, self).__init__()
        """define the googlenet architecture"""
        self.pre_layer=nn.Sequential(
            nn.Conv2d(3,192,kernel_size=3,padding=1),
            #输入一个三通道图像，输出一个192大小
            #默认步长为一
            nn.BatchNorm2d(192),
            nn.ReLU(),
            )
        #开始定义中间的Inception,九层妖塔
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)
        # (self,in_planes,n1x1,n3x3red,n3x3,n5x5red,n5x5,pool_planes):
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        #  in_planes = 上一次的(n1x1+n3xn3+n5xn5+pool_planes)
        self.a4 = Inception(480, 192,  96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool =  nn.AvgPool2d(8,stride=1)
        self.linear = nn.Linear(1024,n_classes)

    def forward(self,x):
        out = self.pre_layer(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
if __name__ == '__main__':
    m1=GoogLeNet(5)
    print(m1)