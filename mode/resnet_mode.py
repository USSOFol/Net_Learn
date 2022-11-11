import torch.nn as nn
import torch
import torch.nn.functional as F
"""
resnet存在两种特殊残差网络块，同样是为了解决网络深度加深后的梯度爆炸和梯度消失问题
使用batchnorm可以在一定程度上阻碍梯度爆炸与梯度消失问题
这也仅仅是对十几层的网络存在效果
网络过深则没有效果了，因此引入了这么一个模型
"""
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self,in_plans,planes,stride=1):
        super(BasicBlock, self).__init__()
        ## 若stride为之时，及输入输出的大小不变，仅仅是个数变化之时
        self.conv1 = nn.Conv2d(in_channels=in_plans,out_channels=planes,kernel_size=(3,3),stride=(stride,stride),padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(in_channels=planes,out_channels=planes,kernel_size=(3,3),stride=(1,1),padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        ## 由于resnet网络存在升维度或者降维度的情况
        if stride !=1 or in_plans != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_plans,out_channels=self.expansion*planes,kernel_size=1,stride=stride,bias=False)
                ,nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self,x):
        out=self.conv1(x)
        out=self.bn1(out)
        out=F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module()):
    def __init__(self,block,num_blocks,num_classes=10):
        # block:残差结构
        # num_block：残差结构数目
        # num_classes:区分的种类数目
        super(ResNet,self).__init__()
        self.in_plans=64

        self.conv1 = nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        """
        整个resnet网络无论类型如何都遵循这一个降采样和升维度的过程
        初始：3 224 224
        卷积：64 112 112
        池化：64 56 56
        残差和1：64 56 56
        残差和2：128 28 28
        残差和3：256 14 14
        残差和4：512 7 7
        池化：512 1 1 
        """
        #block,
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion,num_classes)


    def _make_layer(self,block,planes,num_blocks,stride):
        strides = [stride] +[1]*(num_blocks-1)
        # 第一次进行降采样，后续保持stride = 1
        layers=[]
        # 空列表用于存储后续的残差层
        for stride in strides:
            layers.append(
                block(self.in_plans,planes,stride)
            )
            self.in_plans = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out,4)
        out = out.view(out.size(0),-1)
        out = self.linear(out)
        return out

def resnet18(**kwargs):
    return ResNet(BasicBlock, [2,2,2,2], **kwargs)

if __name__ == "__main__":
    res18 = resnet18()
    print(res18)





