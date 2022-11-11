import torch
import torch.nn as nn
class Alexnet(torch.nn.Module):
    def __init__(self,num_classes=10):
        """输入为224*224*3的图像"""
        super(Alexnet,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=96,kernel_size=11,stride=4,padding=0),
            #二维卷积核，3*96尺寸11步长4无填充,
            nn.BatchNorm2d(96),
            # batch_noem：通过减少网络内部
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            # 二维卷积核，3*96尺寸11步长4无填充
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            # 二维卷积核，3*96尺寸11步长4无填充
            nn.BatchNorm2d(384),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            # 二维卷积核，3*96尺寸11步长4无填充
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 二维卷积核，3*96尺寸11步长4无填充
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.fc=nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216,4096),
            nn.ReLU(),
        )
        self.fc1=nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
        )
        self.fc2=nn.Sequential(
            nn.Linear(4096,num_classes)
        )
    def forward(self,x):
        out=self.layer1(x)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.layer4(out)
        out=self.layer5(out)
        out=out.reshape(out.size(0),-1)
        out=self.fc(out)
        out=self.fc1(out)
        out=self.fc2(out)
        return out
if __name__ == '__main__':
    a=Alexnet()
    print(a)



