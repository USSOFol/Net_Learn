from torchvision import transforms
import torchvision
from torch.utils import data
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
"""获取训练数据集"""
"""
input:
batch_size
resize:图片转换形状
thread：线程
download_flag：是否下载训练集
data_dir
"""
def load_data(data_dir="D:\code\python\attention_project/CIFAR100",batch_size=None,augment=False,resize=224, thread=2,download_flag=False,shuffle=True,valid_size=0.1,random_seed=1):
    # load data,CIFAR100数据集大小为100*100
    # 定义批量操作：
    train_transform = transforms.Compose([
        transforms.RandomCrop(32,padding=4),
        #随机裁剪
        transforms.RandomHorizontalFlip(),
        #随机翻转
        transforms.ToTensor(),
        #转张良
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    """获取数据集"""
    cifar100_train = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, transform=train_transform, download=download_flag)
    cifar100_test = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, transform=test_transform, download=download_flag)
    return (
            data.DataLoader(cifar100_train, batch_size,shuffle=True,
                        num_workers=thread),
            data.DataLoader(cifar100_test, batch_size,shuffle=True,
                        num_workers=thread)
            )

if __name__ == '__main__':
    iter_test, iter_train = load_data(batch_size=100, download_flag=True)
    for x, y in iter_test:
        print(x.shape)
        break