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
def load_data(data_dir="../CIFAR10",batch_size=None,augment=False,resize=224, thread=4,download_flag=False,shuffle=True,valid_size=0.1,random_seed=1):

    """设置图片参数"""
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    """正规化一个张量图片，使用标准差或者平均值，对于一个三通道图片需要三个变量
    output[channel] = (input[channel] - mean[channel]) / std[channel]"""
    #trans = [transforms.ToTensor()]
    """ToTensor将PIL or numpy 转换为tensor格式"""
    """生成对象"""
    """将读进来的数据从[0,255]转到[0,1]"""
    #if resize:
    #如果resize那为1"""
     #   trans.insert(0, transforms.Resize(resize))

    """把上述参数链接在一起合并为trans"""
    #trans = transforms.Compose([trans, normalize])
    """注意，这个是加载到内存里"""
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # 将图片转换为tensor
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            normalize,
        ])
    valid_transform = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        normalize,
    ])
    """获取数据集"""
    cifar10_train = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, transform=train_transform, download=download_flag)
    cifar10_test = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, transform=valid_transform, download=download_flag)

    """对数据集进行打乱操作"""
    num_train=len(cifar10_train)
    indices=list(range(num_train))
    """获取索引表"""
    split = int(np.floor(num_train*valid_size))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        """获取打乱后的索引"""
    train_idx,test_idx=indices[split:],indices[:split]
    train_sampler=SubsetRandomSampler(train_idx)
    test_sampler=SubsetRandomSampler(test_idx)

    """返回迭代器"""
    return (
        data.DataLoader(cifar10_train, batch_size,sampler=train_sampler,
                        num_workers=thread),
        data.DataLoader(cifar10_test, batch_size,
                        num_workers=thread)
    )
"""
torch.utils.data.DataLoader(dataset, 
batch_size=1, 
#分组
shuffle=None,
#ture的话每轮打乱一次
sampler=None,
#定义从数据集绘制样本的策略
batch_sampler=None, 
#
num_workers=0, 
collate_fn=None,
pin_memory=False,
drop_last=False, 
timeout=0, 
worker_init_fn=None, 
multiprocessing_context=None,
generator=None, 
*, 
prefetch_factor=2, 
persistent_workers=False, 
pin_memory_device='')
"""
if __name__ == '__main__':
    iter_test, iter_train = load_data(batch_size=512, download_flag=False)
    for x, y in iter_test:
        print(x.shape)
        break