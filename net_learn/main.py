"""本代码用于实现lenet神经网络，这是主函数"""
from data_load import load_data
from mode.googlenet_mode import GoogLeNet
from parameters import Config
import torch
from train import train
from test import test
import matplotlib.pyplot as plt
"""准备训练数据"""
if __name__ == '__main__':
    config=Config()
    """加载parameter"""
    train_dataloader,test_dataloader=load_data(batch_size=config.batch_size(),resize=config.img_size(),download_flag=True)
    """"""
    torch.manual_seed(config.random_seed())
    """set model"""
    net=GoogLeNet(config.n_classes()).to(config.device())
    """set optimizer"""
    optimizer=torch.optim.Adam(net.parameters(),lr=config.lr())
    ax1=[]
    ay=[]
    plt.ion()
    """开启一个绘图的窗口"""
    train_loss1=[]
    for t in range(config.epochs()):
        print(f"epoch {t+1}\n----------------------")
        train(train_dataloader,net,config.loss_func(),optimizer,config.device(),flag_plot=False,ax=ax1,ay=train_loss1,epoch=t)
        test(test_dataloader,net,config.loss_func(),config.device())
    print("Done!")


















