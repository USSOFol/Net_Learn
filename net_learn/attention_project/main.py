"""本代码用于实现Attention神经网络，这是主函数"""
from Data_loader.data_load import load_data
from Mode.VGG_attention import AttenVGG
from Parameters import parameters
import torch
from train_test.train import train
from train_test.test import test
import matplotlib.pyplot as plt
"""准备训练数据"""
if __name__ == '__main__':
    config =parameters.Config(batch_size=100,n_classes=100)
    train_data,test_data=load_data(data_dir='CIFAR100',batch_size = config.batch_size())
    model = AttenVGG(sample_size=config.img_size(),num_classes=config.n_classes()).to(config.device())
    """for (x,y) in train_data:
        print(x.size())
        break"""

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr())
    ax1 = []
    ay = []
    plt.ion()
    """开启一个绘图的窗口"""
    train_loss1 = []
    for t in range(config.epochs()):
        print(f"epoch {t + 1}\n----------------------")
        train(train_data, model, config.loss_func(), optimizer, config.device(), flag_plot=False, ax=ax1,
              ay=train_loss1, epoch=t)
        test(test_data, model, config.loss_func(), config.device())
    print("Done!")
    save_path = 'Model_save/Attention_VGG.pth'
    torch.save(model.state_dict(), save_path)

