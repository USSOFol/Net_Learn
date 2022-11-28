"""本代码用于实现Attention神经网络，这是主函数"""
from SSD.data_loader import data
from SSD.Mode.mode_set import TinySSD
from SSD.parameters import parameters
import torch
from SSD.train_test.train import train
from SSD.train_test.test import test
import matplotlib.pyplot as plt
import cv2
from SSD.utiles.predict import predict
"""准备训练数据"""





if __name__ == '__main__':
    config =parameters.Config(batch_size=36,n_classes=1,epoch=10)
    sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
             [0.88, 0.961]]
    ratios = [[1, 2, 0.5]] * 5
    train_data,test_data=data.load_data_bananas(batch_size = config.batch_size())
    model = TinySSD(num_classes=config.n_classes(),sizes=sizes,ratios = ratios).to(config.device())
    """for (x,y) in train_data:
        print(x.size())
        break"""

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr())
    ax1 = []
    ay = []
    plt.ion()
    """开启一个绘图的窗口"""
    train_loss1 = []
    for feature , target in train_data:
        print(feature.size())
        print(target.size())
        break


    for t in range(config.epochs()):
        print(f"epoch {t + 1}\n----------------------")
        train(train_data, model, config.loss_func(), optimizer, config.device(), flag_plot=False, ax=ax1,
              ay=train_loss1, epoch=t)
        # test(test_data, model, config.loss_func(), config.device())
    print("Done!")
    img_test = cv2.imread("banana.png")
    img_test = torch.from_numpy(img_test).permute(2,0,1)
    X = img_test.unsqueeze(0).float()
    img = X.squeeze(0).permute(1, 2, 0).long()
    output = predict(X,net=model,device= config.device())
    print(output)


    # save_path = '../Model_save/SSD.pth'
    # torch.save(model.state_dict(), save_path)
