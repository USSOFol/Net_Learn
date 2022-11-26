import matplotlib.pyplot as plt
from SSD.utiles.loss import calc_loss
from SSD.utiles.loss import cls_eval
from SSD.utiles.loss import bbox_eval
from SSD.utiles.box import Box

def train(dataloader, model, loss_fn, optimizer,device,flag_plot=False,ax=[],ay=[],epoch=0):
    """
    :param dataloader: 数据
    :param model: 模型
    :param loss_fn:损失函数
    :param optimizer: 优化函数
    :param device: 计算地点
    :param flag_plot: 画图标志
    :param ax: 横坐标
    :param ay: 纵坐标
    :param epoch: 轮
    :return:
    """
    """先前向传播，计算损失函数，使用优化器反向传播更新权值，顺便计算每100轮后的损失"""
    size = len(dataloader.dataset)
    model.train()
    for batch, (features, target) in enumerate(dataloader):
        """1. 前向传播"""
        X, y = features.to(device), target.to(device)
        # 图片和真实值
        # 数据入cuda
        # Compute prediction error
        # return anchors,cls_pred,bbox_pred
        anchors, cls_preds,bbox_preds = model(X)
        # 生成返回锚框，类别预测，偏移量预测
        bbox_labels, bbox_masks, cls_labels = Box.multibox_target(anchors,y)
        # 根据类别和偏移量的预测和标注值计算损失函数
        loss = calc_loss(cls_preds,cls_labels,bbox_preds,bbox_labels,bbox_masks)
        """item()是提取的数字而不是张量"""
        loss = loss.mean()

        """2.反向传播"""
        # Backpropagation 反向传播
        optimizer.zero_grad()
        # 当前梯度归零
        loss.backward()
        # 反向传播计算梯度
        optimizer.step()
        # 更新梯度

        if batch % 50 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            #plot
            if flag_plot:
                ax.append(epoch*1000+batch)
                ay.append(loss*X.size(0))
                plt.clf()  # 清除之前画的图
                plt.plot(ax, ay)  # 画出当前 ax 列表和 ay 列表中的值的图形
                plt.pause(0.001)  # 暂停一秒
                plt.ioff()  # 关闭画图的窗口"""