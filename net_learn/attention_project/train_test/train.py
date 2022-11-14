import matplotlib.pyplot as plt
def train(dataloader, model, loss_fn, optimizer,device,flag_plot=False,ax=[],ay=[],epoch=0):
    """先前向传播，计算损失函数，使用优化器反向传播更新权值，顺便计算每100轮后的损失"""
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred,_,_,_ = model(X)
        loss = loss_fn(pred, y.squeeze())
        """item()是提取的数字而不是张量"""


        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        if batch % 50 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            """ plot"""
            if flag_plot:
                ax.append(epoch*1000+batch)
                ay.append(loss*X.size(0))
                plt.clf()  # 清除之前画的图
                plt.plot(ax, ay)  # 画出当前 ax 列表和 ay 列表中的值的图形
                plt.pause(0.001)  # 暂停一秒
                plt.ioff()  # 关闭画图的窗口

    """为什么要这么写"""