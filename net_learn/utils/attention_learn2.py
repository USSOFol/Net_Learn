import torch
import torch.nn as nn
import matplotlib.pyplot as plt

weights = torch.ones((2, 10)) * 0.1
values = torch.arange(20.0).reshape((2, 10))
print(weights,'\n',values)
print("b",weights.unsqueeze(1),'\n','q:', values.unsqueeze(-1))
print(torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1)))
print(nn.Parameter(torch.rand((1,), requires_grad=True)))
#keys.shape[1]).reshape((-1, keys.shape[1])
#print(queries.repeat_interleave())
class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))
        #设置为可训练类型,初始化为0-1之间的随机数

    def forward(self, queries, keys, values):
        # queries为待训练量和attention_weights的形状为(查询个数，“键－值”对个数)
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        #print(queries)
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w)**2 / 2, dim=1)
        # 按照行进行计算
        # values的形状为(查询个数，“键－值”对个数)
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)
        # 50x1x49 bmm 50x49x1
n_train = 50  # 训练样本数
x_train, _ = torch.sort(torch.rand(n_train) * 5)   # 排序后的训练样本

def f(x):
    return 2 * torch.sin(x) + x**0.8

y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))  # 训练样本的输出
x_test = torch.arange(0, 5, 0.1)  # 测试样本
y_truth = f(x_test)  # 测试样本的真实输出
n_test = len(x_test)  # 测试样本数
print(n_train)
# X_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输入
X_tile = x_train.repeat((n_train, 1))
#按照行进行50次数重复
print(X_tile)
# Y_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输出
Y_tile = y_train.repeat((n_train, 1))
#按照行进行50次数重复
# keys的形状:('n_train'，'n_train'-1)
print(X_tile)
keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
#reshape(-1,n)为按照n列调整，自动分配行,reshape(n,-1)为按照n行调整，自动分配列
#这一句是去除对角线后在将其转为50行的张量
print('keys:',keys)
# keys为50*49的矩阵
print('x_train:',x_train)
# values的形状:('n_train'，'n_train'-1)
values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
# values也是这么处理的
print('values',values.unsqueeze(-1).size())
net = NWKernelRegression()
print(net)
loss = nn.MSELoss(reduction='none')
# 均方根误差
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
# 梯度下降法
# animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(10):
    trainer.zero_grad()
    l = loss(net(x_train, keys, values), y_train)
    l.sum().backward()
    trainer.step()
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    #animator.add(epoch + 1, float(l.sum()))
with torch.no_grad():
    y_pred =  net(x_test,keys,values)
print(y_pred)
plt.figure()
plt.plot(x_test.reshape(-1),y_pred,x_test,y_truth)
plt.show()

