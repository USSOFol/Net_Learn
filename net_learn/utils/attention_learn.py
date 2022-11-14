import torch
from torch import nn
from d2l import torch as d2l
import torch.nn.functional as F
import matplotlib.pyplot as plt
n_train = 50  # 训练样本数
x_train, _ = torch.sort(torch.rand(n_train) * 5)   # 排序后的训练样本
# 获取0-5的随机数
def f(x):
    return 2 * torch.sin(x) + x**0.8

y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))  # 训练样本的输出
x_test = torch.arange(0, 5, 0.1)  # 测试样本

y_truth = f(x_test)  # 测试样本的真实输出
n_test = len(x_test)  # 测试样本数
print(n_test)
## 绘图
plt.plot(x_test,y_truth,x_train,y_train,'bs')

# 绘制图片，其中曲线为真实值，点可以认为是采样点加上其随机偏差

## 平均汇聚
y_hat = torch.repeat_interleave(y_train.mean(), n_test)
print(y_hat)
print(y_train.mean())
plt.plot(x_test,y_hat,'-')
plt.grid()


## 非参数注意力汇聚
X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
# 50*50的张量
# 问题：如果是在神经网络中如何使用？？？看不懂操作啊
print(X_repeat)
attention_weights= F.softmax(-(X_repeat-x_train)**2/2,dim =0)
print(attention_weights)
y_hat = torch.matmul(attention_weights,y_train)
plt.plot(x_test,y_hat)
plt.show()

"""批量矩阵乘法"""
X = torch.ones((2, 1, 4))
Y = torch.ones((2, 4, 6))
print(X)
print(Y)
print(torch.bmm(X, Y))





