import torch.nn as nn
import torch
import numpy
import torch.nn.functional as F1
a = torch.tensor([[[1.,0.245,11],[1.,0.88,0.66]]])
c = nn.ReLU(inplace=False)
print(c(a))
e = nn.ReLU(inplace=True)
print(e(a))
t,y,u=a.size()
print(a.view(t,1,-1))
class F:
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def __sum__(self):
        return self.x+self.y
print(F(1,2))
class Outer:

    def __init__(self,fun):  # 函数名作为参数
        self.fun = fun

    def __call__(self, *args, **kwargs):  # 1、__call__自动调用和返回内部函数
        print('执行函数前')
        result = self.fun(*args, **kwargs)  # 2、内部函数引用外部函数变量
        print('执行函数后')
        return result

@Outer         # 装饰器使用
def func(a):
    print('普通函数')
    return a

f = func(3)
print(f)  # 返回：3
queries = torch.tensor([[1,1],[2,2],[3,3]])
keys = torch.tensor([[2,3],[1,3],[1,2]])
print(((queries - keys) )**2 / 2)
cc= nn.functional.softmax(-((queries - keys) )**2 / 2, dim=1)
print(nn.functional.softmax(-((queries - keys) )**2 / 2, dim=1))

print(cc.squeeze(-2).size())
print(cc.reshape(-1))
aaa=torch.tensor([[[4,5],[2,1]]])
print(aaa.reshape(-1))
aaaa=torch.tensor([1,2,3,4])
aaaa1=torch.tensor([3,2,1,0])
al=aaaa.reshape((2,2,1))
al1=aaaa1.reshape((2,1,2))
print(al1)
print(al)
print(torch.bmm(al,al1))

x = torch.randn(2, 3,4)
print(x)
print(x[:,-1,:])

print(queries.squeeze(1).size())


op = nn.Conv2d(in_channels=3, out_channels=1,kernel_size=1, padding=0, bias=False)
l = torch.randn(2, 1, 2, 2)
print(l)
print(l.view(2,1,-1))
print(F1.softmax(l.view(2,1,-1), dim=2).view(2,1,2,2))
print()
aa = torch.randn(2, 3, 2, 2)
print(l.expand_as(aa))

print(torch.mul(l.expand_as(aa),aa))

aa = torch.tensor([[1,1],[1,2]])
bb = torch.tensor([[1,1],[1,3]])
print(torch.mul(aa,bb))
print(l.view(2,1,-1).sum(dim=2))