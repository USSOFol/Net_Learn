import torch.nn as nn
import torch
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

