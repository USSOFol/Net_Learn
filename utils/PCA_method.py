"""
PCA方法的实现
"""
import torch
import torch.nn as nn
import sys
class PCA(nn.Module):
    def __init__(self,out_features):
        """PCA用于降维，不用于升维"""
        super(PCA,self).__init__()
        #self.in_feature = in_features
        # 原有的维度
        self.out_feature = out_features
        # 需要提取到的维度
        #self.lay1 = self._feature_norm()

    def _feature_norm(self,input):
        N,I,H,G=input.size()
        if I < self.out_feature:
            raise SystemExit('the dim of data must be lager than the out_features ')
        means = torch.mean(input.view(N,I,-1),dim = 1).repeat_interleave(I,dim = 0).view(N,I,-1)
        means_central = input.view(N,I,-1) - means
        stds = torch.std(input.view(N,I,-1),dim =1).repeat_interleave(I,dim = 0).view(N,I,-1)
        input_norm = means_central/stds
        return input_norm
    def _pca(self,input):
        input = torch.bmm(input,input.transpose(1,2))
        U,S,D = torch.svd(input)
        U = U[:,:,:self.out_feature]
        #选择需要的主成分
        return U

    def forward(self,input):
        N, I, H, G = input.size()
        input1 = self._feature_norm(input)
        U = self._pca(input1)
        U = U.transpose(1,2)
        #选择需要的前out_feature个输出
        return torch.bmm(U,input.view(N,I,-1)).view(N, self.out_feature, H, G)


if __name__ == "__main__":
    #a= torch.tensor([  [  [[1.],[2]],[ [3],[4]]   ]   ])
    a = torch.randn(2,3,2,2)
    print(a.size())
    cc = a.view(2,3,-1)
    print(cc)
    means = torch.mean(cc,dim = 1)
    means = means.repeat_interleave(3,dim = 0).view(2,3,-1)

    c_a =cc - means
    stds = torch.std(cc, dim=1).repeat_interleave(3,dim = 0).view(2,3,-1)
    pca = PCA(2)
    print(pca(a))
    #print(torch.svd(cc))
    d=cc.transpose(1,2)
    print(d)
    xx=torch.bmm(cc,d)
    print(xx)














