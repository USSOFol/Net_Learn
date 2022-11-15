import torch
import torch.nn as nn
import torch.nn.functional as F
import d2l.torch as d2l
def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences.

    Defined in :numref:`sec_seq2seq_decoder`"""
    maxlen = X.size(1)
    print("X1:",X)
    print("maxlen:",maxlen)
    mask = torch.arange((maxlen), dtype=torch.float32,device=X.device)[None, :] < valid_len[:, None]
    print('mask:',mask)
    X[~mask] = value
    return X
def masked_softmax(x,valid_lens):
    """超过有效长度的位置都将被掩蔽置为0"""
    if valid_lens is None:
        return F.softmax(x,dim = -1)
    else:
        shape = x.shape
        if valid_lens.dim() ==1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
            print("1:",valid_lens)
        else:
            valid_lens = valid_lens.reshape(-1)
            print("2:", valid_lens)
        print('valid:',valid_lens)
        x = sequence_mask(x.reshape(-1, shape[-1]), valid_lens,value=-1e6)
        print('x:',x)
    return nn.functional.softmax(x.reshape(shape), dim=-1)

#@save
class AdditiveAttention(nn.Module):
    """加性注意力"""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        # 全连接层，键
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        # 全连接层，查询量
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        # 随机丢弃一部分权值，将数值改为0

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 在维度扩展后，
        # queries的形状：(batch_size，查询的个数，1，num_hidden)
        # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
        # 使用广播方式进行求和
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # self.w_v仅有一个输出，因此从形状中移除最后那个维度。
        # scores的形状：(batch_size，查询的个数，“键-值”对的个数)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        return torch.bmm(self.dropout(self.attention_weights), values)

if __name__ == '__main__':
    """x = torch.randn(2,3,1,2)
    print(x.dim())
    re = F.softmax(x,dim= -1)
    #按照行计算softmax
    print(re)"""
    a=torch.rand(2, 2, 4)
    print('a:',a)

    b=torch.tensor([2, 3])
    #指定a中第一个中只有2列有效，第二个为只有第三列有效
    print('b',b)
    print(masked_softmax(a,b))
    dropout = nn.Dropout(0.5)
    print(dropout(a))
##################################################
    queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
    # values的小批量，两个值矩阵是相同的
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(
        2, 1, 1)
    valid_lens = torch.tensor([2, 6])

    attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8,
                                  dropout=0.1)
    attention.eval()
    print(attention(queries, keys, values, valid_lens))


