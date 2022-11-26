本文章主要详细介绍一下多框检测的深度学习模型，详细介绍一下为什么这个模型可以达到两个作用：

1、找出物体所在位置

2、对物体进行识别

这里先提供一个简单的例子，这里面的香蕉的框的对角坐标为1.png,0,68,175,118,223

![1](D:\code\python\net_control\SSD\data\banana-detection\bananas_train\images\1.png)

先来看一下传输的forward：

```python
    def forward(self,x):
        anchors, cls_pred, bbox_pred = [None]*5,[None]*5,[None]*5
        for i in range(5):
            # getattr(self,"blk_{i}"),返回self.blk_i
            x, anchors[i], cls_pred[i], bbox_pred[i] = self.blk_forward(x,
                                                                        getattr(self,f"blk_{i}"),
                                                                        self.sizes[i],
                                                                        self.ratios[i],
                                                                        getattr(self,f"cls_{i}"),
                                                                        getattr(self,f"bbox_{i}"))
        anchors = torch.cat(anchors, dim =1)
        cls_pred = self.concat_pred(cls_pred)
        cls_pred = cls_pred.reshape(cls_pred.shape[0],-1,self.num_class+1)
        bbox_pred =self.concat_pred(bbox_pred)
        return anchors,cls_pred,bbox_pred
```

一层一层讲解，看这个网络结构到底是什么样子的。

首先是这个网络的输入，为一个(batch_size,3,img_height,img_width)的形状，首先是五层神经网络。

blk_forward:

```python
    def blk_forward(self,x,blk,size,ratio,cls_prdictor,bbox_predictor):
        """
        :param x: 输入的训练值
        :param blk:
        :param size:锚框长宽缩放比
        :param ratio:锚框宽高比
        :param cls_prdictor:
        :param bbox_predictor:
        :return:
        """
        y =blk(x)
        anchors = Box.multibox_prior(y,sizes=size,ratios = ratio)
        # 返回锚框的对角值
        cls_preds = cls_prdictor(y)
        bbox_preds = bbox_predictor(y)
        return (y , anchors, cls_preds, bbox_preds)
```

```python
    def get_blk(self,i):
        # 选择层
        if i == 0 :
            blk = self.base_net()
        elif i == 1:
            blk = self.down_sample_blk(64,128)
        elif i==4:
            blk = nn.AdaptiveAvgPool2d((1,1))
            # 自适应平均池化层,输出个数为N C 1 1
        else:
            blk = self.down_sample_blk(128,128)
        return blk
```

```python
    def down_sample_blk(self,in_channels,out_channels):
        """
        降采样卷积层
        :param in_channels:
        :param out_channels:
        :return:
        """
        blk = []
        for _ in range(2):
            blk.append(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1))
            blk.append(nn.BatchNorm2d(out_channels))
            blk.append(nn.ReLU())
            in_channels = out_channels
        blk.append(nn.MaxPool2d(2))
        return nn.Sequential(*blk)
```

```python
    def base_net(self):
        blk = []
        num_filter = [3,16,32,64]
        for i in range(len(num_filter)-1):
            blk.append(self.down_sample_blk(num_filter[i],num_filter[i+1]))
        return nn.Sequential(*blk)
```

```python
        for i in range(5):
        	# 这里设置blk,cld,bbox
            setattr(self,f'blk_{i}',self.get_blk(i))
            # 输入为对象,字符串,属性值,设置属性值，但是该属性不一定存在，返回就是self.blk_i = blk
            setattr(self,f"cls_{i}",self.cls_preictor(idx_to_in_channels[i],num_anchors,self.num_class))
            #
            setattr(self,f"bbox_{i}",self.bbox_predictor(idx_to_in_channels[i],num_anchors))
            #
```

```python
    def get_blk(self,i):
        # 选择层
        if i == 0 :
            blk = self.base_net()
        elif i == 1:
            blk = self.down_sample_blk(64,128)
        elif i==4:
            blk = nn.AdaptiveAvgPool2d((1,1))
            # 自适应平均池化层,输出个数为N C 1 1
        else:
            blk = self.down_sample_blk(128,128)
        return blk
```

```python
    def base_net(self):
        blk = []
        num_filter = [3,16,32,64]
        for i in range(len(num_filter)-1):
            blk.append(self.down_sample_blk(num_filter[i],num_filter[i+1]))
        return nn.Sequential(*blk)
```

这里一层一层的进行说明：

输入： (batch_size,3,256,256)尺寸的数据，进入第零层，此时的代码为

```python
 
i = 0 
x, anchors[i], cls_pred[i], bbox_pred[i] = self.blk_forward(x,
                                                                        getattr(self,f"blk_{i}"),
                                                                        self.sizes[i],
                                                                        self.ratios[i],
                                                                        getattr(self,f"cls_{i}"),
                                                                        getattr(self,f"bbox_{i}"))
```

```python
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
             [0.88, 0.961]]
    ratios = [[1, 2, 0.5]] * 5
```

