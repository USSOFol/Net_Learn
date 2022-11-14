```python
transforms.RandomCrop(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant')
"""size：期望随机裁剪之后输出的尺寸
padding：填充边界的值，单个（int）,两个（[左/右，上/下]），四个（各个边界）
pad_if_needed :bool值，避免数组越界
fill:填充
padding_mode ：填充模式
“constant”:利用常值进行填充
“edge”:利用图像边缘像素点进行填充
“reflect”：利用反射的方式进行填充[1, 2, 3, 4] 》[3, 2, 1, 2, 3, 4, 3, 2]
“symmetric”：对称填充方法[1, 2, 3, 4] 》》[2, 1, 1, 2, 3, 4, 4, 3]
"""
```

