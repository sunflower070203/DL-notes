#张量

import torch

#1.根据指定数据创建张量
a = torch.tensor([1,2,3])
print(a)

#2.根据指定形状创建张量
b = torch.zeros((2,3)) #全0张量 
print(b)
c = torch.ones((2,3)) #全1张量
print(c)
d = torch.rand((2,3)) #随机张量 
print(d)

#3.创建指定类型张量
e = torch.IntTensor([1,2,3]) #整数张量
f= torch.FloatTensor([1.0,2.0,3.0]) #浮点数张量
g= torch.LongTensor([1,2,3]) #长整数张量
print(e)
print(f)
print(g)

#4.numpy nd 数组
import numpy as np
h = np.array([1,2,3])
i = torch.IntTensor(h)
print(i)