#自动微分模块
#梯度=损失函数对权重参数的导数
"""
1.自动微分简介
自动微分是深度学习中非常重要的一个概念，它使得我们能够自动计算模型参数的梯度，从而进行优化。PyTorch中的自动微分模块叫做autograd，它提供了一个动态计算图，可以在运行时构建和修改计算图。

==对损失函数求导，结合反向传播，更新权重参数w,b
==权重更新公式为：w = w - learning_rate * dw
    其中dw是损失函数对权重参数w的梯度，learning_rate是学习率，控制权重更新的步长。
==偏置参数b的更新公式为：b = b - learning_rate * db
    其中db是损失函数对偏置参数b的梯度。

2.相关API
- torch.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False, only_inputs=True, allow_unused=False) #计算指定输出张量对输入张量的梯度
- torch.autograd.backward(tensors, grad_tensors=None, retain_graph=None, create_graph=False, inputs=None, grad_variables=None) #通过反向传播计算所有依赖于输出张量的输入张量的梯度
- torch.autograd.Function   #自定义函数
- torch.autograd.Variable #已废弃，建议直接使用torch.tensor
- torch.autograd.set_grad_enabled(mode) #启用或禁用梯度计算

3.梯度计算
pytorch不支持张量向量化求导，必须使用autograd.grad或者backward函数来计算梯度。autograd.grad函数可以计算指定输出张量对输入张量的梯度，而backward函数则是通过反向传播来计算所有依赖于输出张量的输入张量的梯度。

y.backward() #y为标量
"""
import torch

w= torch.tensor(2.0, requires_grad=True, dtype=torch.float) #创建一个可求导的张量

loss = 2*w**2 #定义损失函数
loss.sum().backward() #计算损失函数对w的梯度 并且保证loss是1个标量

w.data -= 0.1 * w.grad.data #更新权重参数w
print(w)