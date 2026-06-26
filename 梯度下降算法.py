"""
梯度下降算法
1.简介
寻找损失函数最小化，因为梯度的方向是函数值下降最快的方向

2.公式
w = w - lr * dw
b = b - lr * db

3.相关术语
Epoch: 使用全部数据对模型进行一次训练的过程称为一个Epoch。
Batch_size: 在训练过程中，将数据分成小批次进行训练，每个批次的大小称为Batch_size。
Iteration: 使用一次Batch数据对模型进行一次参数更新的过程
"""


#梯度下降方法——根本区别在于batch_size的大小

#1.BGD  (N)
"""
1.简介
BGD（Batch Gradient Descent）是指在每次迭代中使用整个训练数据集来计算梯度并更新模型参数的方法。由于需要使用整个数据集进行计算，BGD在处理大规模数据时可能会非常慢，并且可能会陷入局部最小值。

2.Batch_size: batch_size = len(train_data) #等于整个数据集的大小
"""

#2.SGD (1)
"""
1.简介
SGD（Stochastic Gradient Descent）是指在每次迭代中随机选择一个样本来计算梯度并更新模型参数的方法。由于每次迭代只使用一个样本进行计算，SGD在处理大规模数据时非常快，并且可以跳出局部最小值，但可能会导致参数更新的噪声较大。

2.Batch_size: batch_size = 1 #每次只使用一个样本进行计算
"""

#3.MBGD (B)
"""
1.简介
MBGD（Mini-Batch Gradient Descent）是指在每次迭代中使用一个小批次的样本来计算梯度并更新模型参数的方法。MBGD结合了BGD和SGD的优点，既可以加快训练速度，又可以减少参数更新的噪声。

2.Batch_size: batch_size = B #每次使用M个样本进行计算，B通常是一个小于整个数据集大小的整数，例如32、64等。
"""