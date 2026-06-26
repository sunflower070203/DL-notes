"""
1.多分类任务损失函数
在多分类任务中，常用的损失函数是交叉熵损失函数（Cross-Entropy Loss）。交叉熵损失函数用于衡量模型预测的概率分布与真实标签之间的差异。对于多分类问题，交叉熵损失函数的定义如下：
L = -∑(y_i * log(p_i))
p_i=Softmax(f_i) 

其中y_i是实际标签的one-hot编码，
p_i是模型预测的概率分布，
f_i是模型输出的原始分数（logits）。也就是加权求和之后的结果。

交叉熵损失函数在多分类任务中表现良好，因为它能够有效地衡量预测概率分布与真实标签之间的差异，并且在训练过程中能够提供有意义的梯度信息。

"""
import torch
import torch.nn as nn

def dm_01():
    y_true = torch.tensor([1,2]) #实际标签的one-hot编码
    y_pred = torch.tensor([[0.1, 0.8, 0.1],[0.2, 0.3, 0.5]],dtype=torch.float32) #模型预测的概率分布
    criterion = nn.CrossEntropyLoss() #定义交叉熵损失函数
    loss = criterion(y_pred, y_true) #计算损失
    print(f"交叉熵损失: {loss}")


"""
2.二分类任务损失函数
在二分类任务中，常用的损失函数是二元交叉熵损失函数（Binary Cross-Entropy Loss）。二元交叉熵损失函数用于衡量模型预测的概率与真实标签之间的差异。对于二分类问题，二元交叉熵损失函数的定义如下：
L = -[y * log(p) + (1 - y) * log(1 - p)]

其中y是实际标签（0或1），
p是模型预测的概率值（介于0和1之间）。

二元交叉熵损失函数在二分类任务中表现良好，因为它能够有效地衡量预测概率与真实标签之间的差异，并且在训练过程中能够提供有意义的梯度信息。



"""

def dm_02():
    y_true = torch.tensor([1,0]) #实际标签
    y_pred = torch.tensor([[0.8],[0.2]],dtype=torch.float32) #模型预测的概率值
    criterion = nn.BCELoss() #定义二元交叉熵损失函数
    loss = criterion(y_pred, y_true.float()) #计算损失
    print(f"二元交叉熵损失: {loss}")


if __name__ == "__main__":
    dm_01()
    dm_02()



"""
3.MAE（Mean Absolute Error）损失函数
MAE损失函数用于衡量模型预测值与真实值之间的平均绝对误差。它的定义如下：
L = (1/n) * ∑|y_i - p_i|
其中y_i是实际值，p_i是模型预测值，n是样本数量。
MAE损失函数在回归任务中表现良好，因为它能够有效地衡量预测值与真实值之间的差异，并且对异常值不敏感。

缺点：
- MAE损失函数在优化过程中可能会导致梯度不连续，影响模型的训练效率。
- 零点导数问题：当预测值与真实值相等时，MAE的梯度为零，可能导致模型无法继续学习。



4.MSE（Mean Squared Error）损失函数
MSE损失函数用于衡量模型预测值与真实值之间的平均平方误差。它的定义如下：
L = (1/n) * ∑(y_i - p_i)^2
其中y_i是实际值，p_i是模型预测值，n是样本数量。
MSE损失函数在回归任务中表现良好，因为它能够有效地衡量预测值与真实值之间的差异，并且对异常值敏感。

缺点：
- MSE损失函数对异常值敏感，可能导致模型过度拟合异常值。
- 如果差值过大，可能会导致梯度爆炸，影响模型的训练效率。
"""

"""
5.Smooth L1 Loss（平滑L1损失函数）
Smooth L1 Loss是一种结合了MAE和MSE优点的损失函数，定义如下：
L = (1/n) * ∑smooth_l1(y_i - p_i)
其中smooth_l1(x) = 0.5 * x^2   if |x|<1 
                   |x| - 0.5  otherwise
Smooth L1 Loss在回归任务中表现良好，因为它能够有效地衡量预测值与真实值之间的差异，并且对异常值具有一定的鲁棒性。

缺点：
- Smooth L1 Loss在优化过程中可能会导致梯度不连续，影响模型的训练效率。
- Smooth L1 Loss在某些情况下可能会导致模型过度拟合异常值。



"""