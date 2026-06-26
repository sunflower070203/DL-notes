#人工神经网络 ANN

"""
1.简介
人工神经网络（Artificial Neural Network，ANN）是一种受生物神经网络启发而设计的计算模型。它由大量的节点（或称为“神经元”）组成，这些节点通过连接（或称为“权重”）相互作用。ANN广泛应用于模式识别、分类、回归等任务。

2.基本结构
- 输入层（Input Layer）：接收外部输入数据。
- 隐藏层（Hidden Layer）：位于输入层和输出层之间，负责处理和转换输入数据。一个ANN可以有多个隐藏层。
- 输出层（Output Layer）：生成最终的输出结果。

3.前向传播
前向传播是指数据从输入层经过隐藏层传递到输出层的过程。在每个神经元中，输入数据会与权重相乘并加上偏置，然后通过激活函数进行非线性变换，最终传递到下一层。

4.反向传播
反向传播是ANN中用于训练模型的关键算法。它通过计算损失函数相对于每个权重的梯度，来调整权重以最小化损失函数。反向传播通常与梯度下降算法结合使用，以更新权重参数。

5.激活函数
激活函数用于引入非线性，使得神经网络能够学习复杂的模式。常见的激活函数包括：
- Sigmoid函数
- ReLU函数
- Tanh函数
通常位于隐藏层和输出层的神经元中。

#神经元的构建
1.输入层：接收输入数据
2.权重：连接输入层和隐藏层的权重参数
3.偏置：每个神经元都有一个偏置参数，用于调整输出
4.激活函数：引入非线性，使得神经网络能够学习复杂的模式 f(b+wi xi)
5.输出层：生成最终的输出结果

"""

#2. 多个神经元搭建神经网络
"""

输入层——>隐藏层1——>隐藏层2——>输出层
输入层：接收输入数据  有几个特征就有几个输入神经元
隐藏层：由多个神经元组成，每个神经元都有自己的权重和偏置参数，负责处理和转换输入数据
输出层：生成最终的输出结果，输出神经元的数量取决于具体的任务，例如分类任务中输出神经元的数量等于类别的数量

"""

import torch
import torch.nn as nn
#1.定义一个类，继承nn.Module

class ModelDemo(nn.Module):

    #1.1在__init__()方法中,完成初始化，搭建神经网络
    def __init__(self):
        super().__init__()
    #1.2搭建神经网络
    #输入层——>隐藏层1——>隐藏层2——>输出层

    #隐藏层1：输入特征数量为3，输出特征数量为3，使用sigmoid激活函数
        self.hidden1 = nn.Sequential(
            nn.Linear(3,3), #线性层
            nn.Sigmoid() #激活函数
        )
    #隐藏层2：输入特征数量为3，输出特征数量为2，使用relu激活函数
        self.hidden2 = nn.Sequential(
            nn.Linear(3,2), #线性层
            nn.ReLU() #激活函数
        )
    #输出层：输入特征数量为2，输出特征数量为2
        self.output = nn.Linear(2,2) #线性层


    #1.3参数初始化，第一个隐藏层xavier初始化，第二个隐藏层he初始化
        nn.init.xavier_uniform_(self.hidden1[0].weight) #xavier初始化
        nn.init.zeros_(self.hidden1[0].bias) #偏置初始化为0
        nn.init.kaiming_uniform_(self.hidden2[0].weight, nonlinearity='relu') #he初始化
        nn.init.zeros_(self.hidden2[0].bias) #偏置初始化为0
       
#todo:1.前向传播
    def forward(self,x):
        x = self.hidden1(x) #输入数据经过隐藏层1,加权求和并且经过激活函数
        x = self.hidden2(x) #输入数据经过隐藏层2,加权求和并且经过激活函数
   #输入数据经过输出层,加权求和+softmax函数
        x = self.output(x)
        x = torch.softmax(x, dim=1) #对输出层进行softmax激活函数 #dim=1表示对每行进行softmax计算,dim=0表示对每列进行softmax计算
        return x
    

#todo:2.模型训练
def train():
    #1.创建模型实例
    model = ModelDemo() 

    #2.调用神经网络
    input=torch.randn(size=(5,3)) #输入数据
    output = model(input) #模型预测输出结果
   
    #3.计算和查看模型参数
    print("========计算模型参数========") 
    from torchsummary import summary
    summary(model, input_size=(5,3)) #输入数据的形状为(5,3)
    print("========查看模型参数========")
    for name, param in model.named_parameters():
        print(name, param.data) #打印参数名称和参数值
        
#todo3.模型预测
if __name__ == '__main__':
    train()
