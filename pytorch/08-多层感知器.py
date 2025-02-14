import numpy as np

import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

'''
数据预处理
返回tensor的特征值和目标值
'''


def data_preprocess():
    # 这里我是用的数据集都是散列分布的，没有去专门找数据或者处理数据，凑合做演示就行了
    data = pd.read_csv('dataset/creditcardfraud.csv')
    # 由于我本机内存比较小，所以只取前1000条数据
    # data = data[:1000]
    data = data.sample(n=1000, random_state=150)  # random_state参数用于设置随机种子，确保结果可重复
    print(data.head())
    print(data.info())
    # 查看Class列有多少个不同数值，即做去重处理，
    print(data.Class.unique())
    print(data.Amount.unique())
    # 根据'Class'和'Time'列对数据进行分组
    print(data.groupby(['Class', 'Time']).size())
    # 这一步的目的是将分类数据转换为模型可以处理的格式,
    # 比如如果是性别Gender列（male、female）转换为虚拟变量，
    # 我们想要实现的效果就是加两个列，male、female,然后里面的值分别为0 1 或者true false，用以区分
    # 相当于你有一个X列，X列中有N个不同的值， 那么执行此操作后就会变成N个不同的列，里面的值分别为0 1 或者true false，用以区分
    dummies = pd.get_dummies(data.Time)
    print(dummies)
    #     将新增的列插入到data中
    data = pd.concat([data, dummies], axis=1)
    #     删除旧的列
    data = data.drop(['Time'], axis=1)
    print(data)
    # 统计某个字段不同值的数据占比，可以用来判断是否均匀，分析数据
    print(data.V1.value_counts())
    #
    Y = data.Class
    print(Y.dtype)
    #     转成torch
    Y = torch.from_numpy(Y.values).type(torch.FloatTensor)
    # 从数据框 data 的列名中筛选出除 Class 列之外的所有列名，并生成一个包含这些列名的列表。
    # 实际的训练数据中，列是在太多了
    X = data[[c for c in data.columns if c != 'Class']].values.astype(np.float32)
    X = torch.from_numpy(X).type(torch.FloatTensor)
    print(X.shape)
    print(Y.shape)
    return X, Y


class Model(nn.Module):
    '''
    初始化所有层
    '''

    def __init__(self):
        super(Model, self).__init__()
        # 初始化线性层和激活函数
        self.linear1 = nn.Linear(1026, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 1)

    '''
    定义模型的运算过程
    '''

    def forward(self, input):
        # 检查输入是否为张量
        if not isinstance(input, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")

        # 检查输入维度是否匹配
        if input.dim() != 2 or input.size(1) != self.linear1.in_features:
            raise ValueError(f"Input tensor must have shape (batch_size, {self.linear1.in_features})")

        # 前向传播
        # 使用relu激活函数
        out = F.relu(self.linear1(input))
        out = F.relu(self.linear2(out))
        # 因为是逻辑回归模型，所以最后一层要用sigmoid
        out = F.sigmoid(self.linear3(out))
        return out


def get_model():
    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return model, optimizer


if __name__ == '__main__':
    X, Y = data_preprocess()
    model, optimizer = get_model()
    #     定义损失函数
    criterion = nn.BCELoss()
    # 设置批量大小
    batch_size = 64
    batches = 16
    no_of_batches = int(len(X) / batches)
    epoches = 50
    for epoch in range(epoches):
        for i in range(no_of_batches):
            start = i * batches
            end = start + batches
            x = X[start:end]
            y = Y[start:end]
            y_pred = model(x)
            print(y_pred.shape)
            print(y.shape)
            loss = criterion(y_pred, y)
            # 先清空梯度,避免之前计算的梯度值对本次计算产生影响
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("epoch:{} loss:{}".format(epoch, loss))
    print(criterion(Model(X), Y))
