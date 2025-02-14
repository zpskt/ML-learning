
import pandas as pd
import torch
from torch import nn

# 检查是否有可用的GPU，并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # 读取数据
    data = pd.read_csv('dataset/creditcardfraud.csv')
    print(data)
    # 删除数据框中所有包含缺失值的行，确保数据完整
    data = data.dropna()
    # 首先打印数据的前几行，以便查看数据结构
    print(data.head())
    # 将处理后的数据框转换为 NumPy 数组
    data_df = pd.DataFrame(data)
    # 从数据集中提取特征变量
    X = data_df.iloc[:5000, :-1]
    # 从数据集中提取目标变量  提取最后一列，并将其中的值 0 替换为 0，值 1 替换为 1
    Y = data_df.iloc[:5000, -1].replace({0: 0, 1: 1})
    #     数据预处理 转换为tensor,数据类型要转换一下
    X = torch.tensor(X.values, dtype=torch.float32).to(device)
    print("X.shape is {} \n", X.shape)
    #     现在Y的格式还是一维数据，需要将其转换为二维数据
    Y = torch.tensor(Y.values.reshape(-1, 1), dtype=torch.float32).to(device)

    '''
    定义了一个简单的神经网络模型，使用PyTorch的nn.Sequential构建。
    模型包含一个线性层（输入维度为30，输出维度为1）和一个Sigmoid激活函数。该模型用于将30维输入映射到一个0到1之间的概率值。
    '''
    model = nn.Sequential(
        nn.Linear(30, 1),
        nn.Sigmoid()
    ).to(device)
    print(model)
    #     定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #     将数据分批次进行训练
    batches = 16
    print(len(X))
    no_of_batches = int(len(X) / batches)
    print("no_of_batches is ", no_of_batches)
    epoches = 50
    for epoch in range(epoches):
        for i in range(no_of_batches):
            start = i * batches
            end = start + batches
            x = X[start:end]
            y = Y[start:end]
            y_pred = model(x)
            loss = criterion(y_pred, y)
            # 先清空梯度,避免之前计算的梯度值对本次计算产生影响
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("epoch:{} loss:{}".format(epoch, loss))

    print(model.state_dict())