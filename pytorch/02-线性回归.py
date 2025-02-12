import torch
import torch as t
from matplotlib import pyplot as plt
from torch import nn

# 设置PyTorch的随机数生成器种子为1000，以此确保每次随机操作产生一样的结果，从而得到一致的实验结果
t.manual_seed(1000)


# 生成线性回归的随机模拟数据
def get_fake_data(batch_size=100):
    x = t.rand(batch_size, 1) * 20
    # y=2x+3+(随机噪声)
    y = x * 2 + (1 + t.randn(batch_size, 1)) * 3

    return x, y


def main():
    x, y = get_fake_data()
    '''
    第一个参数 1：输入特征的数量。这里表示模型的输入是一个单一特征（即一维数据）。
    第二个参数 1：输出特征的数量。这里表示模型的输出也是一个单一特征（即一维数据）。
    '''
    model = nn.Linear(1, 1)
    # 损失函数
    criterion = nn.MSELoss()
    #     梯度下降
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    # for循环 迭代
    for epoch in range(1000):
        # 前向传播
        y_pred = model(x)
        # 损失函数
        loss = criterion(y_pred, y)
        # 将模型参数的梯度清零，防止梯度累积
        optimizer.zero_grad()
        # 求解梯度
        loss.backward()
        # 将模型参数的梯度清零，防止梯度累积
        optimizer.step()
        # 打印损失
        if epoch % 100 == 0:
            print('epoch:', epoch, 'loss:', loss.item())
    print("训练完成")
    #     展示效果
    '''
    张量 x 和 y 转换为 NumPy 数组并绘制散点图。具体步骤如下：
    使用 squeeze 方法去除维度为 1 的维度。
    使用 numpy 方法将张量转换为 NumPy 数组。
    使用 plt.scatter 方法绘制散点图。
    '''
    plt.scatter(x.squeeze().numpy(), y.squeeze().numpy())
    plt.plot(x.numpy(), model(x).detach().numpy())
    plt.show()


if __name__ == '__main__':
    main()
