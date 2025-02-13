import torch
from matplotlib import pyplot as plt

# 设置PyTorch的随机数生成器种子为1000，以此确保每次随机操作产生一样的结果，从而得到一致的实验结果
torch.manual_seed(1000)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"


# 生成线性回归的随机模拟数据
def get_fake_data(batch_size=100):
    x = torch.rand(batch_size, 1) * 20
    # y=2x+3+(随机噪声)
    y = x * 2 + (1 + torch.randn(batch_size, 1)) * 3

    return x, y


def main():
    x, y = get_fake_data()
    # 手动写权重和偏置
    x.requires_grad = True
    w = torch.randn(1, 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    print(w.shape)
    # 线性回归的模型公式为 f（x）=wx+b
    for epoch in range(5):
        for xi, yi in zip(x, y):
            # y_pred = w * x + b 只能用于一维张量， 这个方法可以多维度
            # 假设 x 是 0D 张量，将其扩展为 1D 张量
            y_pred = torch.matmul(xi, w) + b
            #  计算损失
            loss = (yi - y_pred).pow(2).mean()
            if not w.grad is None:
                w.grad.data.zero_()
            if not b.grad is None:
                b.grad.data.zero_()
            # 反向传播，对其求导
            loss.backward()
            # 更新参数 0.001是学习率，学习率不要太高了，太高了会导致梯度消失，里面的值变成nan
            with torch.no_grad():
                w.data -= 0.001 * w.grad.data
                b.data -= 0.001 * b.grad.data
            # 清空梯度
            w.grad.data.zero_()
            b.grad.data.zero_()
    # 该错误是由于尝试将位于GPU（cuda:0）上的PyTorch张量直接转换为NumPy数组引起的。
    # NumPy不支持直接处理GPU上的数据，因此需要先将张量从GPU移动到CPU再进行转换。
    print(w)
    print(b)
    plt.scatter(x.detach().numpy(), y.detach().numpy())
    plt.plot(x.detach().numpy(), w.detach().numpy() * x.detach().numpy() + b.detach().numpy(), c='r')
    plt.show()


if __name__ == '__main__':
    main()
