import torch


def main():
    x = torch.ones(2, 3)
    #     如果你想保存梯度，你需要设置 requires_grad=True
    x.requires_grad = True
    # 或者这么写
    x = torch.ones(2, 3, requires_grad=True)
    # 计算之前的数据及其梯度
    print(x.data)
    # 初始为None，表示张量x的梯度，形状和x相同
    print(x.grad)
    # 初始为None，表示张量x的梯度函数，它是一个指向生成 x 的操作的函数对象，用于在反向传播时计算梯度。
    # 如果 x 是通过某些操作生成的，则 x.grad_fn 会指向该操作的梯度计算函数
    print(x.grad_fn)
    # 开始进行计算
    y = (x + 2) * (x + 5) * x
    # y = x*x
    print(y)
    # 这里输出None，因为我们设定的时候没有设置y保存梯度
    print(y.grad)
    # 这里有值，是因为y是通过某些操作生成的，并且这些操作涉及到我们曾经设置过需要保存梯度的张量x。
    # 所以y.grad_fn指向 张量x的梯度计算函数。它记录了y的操作信息
    print(y.grad_fn)

    # 反向传播 这一步的数据操作就是微分求导，就像数学中 d y / d x
    # 注意 这里反向传播中智能是根据标量反向传播，所以还需要对y进行标量化处理
    y = y.mean()
    print(y)
    y.backward()
    # 求导以后函数变为 y = f(x) = 3x**2+14x+10 ,带入x为1后，y=f(x)=27,
    # 因为y还做了mean操作，所以x的所有梯度变成了 27/6 = 4.5
    print("y=f(x) 这个函数进行求导以后 {} \n", x.grad)

    #     如果你的某个计算不想保留x的梯度
    with torch.no_grad():
        print((x*x).requires_grad)
    # 创建一个与 x 具有相同数据的新张量 y，但 y 不会保存任何梯度信息
    y = x.detach()
    print(y)
if __name__ == '__main__':
    main()
