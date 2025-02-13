import torch
import numpy as np


def tensor_compute():
    x = np.array([[1, 2, 3], [4, 5, 6]])
    # numpy 转换为张量
    x1 = torch.from_numpy(x)
    print(x1)
    x2 = torch.ones(2, 3)
    print(x2)
    # 注意加和的时候需要保证两个张量维度相同
    print(x1 + x2)
    print((x1 + x2).shape)
    print(x1 * x2)
    print((x1 * x2).shape)
    # 将张量 x1 的形状从 (2, 3) 转换为 (3, 2)
    print(x1.view(3, 2))


if __name__ == '__main__':
    tensor_compute()
