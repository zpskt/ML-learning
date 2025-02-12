import torch
import numpy as np


# 创建张量示例
def create_tensor():
    # 创建一个张量
    x = torch.tensor([1, 2, 3])
    print(x)

    # 创建一个张量
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(x)

    # 创建一个张量:创建一个三维张量，包含两个二维矩阵
    x = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    print(x)
    #     创建全0矩阵 创建一个形状为2x3的全0矩阵
    x = torch.zeros(2, 3)
    print(x)
    #     创建一个形状为2x3的全1矩阵
    x = torch.ones(2, 3)
    print(x)
    # 从数据构造张量
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(x)
    x = torch.randn(3, 2)
    print(x)
    print(x.shape)


#     numpy和tensor转换示例
def numpy_tensor_convert():
    x = np.array([[1, 2, 3], [4, 5, 6]])
    print(x)
    # numpy 转换为张量
    x = torch.from_numpy(x)
    print(x)
    # 张量转换为numpy
    x = x.numpy()
    print(x)


if __name__ == '__main__':
    numpy_tensor_convert()
