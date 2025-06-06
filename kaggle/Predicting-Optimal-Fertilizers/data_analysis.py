# 数据集分析
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# 加载训练数据集
df_train = pd.read_csv('data/train.csv')

def basic_info():
    '''
    数据集基础分析
    :return:
    '''
    # 查看前几行数据
    print("--查看前几行数据 --")
    print(df_train.head())

    # 查看数据基本信息（缺失值、数据类型等）
    print("--查看数据基本信息 --")
    print(df_train.info())

    # 统计描述（针对数值型字段）
    print("--统计描述 --")
    # 显示完整的描述统计结果，不进行截断
    pd.set_option('display.max_columns', None)
    print(df_train.describe())

    # 恢复列显示限制（可选）
    pd.reset_option('display.max_columns')

def plot_numeric_histograms():
    try:
        # 获取数值型列名列表
        numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()

        # 设置图形布局参数
        ncols = 2
        nrows = (len(numeric_cols) + 1) // 2

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10))
        print("开始绘制直方图")

        for i, col in enumerate(numeric_cols):
            try:
                row, col_idx = divmod(i, ncols)
                sns.histplot(df_train[col], ax=axes[row, col_idx], kde=True)
            except Exception as e:
                print(f"绘制列 {col} 出错: {e}")

        # 隐藏多余的子图
        for j in range(i + 1, nrows * ncols):
            row, col_idx = divmod(j, ncols)
            fig.delaxes(axes[row, col_idx])

        plt.tight_layout()
        plt.show()  # 显示图像
        plt.close()  # 关闭图像资源，避免内存占用过高
    except Exception as e:
        print(f"绘制直方图出错: {e}")

def object_column_analysis():
    # 提取类别型列
    categorical_cols = df_train.select_dtypes(include=['object']).columns.tolist()

    # 绘制条形图
    for col in categorical_cols:
        # 查看其种类及其数量
        print(df_train[col].value_counts())

def target_column_analysis():
    # 查看肥料种类及其数量
    print(df_train['Fertilizer Name'].value_counts())

    # 可视化目标变量分布
    sns.countplot(y='Fertilizer Name', data=df_train, order=df_train['Fertilizer Name'].value_counts().index)
    plt.title('Distribution of Fertilizer Names')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # basic_info()
    # plot_numeric_histograms()
    object_column_analysis()