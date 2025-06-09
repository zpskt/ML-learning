# 数据集分析
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# 加载训练数据集
df_train = pd.read_csv('data/train.csv')
# df_train = pd.read_csv('data/processed_train.csv')

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


def plot_categorical_histograms():
    # 提取类别型列
    categorical_cols = df_train.select_dtypes(include=['object']).columns.tolist()
    soil_types = df_train['Soil Type'].unique()
    crop_types = df_train['Crop Type'].unique()
    print("soil_types:", soil_types)
    print("crop_types:", crop_types)

    # 设置图形布局参数
    ncols = 2
    nrows = (len(categorical_cols) + 1) // 2

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10))
    print("开始绘制类别型列分布")

    # 遍历所有类别型列并绘图
    for i, col in enumerate(categorical_cols):
        try:
            row, col_idx = divmod(i, ncols)  # 修复：使用 ncols 而不是 col
            sns.countplot(data=df_train, x=col, ax=axes[row, col_idx])
            axes[row, col_idx].set_title(f'Count Plot of {col}')
            axes[row, col_idx].tick_params(axis='x', rotation=45)  # 防止标签重叠
            print("{col}",col)
        except Exception as e:
            print(f"绘制列 {col} 出错: {e}")

    # 隐藏多余的子图
    for j in range(i + 1, nrows * ncols):
        row, col_idx = divmod(j, ncols)
        fig.delaxes(axes[row, col_idx])

    plt.tight_layout()
    plt.show()
    plt.close()  # 关闭图像资源，避免内存占用过高


def feature_distribution_by_target():
    '''
    数值类型数据target占比
    :return:
    '''
    # 筛选数值型列（排除 object 类型）
    numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()

    # 明确排除 'id' 和目标列（如果有）
    if 'id' in numeric_cols:
        numeric_cols.remove('id')
    # 排除可能包含的目标列本身（如果有）
    if 'Fertilizer Name' in numeric_cols:
        numeric_cols.remove('Fertilizer Name')

    # 按目标变量分组并计算均值
    grouped = df_train.groupby('Fertilizer Name')[numeric_cols].mean()

    # 打印表格形式的统计结果
    print("\n不同 Fertilizer Name 下各数值特征的平均值：")
    print(grouped)

    # 绘制热力图（Heatmap），展示不同类别下特征均值的差异
    plt.figure(figsize=(14, 10))
    sns.heatmap(grouped.T, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('平均值 Heatmap')
    plt.tight_layout()
    plt.show()
    plt.close()

    # 可选：绘制每个特征的柱状图，显示不同类别的均值
    for col in numeric_cols:
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Fertilizer Name', y=col, data=df_train, estimator=np.mean, ci=None)
        plt.xticks(rotation=45)
        plt.title(f'Mean {col} by Fertilizer Name')
        plt.tight_layout()
        plt.show()
        plt.close()
    # 查看object类型的数据

def categorical_distribution_by_target():
    '''
    查看数据类型的数据分布
    :return:
    '''
    # 提取类别型列（排除 'id' 和目标列）
    categorical_cols = df_train.select_dtypes(include=['object']).columns.tolist()
    if 'id' in categorical_cols:
        categorical_cols.remove('id')
    if 'Fertilizer Name' in categorical_cols:
        categorical_cols.remove('Fertilizer Name')

    target_col = 'Fertilizer Name'

    for col in categorical_cols:
        # 创建交叉表
        cross_tab = pd.crosstab(df_train[target_col], df_train[col], normalize='index')

        # 绘制堆叠柱状图
        cross_tab.plot(kind='bar', stacked=True, figsize=(12, 6))
        plt.title(f'Distribution of {col} by Fertilizer Name (Normalized)')
        plt.xlabel('Fertilizer Name')
        plt.ylabel(f'Relative Frequency of {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.legend(title=col)
        plt.show()
        plt.close()



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
    plot_categorical_histograms()
    # feature_distribution_by_target()
    # categorical_distribution_by_target()