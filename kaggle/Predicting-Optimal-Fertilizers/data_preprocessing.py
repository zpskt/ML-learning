import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, 'data')

def encode_object_columns(df):
    """
    对DataFrame中所有object类型的列进行Label Encoding
    :param df: 输入的DataFrame
    :return: 编码后的DataFrame
    """
    le = LabelEncoder()
    object_cols = df.select_dtypes(include=['object']).columns.tolist()

    for col in object_cols:
        df[col] = le.fit_transform(df[col].astype(str))  # 转为字符串防止异常值报错

    print("✅ 完成对object类型字段的编码")
    return df


def split_train_val(df, target_col, test_size=0.2, random_state=42):
    """
    将数据集切分为训练集和验证集
    :param df: 原始DataFrame
    :param target_col: 目标列名
    :param test_size: 验证集比例
    :param random_state: 随机种子
    :return: X_train, X_val, y_train, y_val
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    print(f"✅ 数据已切分为训练集({len(X_train)})和验证集({len(X_val)})")
    return X_train, X_val, y_train, y_val


def save_processed_data(df, filename='processed_train.csv'):
    """
    保存处理后的DataFrame到CSV文件
    :param df: 处理后的DataFrame
    :param filename: 输出文件名
    """
    output_path = os.path.join(data_dir, filename)
    df.to_csv(output_path, index=False)
    print(f"✅ 已保存处理后的数据至 {output_path}")


def preprocess_and_save(input_file='train.csv', target_col='Fertilizer Name'):
    """
    主函数：完整的数据预处理流程
    """
    # 读取原始数据
    file_path = os.path.join(data_dir, input_file)
    df = pd.read_csv(file_path)
    # 删除无用列，如 id
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    print(f"✅ 已加载数据 {file_path}")

    # 编码object类型字段
    df_encoded = encode_object_columns(df)

    # 保存编码后的数据
    save_processed_data(df_encoded, 'processed_train.csv')

    # 切分训练集和验证集（可选）
    X_train, X_val, y_train, y_val = split_train_val(df_encoded, target_col)

    return X_train, X_val, y_train, y_val


if __name__ == '__main__':
    X_train, X_val, y_train, y_val = preprocess_and_save()
