import os
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, TargetEncoder
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from xgboost import XGBClassifier
import joblib
from data_preprocessing import save_processed_data

le = LabelEncoder()

def add_agricultural_features(df):
    '''
    构造农业领域特征
    :param df:
    :return:
    '''
    # NPK 总和
    df['NPK_Sum'] = df['Nitrogen'] + df['Phosphorous'] + df['Potassium']

    # N/P 比例
    df['N_P_Ratio'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-5)

    # P/K 比例
    df['P_K_Ratio'] = df['Phosphorous'] / (df['Potassium'] + 1e-5)

    # 环境适宜性指标（温度 × 湿度 × 水分）
    df['Env_Index'] = df['Temparature'] * df['Humidity'] * df['Moisture']

    # 肥力综合评分（假设 N=0.3, P=0.3, K=0.4 权重）
    df['Fertility_Score'] = (
            df['Nitrogen'] * 0.3 +
            df['Phosphorous'] * 0.3 +
            df['Potassium'] * 0.4
    )

    # 增加作物对营养元素的偏好特征（示例）
    crop_n_preference = {
        'Wheat': 0.8,
        'Maize': 0.7,
        'Oil seeds': 0.3,
        'Paddy': 0.5,
        'Cotton': 0.6,
        'Barley': 0.7,
        'Millets': 0.5,
        'Sugarcane': 0.4,
        'Ground Nuts': 0.4,
        'Tobacco': 0.5,
        'Pulses': 0.4
    }

    df['Crop_Nitrogen_Preference'] = df['Crop Type'].map(crop_n_preference).fillna(0.5)
    df['Weighted_N'] = df['Nitrogen'] * df['Crop_Nitrogen_Preference']
    df['N_sqrt'] = np.sqrt(df['Nitrogen'])
    df['NK_ratio'] = df['Nitrogen'] / (df['Potassium'] + 1e-5)
    return df


def load_data_and_preprocess(data='train.csv'):
    '''
    加载数据并进行预处理
    :return:
    '''
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')

    print("开始加载数据并处理")

    # -----------------------------
    # 1. 加载数据
    # -----------------------------
    file_path = os.path.join(data_dir, data)
    df = pd.read_csv(file_path)

    # -----------------------------
    # 3. 构造农业领域特征
    # -----------------------------
    # 应用特征构造
    df = add_agricultural_features(df)

    # -----------------------------
    # 4. 特征编码处理
    # -----------------------------
    # 特征和标签
    X = df.drop(columns=['Fertilizer Name', 'id'])
    y = df['Fertilizer Name'].values

    # ✅ 添加这一段来对 y 进行编码
    y = le.fit_transform(y)
    print("✅ 目标变量已编码为数值类型:", dict(enumerate(le.classes_)))

    # 保存 LabelEncoder
    joblib.dump(le, os.path.join(current_dir, 'label_encoder.pkl'))
    print("✅ LabelEncoder 已保存")

    # 类别型列和数值型列
    categorical_cols = ['Soil Type', 'Crop Type']
    numerical_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
    # 所有特征列中筛选出“既不是类别型也不是数值型”的列， 即构造的农业领域新特征（如 NPK 比例、环境因子等）

    agri_feature_cols = [col for col in X.columns if col not in categorical_cols + numerical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', TargetEncoder(), categorical_cols),
            ('num', StandardScaler(), numerical_cols + agri_feature_cols)
        ],
        remainder='passthrough'
    )
    X_processed = preprocessor.fit_transform(X, y)

    # ✅ 保存已经 fit 好的 preprocessor
    joblib.dump(preprocessor, os.path.join(current_dir, 'scaler.pkl'))
    print("✅ ColumnTransformer 已保存为 scaler.pkl")

    # 将编码后的 y 添加为新列
    feature_names = preprocessor.get_feature_names_out()
    # 转换为 pandas DataFrame
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

    # 如果你还想添加原始的 Fertilizer Name（字符串类型）也可以加上
    df_original = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    original_labels = df_original['Fertilizer Name'].values
    X_processed_df['Fertilizer Name'] = original_labels  # 原始字符串标签（便于人工理解）

    # ✅ 把当前处理后的X和y存入一个新的csv中
    save_processed_data(X_processed_df, 'processed_train.csv')
    print("✅ 已保存处理后的数据至 processed_train.csv")

    # 划分训练集/验证集
    X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)
    print("✅ 数据处理完毕")
    return X_train, X_val, y_train, y_val



def main():
    print("🚀 启动特征工程与模型训练流程")
    X_train, X_val, y_train, y_val = load_data_and_preprocess()

    # -----------------------------
    # 5.1 模型训练：XGBoost
    # -----------------------------
    print("\n🌳 正在训练 XGBoost 模型...")

    model = XGBClassifier(
        n_estimators=3000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.7,
        colsample_bytree=0.7,
        eval_metric='mlogloss',
        early_stopping_rounds=20,
        use_label_encoder=False,
        verbosity=0,
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    print("\n🌳 训练 XGBoost 模型完毕")
    # 保存模型
    model.save_model('xgboost_model.json')

    # -----------------------------
    # 单一结果模型评估
    # -----------------------------
    print("\n📋 单一结果模型评估中...")
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    print(f"\n✅ 验证集准确率: {acc:.4f}")
    print("\n📋 分类报告:")
    print(classification_report(y_val, y_pred))


def val():
    # 获取当前脚本所在目录

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    print("")

    # -----------------------------
    # 1. 加载数据
    # -----------------------------
    file_path = os.path.join(data_dir, 'train.csv')
    df = pd.read_csv(file_path)
    # -----------------------------
    # 3. 构造农业领域特征
    # -----------------------------
    # 应用特征构造
    df = add_agricultural_features(df)

    # -----------------------------
    # 4. 特征编码处理
    # -----------------------------
    # 特征和标签
    X = df.drop(columns=['Fertilizer Name', 'id'])

    y = df['Fertilizer Name'].values
    # ✅ 添加这一段来对 y 进行编码
    # 加载 LabelEncoder
    le = joblib.load(os.path.join(current_dir, 'label_encoder.pkl'))
    y = le.fit_transform(y)
    print("✅ 目标变量已编码为数值类型:", dict(enumerate(le.classes_)))

    # 📌 假设你已经保存了 preprocessor：

    preprocessor = joblib.load(os.path.join(current_dir, 'scaler.pkl'))  # 示例路径，请替换为你实际保存的 ColumnTransformer
    # 应用变换（不进行 fit）
    X_processed = preprocessor.transform(X)

    # 加载模型json文件
    model = XGBClassifier()
    model.load_model("xgboost_model.json")

    print("\n📋 多结果模型评估中...")
    y_proba = model.predict_proba(X_processed)
    # 获取 top-3 的预测类别索引
    top_3_indices = np.argsort(y_proba, axis=1)[:, -3:]
    y_val_flat = y.ravel()
    # 判断真实标签是否在 top-3 预测中
    top_3_correct = np.sum([y_val_flat[i] in top_3_indices[i] for i in range(len(y_val_flat))])
    # 计算 Top-3 准确率
    top_3_accuracy = top_3_correct / len(y_val_flat)
    print(f"\n✅ 验证集 Top-3 准确率: {top_3_accuracy:.4f}")

if __name__ == "__main__":
    main()
    # val()
