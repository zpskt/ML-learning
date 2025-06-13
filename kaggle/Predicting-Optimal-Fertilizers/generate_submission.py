import os

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import train_model_v1
def generate_submission(file_name='test.csv', output_file='submission.csv', top_k=3):
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
    file_path = os.path.join(data_dir, file_name)
    df = pd.read_csv(file_path)

    # 提取 id 列
    ids = df['id'].values

    # -----------------------------
    # 3. 构造农业领域特征
    # -----------------------------
    # 应用特征构造
    df = train_model_v1.add_agricultural_features(df)

    # -----------------------------
    # 4. 特征编码处理
    # -----------------------------
    # 特征和标签
    X = df.drop(columns=[ 'id'])

    # 📌 假设你已经保存了 preprocessor：
    import joblib
    preprocessor = joblib.load(os.path.join(current_dir, 'scaler.pkl'))  # 示例路径，请替换为你实际保存的 ColumnTransformer

    # 应用变换（不进行 fit）
    X_processed = preprocessor.transform(X)

    """
    对 test.csv 中的每个样本预测 Top-K 推荐，并保存为 submission.csv
    """
    print("📄 开始生成提交文件...")
    # 加载模型json文件
    model = XGBClassifier()
    model.load_model("xgboost_model.json")

    probas = model.predict_proba(X_processed)


    # 获取 Top-K 类别索引
    top_k_indices = np.argsort(probas, axis=1)[:, -top_k:]
    # 加载 LabelEncoder
    le = joblib.load(os.path.join(current_dir, 'label_encoder.pkl'))
    # 转换为原始类别名称（假设 le 已从训练阶段加载）
    predicted_labels = np.array([le.inverse_transform(row) for row in top_k_indices])

    # 生成提交格式字符串：多个肥料名称用空格分隔
    predicted_strings = [" ".join(row) for row in predicted_labels]

    # 创建提交 DataFrame
    submission_df = pd.DataFrame({
        'id': ids,
        'Fertilizer Name': predicted_strings
    })

    # 保存为 CSV 文件
    submission_df.to_csv(output_file, index=False)
    print(f"✅ 提交文件已保存至 {output_file}")

if __name__ == '__main__':
    generate_submission()