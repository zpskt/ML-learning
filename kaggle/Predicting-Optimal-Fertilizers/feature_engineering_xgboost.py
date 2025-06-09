import os
from collections import Counter

import pandas as pd
from category_encoders import TargetEncoder as ce_TargetEncoder, target_encoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, TargetEncoder
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from xgboost import XGBClassifier
from data_preprocessing import save_processed_data

def add_agricultural_features(df):
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

    return df


def main():
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')

    print("🚀 开始特征工程与模型训练流程")

    # -----------------------------
    # 1. 加载数据
    # -----------------------------
    file_path = os.path.join(data_dir, 'train.csv')
    df = pd.read_csv(file_path)

    # -----------------------------
    # 2. 查看类别分布
    # -----------------------------
    print("\n📊 标签分布:")
    print(Counter(df['Fertilizer Name']))

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
    le = LabelEncoder()
    y = le.fit_transform(y)
    print("✅ 目标变量已编码为数值类型:", dict(enumerate(le.classes_)))

    # 类别型列和数值型列
    categorical_cols = ['Soil Type', 'Crop Type']
    numerical_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
    # 所有特征列中筛选出“既不是类别型也不是数值型”的列， 即构造的农业领域新特征（如 NPK 比例、环境因子等）
    '''
    这些农业特征为什么要单独处理？
        因为：
        它们已经经过人工构造，可能具有更强的非线性关系；
        可能需要与原始数值特征一起标准化；
        不属于原始的类别或数值特征，所以要单独提取出来用于后续预处理。
    '''
    agri_feature_cols = [col for col in X.columns if col not in categorical_cols + numerical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', TargetEncoder(), categorical_cols),
            ('num', StandardScaler(), numerical_cols + agri_feature_cols)
        ],
        remainder='passthrough'
    )

    X_processed = preprocessor.fit_transform(X, y)
    print("✅ 数据已完成预处理")
    # 将编码后的 y 添加为新列
    feature_names = preprocessor.get_feature_names_out()
    # 转换为 pandas DataFrame
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

    # ✅ 添加目标变量
    # X_processed_df['Encoded_Label'] = y  # 编码后的 y（int 类型）

    # 如果你还想添加原始的 Fertilizer Name（字符串类型）也可以加上
    df_original = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    original_labels = df_original['Fertilizer Name'].values
    X_processed_df['Fertilizer Name'] = original_labels  # 原始字符串标签（便于人工理解）

    # ✅ 把当前处理后的X和y存入一个新的csv中
    save_processed_data(X_processed_df, 'processed_train.csv')
    print("✅ 已保存处理后的数据至 processed_train.csv")

    # 划分训练集/验证集
    X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)

    # -----------------------------
    # 5. 模型训练：XGBoost
    # -----------------------------
    print("\n🌳 正在训练 XGBoost 模型...")

    model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.7,
        colsample_bytree=0.6,
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

    # -----------------------------
    # 6. 模型评估
    # -----------------------------
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    print(f"\n✅ 验证集准确率: {acc:.4f}")
    print("\n📋 分类报告:")
    print(classification_report(y_val, y_pred))

    # -----------------------------
    # 7. 保存模型和预处理参数（可选）
    # -----------------------------
    import joblib

    # 保存模型
    joblib.dump(model, os.path.join(current_dir, 'xgb_model.pkl'))

    # 保存编码器和标准化器（用于测试集预测）
    # 提取并保存 TargetEncoder
    target_encoder = preprocessor.named_transformers_['cat']
    joblib.dump(target_encoder, os.path.join(current_dir, 'target_encoder.pkl'))

    # 提取并保存 StandardScaler
    scaler = preprocessor.named_transformers_['num']
    joblib.dump(scaler, os.path.join(current_dir, 'scaler.pkl'))
    # ✅ 新增：保存 LabelEncoder
    joblib.dump(le, os.path.join(current_dir, 'label_encoder.pkl'))

    print("\n💾 模型及预处理器已保存")


if __name__ == "__main__":
    main()
