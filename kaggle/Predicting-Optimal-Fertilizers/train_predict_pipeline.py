# train_predict_pipeline.py

import os

import joblib
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from xgboost import XGBClassifier

from data_preprocessing import save_processed_data

# -----------------------------
# 配置路径和全局变量
# -----------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, 'data')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed_train.csv')
LABEL_ENCODER_PATH = os.path.join(CURRENT_DIR, 'label_encoder.pkl')
SCALER_PATH = os.path.join(CURRENT_DIR, 'scaler.pkl')
XGBOOST_MODEL_PATH = os.path.join(CURRENT_DIR, 'xgboost_model.json')


# -----------------------------
# 特征构造
# -----------------------------
def add_agricultural_features(df):
    """
    构造农业领域相关特征
    """
    df['NPK_Sum'] = df['Nitrogen'] + df['Phosphorous'] + df['Potassium']
    df['N_P_Ratio'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-5)
    df['P_K_Ratio'] = df['Phosphorous'] / (df['Potassium'] + 1e-5)
    df['Env_Index'] = df['Temparature'] * df['Humidity'] * df['Moisture']
    df['Fertility_Score'] = (
            df['Nitrogen'] * 0.3 +
            df['Phosphorous'] * 0.3 +
            df['Potassium'] * 0.4
    )
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


# -----------------------------
# 数据预处理与编码
# -----------------------------
def preprocess_data(df, is_train=True):
    """
    对数据进行预处理（仅用于训练）
    :param df: 原始 DataFrame
    :param is_train: 是否是训练数据
    :return: 处理后的特征矩阵 X 和标签 y
    """
    df = add_agricultural_features(df)

    if is_train:
        X = df.drop(columns=['Fertilizer Name', 'id'])
        y = df['Fertilizer Name'].values
    else:
        X = df.drop(columns=['id'])
        y = None

    return X, y


def encode_and_transform(X, y=None, fit=False):
    """
    使用 ColumnTransformer 编码并标准化特征
    """
    categorical_cols = ['Soil Type', 'Crop Type']
    numerical_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
    agri_feature_cols = [col for col in X.columns if col not in categorical_cols + numerical_cols]

    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import TargetEncoder, StandardScaler

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', TargetEncoder(), categorical_cols),
            ('num', StandardScaler(), numerical_cols + agri_feature_cols)
        ],
        remainder='passthrough'
    )

    if fit:
        X_processed = preprocessor.fit_transform(X, y)
        joblib.dump(preprocessor, SCALER_PATH)
    else:
        preprocessor = joblib.load(SCALER_PATH)
        X_processed = preprocessor.transform(X)

    return X_processed


# -----------------------------
# 模型训练与评估
# -----------------------------
class ModelTrainer:
    def __init__(self, model_type='xgboost', params=None):
        self.model_type = model_type
        self.params = params or {}

    def get_model(self):
        if self.model_type == 'xgboost':
            return XGBClassifier(use_label_encoder=False, **self.params)
        elif self.model_type == 'lightgbm':
            return LGBMClassifier(**self.params)
        else:
            raise ValueError("Unsupported model type")

    def train(self, X_train, y_train, X_val, y_val):
        model = self.get_model()
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        return model

    def evaluate(self, model, X_val, y_val):
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        print(f"\n✅ {self.model_type} 验证集准确率: {acc:.4f}")
        print("\n📋 分类报告:")
        print(classification_report(y_val, y_pred))
        return acc


# -----------------------------
# PyTorch 相关模型
# -----------------------------
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128), nn.ReLU(), nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, x_recon


class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(model, criterion, optimizer, dataloader, epochs=50):
    model.train()
    for epoch in range(epochs):
        print(f" Epoch {epoch + 1}/{epochs}")
        for x_batch, y_batch in dataloader:
            out = model(x_batch)
            loss = criterion(out, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def train_autoencoder(autoencoder, criterion_ae, optimizer_ae, dataloader, epochs=50):
    autoencoder.train()
    for _ in range(epochs):
        for x_batch, _ in dataloader:
            _, recon = autoencoder(x_batch)
            loss = criterion_ae(recon, x_batch)
            optimizer_ae.zero_grad()
            loss.backward()
            optimizer_ae.step()


# -----------------------------
# 主流程
# -----------------------------
def load_and_preprocess(data='train.csv'):
    file_path = os.path.join(DATA_DIR, data)
    df = pd.read_csv(file_path)
    X, y = preprocess_data(df, is_train=(data == 'train.csv'))

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    joblib.dump(le, LABEL_ENCODER_PATH)

    X_processed = encode_and_transform(X, y_encoded, fit=True)

    feature_names = pd.DataFrame(X_processed).columns.tolist()
    df_processed = pd.DataFrame(X_processed, columns=feature_names)
    df_original = pd.read_csv(file_path)
    df_processed['Fertilizer Name'] = df_original['Fertilizer Name'].values
    save_processed_data(df_processed, 'processed_train.csv')

    X_train, X_val, y_train, y_val = train_test_split(X_processed, y_encoded, test_size=0.2, random_state=42,
                                                      stratify=y_encoded)
    return X_train, X_val, y_train, y_val, le.classes_


def train_xgboost(X_train, y_train, X_val, y_val):
    trainer = ModelTrainer('xgboost', {
        'n_estimators': 2000,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'early_stopping_rounds' : 20,
        'eval_metric': 'mlogloss',
        'random_state': 42
    })
    model = trainer.train(X_train, y_train, X_val, y_val)
    model.save_model(XGBOOST_MODEL_PATH)
    trainer.evaluate(model, X_val, y_val)
    return model


def train_lightgbm(X_train, y_train, X_val, y_val, num_class):
    lgb_model = LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=50,
        subsample=0.7,
        colsample_bytree=0.6,
        objective='multiclass',
        num_class=num_class,
        eval_metric='multi_logloss',
        early_stopping_rounds=20,
        random_state=42
    )
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    print("\n💡 LightGBM 训练完成")
    return lgb_model


def train_pytorch_models(X_train, y_train, X_val, y_val):
    X_train_torch = torch.tensor(X_train, dtype=torch.float32)
    y_train_torch = torch.tensor(y_train, dtype=torch.long)
    X_val_torch = torch.tensor(X_val, dtype=torch.float32)

    dataset = TensorDataset(X_train_torch, y_train_torch)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    mlp = MLP(X_train.shape[1], len(np.unique(y_train)))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp.parameters(), lr=0.05)
    train_mlp(mlp, criterion, optimizer, loader, epochs=50)

    ae = Autoencoder(X_train.shape[1], 32)
    ae_loader = DataLoader(TensorDataset(X_train_torch, X_train_torch), batch_size=32, shuffle=True)
    train_autoencoder(ae, nn.MSELoss(), optim.Adam(ae.parameters(), lr=0.05), ae_loader, epochs=50)

    with torch.no_grad():
        X_train_encoded = ae.encoder(X_train_torch).numpy()
        X_val_encoded = ae.encoder(X_val_torch).numpy()

    ae_classifier = LogisticRegression().fit(X_train_encoded, y_train)
    avg_pred = (mlp(X_val_torch).argmax(dim=1).numpy() + ae_classifier.predict(X_val_encoded)) // 2
    acc = accuracy_score(y_val, avg_pred)
    print(f"\n✅ 自编码器集成验证准确率: {acc:.4f}")

    return mlp, ae_classifier


# -----------------------------
# 提交文件生成
# -----------------------------
def generate_submission(file_name='test.csv', output_file='submission.csv', top_k=3):
    file_path = os.path.join(DATA_DIR, file_name)
    df = pd.read_csv(file_name)
    ids = df['id'].values

    X, _ = preprocess_data(df, is_train=False)
    X_processed = encode_and_transform(X, fit=False)

    model = XGBClassifier()
    model.load_model(XGBOOST_MODEL_PATH)
    probas = model.predict_proba(X_processed)

    le = joblib.load(LABEL_ENCODER_PATH)
    top_k_indices = np.argsort(probas, axis=1)[:, -top_k:]
    predicted_labels = np.array([le.inverse_transform(row) for row in top_k_indices])
    predicted_strings = [" ".join(row) for row in predicted_labels]

    submission_df = pd.DataFrame({
        'id': ids,
        'Fertilizer Name': predicted_strings
    })

    submission_df.to_csv(output_file, index=False)
    print(f"✅ 提交文件已保存至 {output_file}")


# -----------------------------
# 主入口
# -----------------------------
if __name__ == "__main__":
    # 数据加载与预处理
    X_train, X_val, y_train, y_val, classes = load_and_preprocess()

    # 模型训练
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
    lgb_model = train_lightgbm(X_train, y_train, X_val, y_val, len(classes))
    mlp, ae_clf = train_pytorch_models(X_train, y_train, X_val, y_val)

    # 生成提交文件
    generate_submission()
