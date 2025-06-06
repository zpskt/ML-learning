import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import os
import joblib

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, 'data')

# 设置设备（GPU 或 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ 当前设备: {device}")


class FertilizerDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.X = features
        self.y = labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)


class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=64):
        super(MLP, self).__init__()
        # 初始化神经网络结构
        self.net = nn.Sequential(
            # 输入层到隐藏层的线性变换
            nn.Linear(input_dim, hidden_dim),
            # 使用ReLU激活函数引入非线性
            nn.ReLU(),
            # 添加Dropout层以防止过拟合，丢弃率0.3
            nn.Dropout(0.3),
            # 第一个隐藏层到自身的线性变换，加深网络理解
            nn.Linear(hidden_dim, hidden_dim),
            # 再次使用ReLU激活函数
            nn.ReLU(),
            # 最后一个隐藏层到输出层的线性变换
            nn.Linear(hidden_dim, num_classes)
        )


    def forward(self, x):
        return self.net(x)


def prepare_data():
    '''
    加载数据并进行预处理
    :return:
    '''
    # 加载预处理后的数据
    file_path = os.path.join(data_dir, 'processed_train.csv')
    df = pd.read_csv(file_path)

    # 特征和标签
    X = df.drop(columns=['Fertilizer Name']).values
    y = df['Fertilizer Name'].values

    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 编码标签
    le = LabelEncoder()
    y = le.fit_transform(y)

    # 划分训练/验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 构建 Dataset 和 DataLoader
    train_dataset = FertilizerDataset(X_train, y_train)
    val_dataset = FertilizerDataset(X_val, y_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

    return train_loader, val_loader, len(le.classes_)


def train_torch_model():
    train_loader, val_loader, num_classes = prepare_data()
    input_dim = train_loader.dataset.X.shape[1]  # 自动获取特征数量
    # 初始化模型、损失函数和优化器
    model = MLP(input_dim=input_dim, num_classes=num_classes).to(device)  # 根据特征数量调整 input_dim
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    epochs = 50
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 验证
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                preds = torch.argmax(outputs, dim=1)

                y_true.extend(y_batch.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        acc = accuracy_score(y_true, y_pred)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}, Val Acc: {acc:.4f}")

        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(current_dir, 'best_torch_model.pth'))
            print("💾 最佳模型已保存")

    print("🎉 训练完成！")
    return model


if __name__ == '__main__':
    trained_model = train_torch_model()
