import os
from collections import Counter

import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from torch.utils.tensorboard import SummaryWriter
import tensorboard
# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, 'data')

writer = SummaryWriter(log_dir=os.path.join(current_dir, 'runs'))

# 在训练开始前定义日志文件路径
log_file_path = os.path.join(current_dir, 'best_log.txt')
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
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def prepare_data():
    '''
    加载数据并进行预处理
    :return:
    '''
    file_path = os.path.join(data_dir, 'train.csv')
    df = pd.read_csv(file_path)

    # 特征和标签
    X = df.drop(columns=['Fertilizer Name'])
    y = df['Fertilizer Name'].values
    '''
    人工校验数据
    '''
    # 检查分类是否均衡
    print(Counter(y))

    # 定义预处理器：类别型列做 OneHot，数值型列标准化
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Soil Type', 'Crop Type']),
        ('num', StandardScaler(), X.columns.difference(['Soil Type', 'Crop Type']))
    ])
    # todo 这里的列变得非常多，需要改
    X_processed = preprocessor.fit_transform(X)

    # 编码标签（虽然你已处理过，但确保是整数形式）
    le = LabelEncoder()
    y = le.fit_transform(y)

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)

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
    # 找一个适合多类别分类的损失函数
    criterion = nn.CrossEntropyLoss()
    # 定义优化器
    # optimizer = optim.Adam(model.parameters(), lr=0.0005)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
    # 训练循环
    epochs = 50
    print(f"🚀 开始训练，共 {epochs} 个 epoch")
    # 初始化最佳准确率
    best_acc = 0.0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    for epoch in range(epochs):
        # 将模型设置为训练模式，以便启用dropout、batch normalization等在训练时需要的特性
        model.train()
        # 初始化总损失为0，用于累计整个训练过程中所有批次的损失值
        total_loss = 0
        # 遍历训练数据加载器，获取每个批次的输入数据和目标标签
        for X_batch, y_batch in train_loader:
            # 将当前批次的输入数据和目标标签移动到指定的设备（如GPU）上
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            # 清零优化器的梯度，避免梯度累积影响下一次计算
            optimizer.zero_grad()
            # 将当前批次的输入数据传递给模型，获取模型的输出结果
            outputs = model(X_batch)
            # 计算模型输出结果与目标标签之间的损失值
            loss = criterion(outputs, y_batch)
            # 对损失值进行反向传播，计算模型参数的梯度
            loss.backward()
            # 根据计算出的梯度，使用优化器对模型参数进行一次更新
            optimizer.step()
            # 累加当前批次的损失值到总损失中，用于后续的损失值跟踪和可视化
            total_loss += loss.item()

        # 将模型设置为评估模式，以禁用dropout等仅在训练期间启用的功能
        model.eval()
        # 初始化列表，用于存储真实标签和预测标签
        y_true, y_pred = [], []
        # 使用torch.no_grad()上下文管理器禁用梯度计算，以提高推理速度并减少内存消耗
        with torch.no_grad():
            # 遍历验证数据集中的每个批次
            for X_batch, y_batch in val_loader:
                # 将批次数据移动到指定设备（如GPU）
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                # 使用模型进行预测
                outputs = model(X_batch)
                # 通过获取每个输出行中最大值的索引来确定预测标签
                preds = torch.argmax(outputs, dim=1)
                # 将真实标签从Tensor转换为numpy数组，并添加到y_true列表中
                y_true.extend(y_batch.cpu().numpy())
                # 将预测标签从Tensor转换为numpy数组，并添加到y_pred列表中
                y_pred.extend(preds.cpu().numpy())

        # 看哪些类别的 precision/recall 很低； 看看是否是学不会某一类还是都学不会
        print(classification_report(y_true, y_pred))
        # 计算模型准确率
        acc = accuracy_score(y_true, y_pred)
        # 打印当前轮次的训练信息，包括轮次、损失值和验证集上的准确率
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss:.4f}, Val Acc: {acc:.4f}")

        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            # 保存模型的状态字典到指定路径
            torch.save(model.state_dict(), os.path.join(current_dir, 'best_torch_model.pth'))
            print("💾 最佳模型已保存")
            # 写入日志文件（覆盖模式，只保留最新一行）
            with open(log_file_path, 'w') as log_file:
                log_file.write(f"Best Accuracy: {best_acc:.4f}\n")

        scheduler.step(total_loss)
        writer.add_scalar('Loss/train', total_loss, epoch)
        writer.add_scalar('Accuracy/val', acc, epoch)

    print("🎉 训练完成！")
    return model


if __name__ == '__main__':
    trained_model = train_torch_model()
