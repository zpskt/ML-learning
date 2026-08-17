#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ML-learning 
@File    ：cifar_cnn.py
@IDE     ：PyCharm 
@Author  ：张鹏
@Date    ：2026/8/18 00:18 
@Description： 
'''
"""
CIFAR-100 图像分类训练脚本
使用 PyTorch 实现的 CNN 模型
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import os

# =========================
# 1. 设备配置
# =========================
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"使用设备: {device}")

# =========================
# 2. 加载数据集
# =========================
TRAIN_PATH = "datasets/train"
with open(TRAIN_PATH, "rb") as f:
    dictionary = pickle.load(f, encoding="bytes")

raw_data = dictionary[b"data"]
images = raw_data.reshape(-1, 3, 32, 32)
labels = np.array(dictionary[b"fine_labels"])

print(f"原始数据形状: {images.shape}")
print(f"标签数量: {len(labels)}")

# =========================
# 3. 数据预处理与分割
# =========================
# 归一化
images = images.astype(np.float32) / 255.0

# 分割训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    images,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print(f"训练集图片: {X_train.shape}")
print(f"验证集图片: {X_val.shape}")

# =========================
# 4. 数据集类定义
# =========================
class CIFARDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = torch.tensor(images, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label

# =========================
# 5. 数据增强与数据加载器
# =========================
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

train_dataset = CIFARDataset(X_train, y_train, transform=train_transform)
val_dataset = CIFARDataset(X_val, y_val, transform=val_transform)

print(f"训练样本: {len(train_dataset)}")
print(f"验证样本: {len(val_dataset)}")

batch_size = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)

# =========================
# 6. CNN 模型定义
# =========================
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # block1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # block2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # block3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 100)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# =========================
# 7. 实例化模型
# =========================
model = CNN().to(device)
print(model)

# =========================
# 8. 定义损失函数和优化器
# =========================
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4
)

# =========================
# 9. 训练函数
# =========================
def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total

    return train_loss, train_accuracy

# =========================
# 10. 验证函数
# =========================
def validate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    validation_accuracy = 100 * correct / total
    return validation_accuracy

# =========================
# 11. 主训练循环
# =========================
def main():
    best_accuracy = 0.0
    train_losses = []
    train_accuracies = []
    val_accuracies = []

    num_epochs = 100

    # 检查是否有已保存的模型
    model_path = "best_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("已加载已保存的模型...")
    else:
        print("未发现模型文件，使用当前模型参数开始训练...")

    for epoch in range(num_epochs):
        # 训练
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # 验证
        validation_accuracy = validate(model, val_loader, device)

        # 保存最佳模型
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            torch.save(model.state_dict(), model_path)
            print(">>> 保存新的最佳模型！")

        # 记录历史
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(validation_accuracy)

        # 打印结果
        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Accuracy: {train_accuracy:.2f}% "
            f"Val Accuracy: {validation_accuracy:.2f}%"
        )

    print(f"最佳验证集准确率为: {best_accuracy:.2f}%")

    # =========================
    # 12. 绘制训练曲线
    # =========================
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label="训练准确率")
    plt.plot(val_accuracies, label="验证准确率")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("CIFAR-100 训练曲线")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("training_curve.png", dpi=150)
    plt.show()
    print("训练曲线已保存为 training_curve.png")

if __name__ == "__main__":
    main()
