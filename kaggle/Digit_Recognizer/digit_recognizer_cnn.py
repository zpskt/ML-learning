#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ML-learning 
@File    ：digit_recognizer_cnn.py
@IDE     ：PyCharm 
@Author  ：张鹏
@Date    ：2026/8/16 17:50 
@Description： 
'''
# ======================================================================
# Code cell 1
# ======================================================================
# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

if __name__ == '__main__':

    import os

    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))

    # ======================================================================
    # Code cell 2
    # ======================================================================
    import random
    import torch
    import pandas as pd
    import numpy as np

    SEED = 42
    # python标准库随机数。数据预处理：确保随机抽样、打乱列表、随机选择都一样。
    random.seed(SEED)
    # numpy 随机数，划分数据集、打乱数据数据，仅影响numpy的数据
    np.random.seed(SEED)
    # 在cpu上的随机数。使得所有初始化参数都是一样的
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        # GPU随机数
        torch.cuda.manual_seed_all(SEED)

    print(f"Random seed set to {SEED}")

    # ======================================================================
    # Code cell 3
    # ======================================================================
    train = pd.read_csv('datasets/train.csv')
    test = pd.read_csv('datasets/test.csv')

    print(train.shape, test.shape)
    train.head()

    # ======================================================================
    # Code cell 4
    # ======================================================================
    import matplotlib.pyplot as plt

    # 分离像素数据和标签值
    labels = train['label']
    pixels = train.drop(columns=['label'])

    # 展示前五个数据
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        img = pixels.iloc[i].values.reshape(28, 28)  # turn 784 flat numbers back into a 28x28 grid
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"label: {labels[i]}")
        axes[i].axis('off')
    plt.show()

    # ======================================================================
    # Code cell 5
    # ======================================================================
    X = train.drop('label', axis=1)
    y = train['label']

    print("Feature shape:", X.shape)
    print("Target shape:", y.shape)

    # ======================================================================
    # Code cell 6
    # ======================================================================
    image = X.iloc[0].values.reshape(28, 28)

    plt.figure(figsize=(4, 4))
    plt.imshow(image, cmap='gray')
    plt.title(f"label: {y.iloc[0]}")
    plt.axis('off')
    plt.show()

    # ======================================================================
    # Code cell 7
    # ======================================================================
    print("Minimum pixel value:", X.min().min())
    print("Maximum pixel value:", X.max().max())
    print("Mean pixel value:", X.mean().mean())

    # ======================================================================
    # Code cell 8
    # ======================================================================
    # 选择一个图片
    sample = X.iloc[0].values

    # 重构像素为 28x28
    sample_image = sample.reshape(28, 28)

    plt.figure(figsize=(4, 4))
    plt.imshow(sample_image, cmap='gray')
    plt.title(f"label: {y.iloc[0]}")
    plt.axis('off')
    plt.show()

    # ======================================================================
    # Code cell 9
    # ======================================================================
    # 将像素值从 0-255 转换格式到 0-1
    # 归一化不会影响么？ 它归一化并不会丢失数据，他们的亮暗对比程度是不变的，只不过绝对值变了，调整了数据尺度后，神经网络可以更快收敛，卷积计算和
    # 速度都会提升，所以亮暗关系其实并未消失或者缩减，只不过比例变了。
    # 还有哪些归一化，都用在什么场景？：1、Min-Max 归一化。2、标准化 3、按样本的L1 或L2 范数归一化
    X = X / 255.0

    print(X.shape)

    # ======================================================================
    # Code cell 10
    # ======================================================================
    # 将数据流转换为numpy
    X_numpy = X.values

    # 重组
    X_images = X_numpy.reshape(-1, 1, 28, 28)

    print(X_images.shape)

    # ======================================================================
    # Code cell 11
    # ======================================================================
    # 展示一个图片数据形状

    print(X_images[0].shape)

    # ======================================================================
    # Code cell 12
    # ======================================================================
    from sklearn.model_selection import train_test_split

    # ======================================================================
    # Code cell 13
    # ======================================================================
    X_train, X_val, y_train, y_val = train_test_split(
        X_images,
        y.values,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ======================================================================
    # Code cell 14
    # ======================================================================
    print("Training images:", X_train.shape)
    print("Validation images:", X_val.shape)

    print("Training labels:", y_train.shape)
    print("Validation labels:", y_val.shape)

    # ======================================================================
    # Code cell 15
    # ======================================================================
    import torch
    from torch.utils.data import Dataset, DataLoader


    # ======================================================================
    # Code cell 16
    # ======================================================================
    class MNISTDataset(Dataset):

        def __init__(self, images, labels):
            self.images = torch.tensor(images, dtype=torch.float32)
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return len(self.images)

        def __getitem__(self, index):
            return self.images[index], self.labels[index]


    # ======================================================================
    # Code cell 17
    # ======================================================================
    train_dataset = MNISTDataset(X_train, y_train)
    val_dataset = MNISTDataset(X_val, y_val)

    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    # ======================================================================
    # Code cell 18
    # ======================================================================
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

    # ======================================================================
    # Code cell 19
    # ======================================================================
    images, labels = next(iter(train_loader))

    print(images.shape)
    print(labels.shape)

    # ======================================================================
    # Code cell 20
    # ======================================================================
    import torch.nn as nn
    import torch.nn.functional as F


    # ======================================================================
    # Code cell 21
    # ======================================================================
    class CNN(nn.Module):

        def __init__(self):
            super().__init__()

            # First convolution layer
            self.conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3
            )

            # Second convolution layer
            self.conv2 = nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3
            )

            # Pooling layer
            self.pool = nn.MaxPool2d(2, 2)

            # Fully connected layer
            self.fc1 = nn.Linear(
                64 * 5 * 5,
                128
            )

            # Output layer
            self.fc2 = nn.Linear(
                128,
                10
            )

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))

            x = self.pool(F.relu(self.conv2(x)))

            x = x.view(x.size(0), -1)

            x = F.relu(self.fc1(x))

            x = self.fc2(x)

            return x


    # ======================================================================
    # Code cell 22
    # ======================================================================
    model = CNN()

    print(model)

    # ======================================================================
    # Code cell 23
    # ======================================================================
    images, labels = next(iter(train_loader))

    output = model(images)

    print("Input shape:", images.shape)
    print("Output shape:", output.shape)

    # ======================================================================
    # Code cell 24
    # ======================================================================
    criterion = nn.CrossEntropyLoss()

    # ======================================================================
    # Code cell 25
    # ======================================================================
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # ======================================================================
    # Code cell 26
    # ======================================================================
    # Number of training passes through the dataset
    num_epochs = 10

    # ======================================================================
    # Code cell 27
    # ======================================================================
    for epoch in range(num_epochs):

        # Put the model in training mode
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, labels)

            # Clear previous gradients
            optimizer.zero_grad()

            # Compute gradients
            loss.backward()

            # Update weights
            optimizer.step()

            # Statistics
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Loss: {train_loss:.4f} "
            f"Accuracy: {train_accuracy:.2f}%"
        )

    # ======================================================================
    # Code cell 28
    # ======================================================================
    # Put the model in evaluation mode
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    validation_accuracy = 100 * correct / total

    print(f"Validation Accuracy: {validation_accuracy:.2f}%")

    # ======================================================================
    # Code cell 29
    # ======================================================================
    # Normalize pixel values
    X_test = test / 255.0

    # Convert to NumPy array
    X_test = X_test.values

    # Reshape for CNN
    X_test = X_test.reshape(-1, 1, 28, 28)

    print("Test shape:", X_test.shape)

    # ======================================================================
    # Code cell 30
    # ======================================================================
    X_test_tensor = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    print(X_test_tensor.shape)

    # ======================================================================
    # Code cell 31
    # ======================================================================
    # Evaluation mode
    model.eval()

    with torch.no_grad():
        outputs = model(X_test_tensor)

        predictions = torch.argmax(outputs, dim=1)

    print(predictions.shape)

    # ======================================================================
    # Code cell 32
    # ======================================================================
    submission = pd.DataFrame({
        "ImageId": np.arange(1, len(predictions) + 1),
        "Label": predictions.numpy()
    })

    submission.head()

    # ======================================================================
    # Code cell 33
    # ======================================================================
    submission.to_csv("submission.csv", index=False)

    print("submission.csv has been created successfully.")

    # ======================================================================
    # Code cell 34
    # ======================================================================
    plt.figure(figsize=(12, 8))

    for i in range(12):
        plt.subplot(3, 4, i + 1)

        image = X_test[i].reshape(28, 28)

        plt.imshow(image, cmap="gray")

        plt.title(f"Prediction: {predictions[i].item()}")

        plt.axis("off")

    plt.tight_layout()

    plt.show()

