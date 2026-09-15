#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Project ：物品匹配
@File    ：train_dinov2.py
@Author  ：张鹏
@Description：
    使用 feature_library 训练 DINOv2 商品分类模型。

    训练阶段：
        image
          ↓
        DINOv2-base
          ↓
        CLS embedding (768)
          ↓
        Linear(768, num_classes)
          ↓
        CrossEntropyLoss

    推理/检索阶段：
        不使用分类 FC
        直接使用 DINOv2 的 768 维 CLS embedding 做向量检索。
"""

from pathlib import Path
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import AutoModel


# ============================================================
# 1. 配置
# ============================================================

DATA_DIR = Path("./feature_library")
CHECKPOINT_DIR = Path("./checkpoints")

MODEL_NAME = "facebook/dinov2-base"

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 20

# DINOv2 本身比较大，微调 backbone 时学习率不能太大
BACKBONE_LR = 1e-5

# 分类头可以使用稍大的学习率
HEAD_LR = 1e-4

WEIGHT_DECAY = 1e-4

VAL_RATIO = 0.2
RANDOM_SEED = 42

NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 2. 固定随机种子
# ============================================================

def set_seed(seed):
    random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. 数据集
# ============================================================

class ProductDataset(Dataset):

    def __init__(
        self,
        samples,
        class_to_idx,
        transform=None,
    ):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, product_name = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = self.class_to_idx[product_name]

        return image, label


# ============================================================
# 4. 扫描数据
# ============================================================

def load_samples(data_dir):

    if not data_dir.exists():
        raise FileNotFoundError(
            f"数据目录不存在: {data_dir.resolve()}"
        )

    product_dirs = sorted(
        [
            path
            for path in data_dir.iterdir()
            if path.is_dir()
        ]
    )

    if not product_dirs:
        raise RuntimeError(
            f"没有找到商品目录: {data_dir.resolve()}"
        )

    class_to_idx = {
        product_dir.name: idx
        for idx, product_dir in enumerate(product_dirs)
    }

    samples = []

    valid_suffixes = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    for product_dir in product_dirs:

        product_name = product_dir.name

        image_paths = [
            path
            for path in product_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in valid_suffixes
        ]

        print(
            f"{product_name}: {len(image_paths)} images"
        )

        for image_path in image_paths:
            samples.append(
                (
                    image_path,
                    product_name,
                )
            )

    if not samples:
        raise RuntimeError(
            "没有找到任何图片。"
        )

    return samples, class_to_idx


# ============================================================
# 5. 分层划分 Train / Val
# ============================================================

def split_samples(
    samples,
    class_to_idx,
    val_ratio,
    seed,
):

    rng = random.Random(seed)

    train_samples = []
    val_samples = []

    # 每个商品单独划分
    # 避免某个类别完全进入 train 或 val
    samples_by_class = {
        class_name: []
        for class_name in class_to_idx
    }

    for image_path, product_name in samples:
        samples_by_class[product_name].append(
            (image_path, product_name)
        )

    for product_name, class_samples in samples_by_class.items():

        rng.shuffle(class_samples)

        num_val = max(
            1,
            int(len(class_samples) * val_ratio)
        )

        val_part = class_samples[:num_val]
        train_part = class_samples[num_val:]

        train_samples.extend(train_part)
        val_samples.extend(val_part)

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    return train_samples, val_samples


# ============================================================
# 6. 数据增强
# ============================================================

def build_transforms():

    # 训练：
    # 保持比例 Resize + RandomResizedCrop
    #
    # 注意：
    # RandomResizedCrop 可能裁掉商品部分。
    # 当前先保持和常规视觉分类 Fine-tuning 一致，
    # 后续会专门实验“完整商品 + Padding”的方案。

    train_transform = transforms.Compose(
        [
            transforms.Resize(256),

            transforms.RandomResizedCrop(
                IMAGE_SIZE,
                scale=(0.8, 1.0),
            ),

            transforms.RandomHorizontalFlip(
                p=0.5
            ),

            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )

    # 验证：
    # 不使用随机增强
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),

            transforms.CenterCrop(
                IMAGE_SIZE
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )

    return train_transform, val_transform


# ============================================================
# 7. DINOv2 分类模型
# ============================================================

class DINOv2Classifier(nn.Module):

    def __init__(
        self,
        model_name,
        num_classes,
    ):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(
            model_name
        )

        hidden_size = (
            self.backbone.config.hidden_size
        )

        print(
            f"DINOv2 embedding dimension: "
            f"{hidden_size}"
        )

        self.classifier = nn.Linear(
            hidden_size,
            num_classes,
        )

    def forward(self, pixel_values):

        outputs = self.backbone(
            pixel_values=pixel_values
        )

        # CLS token
        embedding = outputs.last_hidden_state[:, 0]

        logits = self.classifier(
            embedding
        )

        return logits

    @torch.inference_mode()
    def extract_embedding(self, pixel_values):

        outputs = self.backbone(
            pixel_values=pixel_values
        )

        embedding = outputs.last_hidden_state[:, 0]

        return embedding


# ============================================================
# 8. 训练一个 Epoch
# ============================================================

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = logits.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += images.size(0)

    avg_loss = total_loss / total

    accuracy = (
        correct / total
    )

    return avg_loss, accuracy


# ============================================================
# 9. 验证
# ============================================================

@torch.inference_mode()
def validate(
    model,
    dataloader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        logits = model(images)

        loss = criterion(
            logits,
            labels
        )

        total_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = logits.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += images.size(0)

    avg_loss = total_loss / total

    accuracy = (
        correct / total
    )

    return avg_loss, accuracy


# ============================================================
# 10. 保存 Checkpoint
# ============================================================

def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    best_val_accuracy,
    class_to_idx,
):

    checkpoint = {
        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "best_val_accuracy":
            best_val_accuracy,

        "class_to_idx":
            class_to_idx,

        "num_classes":
            len(class_to_idx),

        "model_name":
            MODEL_NAME,

        "embedding_dimension":
            model.backbone.config.hidden_size,
    }

    torch.save(
        checkpoint,
        path
    )


# ============================================================
# 11. 主程序
# ============================================================

def main():

    set_seed(
        RANDOM_SEED
    )

    print("=" * 60)
    print("DINOv2 Fine-tuning")
    print("=" * 60)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        f"Data: {DATA_DIR.resolve()}"
    )

    # --------------------------------------------------------
    # 数据
    # --------------------------------------------------------

    samples, class_to_idx = load_samples(
        DATA_DIR
    )

    train_samples, val_samples = split_samples(
        samples,
        class_to_idx,
        VAL_RATIO,
        RANDOM_SEED,
    )

    print()
    print(
        f"Total images: {len(samples)}"
    )

    print(
        f"Train images: {len(train_samples)}"
    )

    print(
        f"Val images: {len(val_samples)}"
    )

    print()

    print("Class mapping:")

    for class_name, class_idx in class_to_idx.items():

        print(
            f"  {class_idx}: {class_name}"
        )

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    train_transform, val_transform = (
        build_transforms()
    )

    train_dataset = ProductDataset(
        train_samples,
        class_to_idx,
        train_transform,
    )

    val_dataset = ProductDataset(
        val_samples,
        class_to_idx,
        val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = DINOv2Classifier(
        MODEL_NAME,
        len(class_to_idx),
    )

    model = model.to(
        DEVICE
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Optimizer
    #
    # backbone 小学习率
    # classifier 大学习率
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        [
            {
                "params":
                    model.backbone.parameters(),
                "lr":
                    BACKBONE_LR,
            },
            {
                "params":
                    model.classifier.parameters(),
                "lr":
                    HEAD_LR,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
    )

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    best_checkpoint_path = (
        CHECKPOINT_DIR
        / "dinov2_best.pth"
    )

    last_checkpoint_path = (
        CHECKPOINT_DIR
        / "dinov2_last.pth"
    )

    best_val_accuracy = 0.0

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                DEVICE,
            )
        )

        val_loss, val_accuracy = (
            validate(
                model,
                val_loader,
                criterion,
                DEVICE,
            )
        )

        scheduler.step()

        current_lr_backbone = (
            optimizer.param_groups[0]["lr"]
        )

        current_lr_head = (
            optimizer.param_groups[1]["lr"]
        )

        print(
            f"Epoch "
            f"{epoch:02d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy * 100:.2f}% | "
            f"Backbone LR: {current_lr_backbone:.2e} | "
            f"Head LR: {current_lr_head:.2e}"
        )

        # 保存 last
        save_checkpoint(
            last_checkpoint_path,
            epoch,
            model,
            optimizer,
            scheduler,
            best_val_accuracy,
            class_to_idx,
        )

        # 保存 best
        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            save_checkpoint(
                best_checkpoint_path,
                epoch,
                model,
                optimizer,
                scheduler,
                best_val_accuracy,
                class_to_idx,
            )

            print(
                f"  -> Best model saved: "
                f"{best_val_accuracy * 100:.2f}%"
            )

    print()
    print("=" * 60)
    print(
        f"Best Val Accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )
    print(
        f"Best checkpoint: "
        f"{best_checkpoint_path}"
    )
    print(
        f"Last checkpoint: "
        f"{last_checkpoint_path}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()