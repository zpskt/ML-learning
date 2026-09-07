#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：图像算法
@File    ：train_resnet50.py
@IDE     ：PyCharm
@Author  ：张鹏
@Date    ：2026/9/7
@Description：
    ResNet50 商品分类 Fine-tuning 实验

    实验目的：
    1. 使用 ImageNet 预训练 ResNet50
    2. 使用自己的商品数据进行 Fine-tuning
    3. 训练完成后可以去掉 FC
    4. 使用 Backbone 的 embedding 进行商品相似度检索
"""

from pathlib import Path
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image


# ============================================================
# Config
# ============================================================

DATA_DIR = Path("./feature_library")

OUTPUT_DIR = Path("./checkpoints")

BEST_MODEL_PATH = OUTPUT_DIR / "resnet50_best.pth"
LAST_MODEL_PATH = OUTPUT_DIR / "resnet50_last.pth"

BATCH_SIZE = 32

NUM_EPOCHS = 20

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

VAL_RATIO = 0.2

RANDOM_SEED = 42

NUM_WORKERS = 0


# ============================================================
# Device
# ============================================================

def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# ============================================================
# Seed
# ============================================================

def set_seed(seed):

    random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Dataset
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

        image_path, product_id = self.samples[index]

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform is not None:

            image = self.transform(image)

        label = self.class_to_idx[
            product_id
        ]

        return image, label


# ============================================================
# Load Dataset
# ============================================================

def load_samples(root_dir):

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    samples = []

    product_ids = []

    for product_dir in sorted(
        root_dir.iterdir()
    ):

        if not product_dir.is_dir():
            continue

        product_id = product_dir.name

        product_ids.append(product_id)

        for image_path in sorted(
            product_dir.iterdir()
        ):

            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in image_extensions:
                continue

            samples.append(
                (
                    image_path,
                    product_id,
                )
            )

    product_ids = sorted(
        set(product_ids)
    )

    return samples, product_ids


# ============================================================
# Split Dataset
# ============================================================

def split_dataset(
        samples,
        val_ratio,
        seed,
):

    rng = random.Random(seed)

    samples_by_class = {}

    for image_path, product_id in samples:

        samples_by_class.setdefault(
            product_id,
            []
        ).append(
            (
                image_path,
                product_id,
            )
        )

    train_samples = []
    val_samples = []

    for product_id in sorted(
        samples_by_class
    ):

        class_samples = samples_by_class[
            product_id
        ]

        rng.shuffle(
            class_samples
        )

        if len(class_samples) < 2:

            raise RuntimeError(
                f"Product '{product_id}' "
                f"has only "
                f"{len(class_samples)} image(s). "
                f"At least 2 images are required."
            )

        val_count = max(
            1,
            int(
                len(class_samples)
                * val_ratio
            ),
        )

        val_samples.extend(
            class_samples[:val_count]
        )

        train_samples.extend(
            class_samples[val_count:]
        )

    return train_samples, val_samples


# ============================================================
# Model
# ============================================================

def build_model(
        num_classes,
        device,
):

    weights = (
        models.ResNet50_Weights.DEFAULT
    )

    model = models.resnet50(
        weights=weights
    )

    # ImageNet 分类层：
    #
    # 2048 -> 1000
    #
    # 替换成自己的商品分类：
    #
    # 2048 -> num_classes

    model.fc = nn.Linear(
        2048,
        num_classes,
    )

    model = model.to(device)

    return model, weights


# ============================================================
# Train One Epoch
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

    total_correct = 0

    total_samples = 0

    for images, labels in dataloader:

        images = images.to(device)

        labels = labels.to(device)

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        batch_size = images.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        predictions = (
            outputs.argmax(dim=1)
        )

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

    epoch_loss = (
        total_loss
        / total_samples
    )

    epoch_accuracy = (
        total_correct
        / total_samples
    )

    return epoch_loss, epoch_accuracy


# ============================================================
# Validation
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

    total_correct = 0

    total_samples = 0

    for images, labels in dataloader:

        images = images.to(device)

        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        batch_size = images.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        predictions = (
            outputs.argmax(dim=1)
        )

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

    epoch_loss = (
        total_loss
        / total_samples
    )

    epoch_accuracy = (
        total_correct
        / total_samples
    )

    return epoch_loss, epoch_accuracy


# ============================================================
# Save Checkpoint
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

        "model":
            "resnet50",

    }

    torch.save(
        checkpoint,
        path,
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)

    print(
        "ResNet50 Fine-tuning Experiment"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    set_seed(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # Check Dataset
    # --------------------------------------------------------

    if not DATA_DIR.exists():

        raise FileNotFoundError(
            f"Dataset not found: "
            f"{DATA_DIR}"
        )

    # --------------------------------------------------------
    # Load Samples
    # --------------------------------------------------------

    samples, product_ids = (
        load_samples(
            DATA_DIR
        )
    )

    if not samples:

        raise RuntimeError(
            "No images found."
        )

    print(
        f"Images: {len(samples)}"
    )

    print(
        f"Products: {len(product_ids)}"
    )

    print()

    # --------------------------------------------------------
    # Class Mapping
    # --------------------------------------------------------

    class_to_idx = {
        product_id: index
        for index, product_id
        in enumerate(product_ids)
    }

    print("Classes:")

    print("-" * 60)

    for product_id in product_ids:

        print(
            f"{class_to_idx[product_id]:2d} "
            f"{product_id}"
        )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train_samples, val_samples = (
        split_dataset(
            samples,
            VAL_RATIO,
            RANDOM_SEED,
        )
    )

    print()

    print(
        f"Train images: "
        f"{len(train_samples)}"
    )

    print(
        f"Val images:   "
        f"{len(val_samples)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model, weights = build_model(
        num_classes=len(product_ids),
        device=device,
    )

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    train_transform = transforms.Compose([

        transforms.Resize(
            256
        ),

        transforms.RandomResizedCrop(
            224,
            scale=(0.8, 1.0),
        ),

        transforms.RandomHorizontalFlip(),

        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
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
    ])

    val_transform = transforms.Compose([

        transforms.Resize(
            256
        ),

        transforms.CenterCrop(
            224
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
    ])

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

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
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
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
    # Output Directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_accuracy = 0.0

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
            )
        )

        val_loss, val_accuracy = (
            validate(
                model,
                val_loader,
                criterion,
                device,
            )
        )

        scheduler.step()

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        print()

        print(
            f"Epoch "
            f"{epoch:02d}/{NUM_EPOCHS}"
        )

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Acc:  "
            f"{train_accuracy * 100:.2f}%"
        )

        print(
            f"Val Loss:   "
            f"{val_loss:.4f}"
        )

        print(
            f"Val Acc:    "
            f"{val_accuracy * 100:.2f}%"
        )

        print(
            f"LR:         "
            f"{current_lr:.8f}"
        )

        # ----------------------------------------------------
        # Save Last
        # ----------------------------------------------------

        save_checkpoint(
            LAST_MODEL_PATH,
            epoch,
            model,
            optimizer,
            scheduler,
            best_val_accuracy,
            class_to_idx,
        )

        # ----------------------------------------------------
        # Save Best
        # ----------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = (
                val_accuracy
            )

            save_checkpoint(
                BEST_MODEL_PATH,
                epoch,
                model,
                optimizer,
                scheduler,
                best_val_accuracy,
                class_to_idx,
            )

            print(
                f"★ Best model saved: "
                f"{best_val_accuracy * 100:.2f}%"
            )

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    print()

    print("=" * 60)

    print(
        "Training Finished"
    )

    print("=" * 60)

    print(
        f"Best Val Accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )

    print(
        f"Best Model: "
        f"{BEST_MODEL_PATH}"
    )

    print(
        f"Last Model: "
        f"{LAST_MODEL_PATH}"
    )

    print("=" * 60)


if __name__ == "__main__":

    main()

