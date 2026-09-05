#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ML-learning 
@File    ：train_rtdetr.py
@IDE     ：PyCharm 
@Author  ：张鹏
@Date    ：2026/9/5 08:17 
@Description： 
'''
import os
import json
import math
from pathlib import Path

from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
)

# ============================================================
# 配置
# ============================================================

DATASET_DIR = Path(
    "/Users/zhangpeng/Desktop/workspace/github/ML-learning/pytorch/"
    "物品匹配/目标检测/data/Bottle"
)

# 预训练 RT-DETR
PRETRAINED_MODEL = "PekingU/rtdetr_r50vd_coco_o365"

# 输出目录
OUTPUT_DIR = Path("./checkpoints")

# 训练参数
NUM_CLASSES = 6
BATCH_SIZE = 4
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# 从哪个 epoch 开始
RESUME = True

# 使用 GPU / CPU
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    # if torch.backends.mps.is_available():
    #     return torch.device("mps")

    return torch.device("cpu")


DEVICE = get_device()

print(f"Using device: {DEVICE}")

# ============================================================
# COCO Dataset
# ============================================================

class COCODataset(Dataset):

    def __init__(
        self,
        root_dir,
        split,
        processor,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.split_dir = self.root_dir / split
        self.processor = processor

        json_files = list(self.split_dir.glob("*.json"))

        if len(json_files) != 1:
            raise RuntimeError(
                f"{self.split_dir} 中应该有且只有一个 JSON，"
                f"实际找到 {len(json_files)} 个"
            )

        self.annotation_file = json_files[0]

        with open(
            self.annotation_file,
            "r",
            encoding="utf-8",
        ) as f:
            self.coco = json.load(f)

        self.images = self.coco["images"]
        self.categories = self.coco["categories"]

        # image_id -> annotations
        self.annotations = {}

        for ann in self.coco["annotations"]:

            image_id = ann["image_id"]

            if image_id not in self.annotations:
                self.annotations[image_id] = []

            self.annotations[image_id].append(ann)

        # COCO category_id -> 连续 label
        self.category_to_label = {
            category["id"]: index
            for index, category in enumerate(
                self.categories
            )
        }

        print(
            f"[{split}] images={len(self.images)}, "
            f"annotations={len(self.coco['annotations'])}"
        )

        print(
            f"[{split}] categories={self.categories}"
        )
        category_counts = Counter(
            ann["category_id"]
            for ann in self.coco["annotations"]
        )

        print("Category counts:")

        for category in self.categories:
            category_id = category["id"]
            category_name = category["name"]

            print(
                f"  id={category_id}, "
                f"name={category_name}, "
                f"count={category_counts.get(category_id, 0)}"
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image_info = self.images[index]

        image_id = image_info["id"]
        image_path = (
            self.split_dir /
            image_info["file_name"]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        annotations = self.annotations.get(
            image_id,
            []
        )

        coco_annotations = []

        for ann in annotations:

            category_id = ann["category_id"]

            if category_id not in self.category_to_label:
                continue

            coco_annotations.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": ann["bbox"],
                    "area": ann.get(
                        "area",
                        ann["bbox"][2] * ann["bbox"][3],
                    ),
                    "iscrowd": ann.get(
                        "iscrowd",
                        0,
                    ),
                }
            )

        # Transformers 的 image processor
        # 接收 COCO 格式 annotation
        encoded = self.processor(
            images=image,
            annotations={
                "image_id": image_id,
                "annotations": coco_annotations,
            },
            return_tensors="pt",
        )

        # 去掉 batch 维度
        pixel_values = encoded["pixel_values"].squeeze(0)

        labels = encoded["labels"][0]

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }


# ============================================================
# DataLoader
# ============================================================

def collate_fn(batch):

    pixel_values = torch.stack(
        [
            item["pixel_values"]
            for item in batch
        ]
    )

    labels = [
        item["labels"]
        for item in batch
    ]

    return {
        "pixel_values": pixel_values,
        "labels": labels,
    }


# ============================================================
# 构建模型
# ============================================================

def build_model():

    print(
        f"Loading pretrained model: "
        f"{PRETRAINED_MODEL}"
    )

    model = AutoModelForObjectDetection.from_pretrained(
        PRETRAINED_MODEL,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )

    return model


# ============================================================
# Optimizer
# ============================================================

def build_optimizer(model):

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    return optimizer


# ============================================================
# 保存 checkpoint
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    best_loss,
    path,
):

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_loss": best_loss,
    }

    torch.save(
        checkpoint,
        path,
    )

    print(
        f"Checkpoint saved: {path}"
    )


# ============================================================
# 加载 checkpoint
# ============================================================

def load_checkpoint(
    model,
    optimizer,
    scheduler,
    path,
):

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    start_epoch = checkpoint["epoch"] + 1
    best_loss = checkpoint["best_loss"]

    print(
        f"Resume training from epoch "
        f"{start_epoch}"
    )

    print(
        f"Best loss: {best_loss:.6f}"
    )

    return start_epoch, best_loss


# ============================================================
# 训练
# ============================================================

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    epoch,
):

    model.train()

    total_loss = 0.0

    for step, batch in enumerate(dataloader):

        pixel_values = batch[
            "pixel_values"
        ].to(DEVICE)

        labels = [
            {
                key: value.to(DEVICE)
                for key, value in label.items()
            }
            for label in batch["labels"]
        ]

        outputs = model(
            pixel_values=pixel_values,
            labels=labels,
        )

        loss = outputs.loss

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if step % 10 == 0:

            print(
                f"Epoch [{epoch}] "
                f"Step [{step}/{len(dataloader)}] "
                f"Loss: {loss.item():.6f}"
            )

    return total_loss / len(dataloader)


# ============================================================
# 验证
# ============================================================

@torch.no_grad()
def validate(
    model,
    dataloader,
):

    model.eval()

    total_loss = 0.0

    for batch in dataloader:

        pixel_values = batch[
            "pixel_values"
        ].to(DEVICE)

        labels = [
            {
                key: value.to(DEVICE)
                for key, value in label.items()
            }
            for label in batch["labels"]
        ]

        outputs = model(
            pixel_values=pixel_values,
            labels=labels,
        )

        total_loss += outputs.loss.item()

    return total_loss / len(dataloader)


# ============================================================
# 主函数
# ============================================================

def main():

    print("=" * 60)
    print("RT-DETR Training")
    print("=" * 60)

    print(
        f"use Device: {DEVICE}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Processor
    # --------------------------------------------------------

    processor = AutoImageProcessor.from_pretrained(
        PRETRAINED_MODEL
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = COCODataset(
        DATASET_DIR,
        "train",
        processor,
    )

    valid_dataset = COCODataset(
        DATASET_DIR,
        "valid",
        processor,
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = build_model()

    model = model.to(DEVICE)

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = build_optimizer(
        model
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
    )

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    last_checkpoint = (
        OUTPUT_DIR / "last.pth"
    )

    start_epoch = 0
    best_loss = math.inf

    if RESUME and last_checkpoint.exists():

        start_epoch, best_loss = load_checkpoint(
            model,
            optimizer,
            scheduler,
            last_checkpoint,
        )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(
        start_epoch,
        NUM_EPOCHS,
    ):

        print()
        print("=" * 60)
        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}"
        )
        print("=" * 60)

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            epoch + 1,
        )

        valid_loss = validate(
            model,
            valid_loader,
        )

        scheduler.step()

        print(
            f"\nEpoch {epoch + 1}: "
            f"train_loss={train_loss:.6f}, "
            f"valid_loss={valid_loss:.6f}"
        )

        # ----------------------------------------------------
        # 保存 last
        # ----------------------------------------------------

        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            best_loss,
            OUTPUT_DIR / "last.pth",
        )

        # ----------------------------------------------------
        # 保存 best
        # ----------------------------------------------------

        if valid_loss < best_loss:

            best_loss = valid_loss

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                best_loss,
                OUTPUT_DIR / "best.pth",
            )

            print(
                f"Best model updated! "
                f"valid_loss={best_loss:.6f}"
            )

    print()
    print("=" * 60)
    print("Training finished")
    print("=" * 60)


if __name__ == "__main__":
    main()