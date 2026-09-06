#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：_annotations.coco.json 
@File    ：build_feature_library.py
@IDE     ：PyCharm 
@Author  ：张鹏
@Date    ：2026/9/6 23:11 
@Description： 创建特征库，目前写在pt文件里面
'''
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image


# ============================================================
# Config
# ============================================================

FEATURE_LIBRARY_DIR = Path("./feature_library")
OUTPUT_PATH = Path("./features.pt")

BATCH_SIZE = 32


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
# Model
# ============================================================

def build_model(device):
    weights = models.ResNet50_Weights.DEFAULT

    model = models.resnet50(weights=weights)

    # 去掉 ImageNet 分类层
    model.fc = nn.Identity()

    model = model.to(device)
    model.eval()

    return model, weights


# ============================================================
# Load Images
# ============================================================

def load_images(root_dir):
    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    image_paths = []
    product_ids = []

    for product_dir in sorted(root_dir.iterdir()):

        if not product_dir.is_dir():
            continue

        product_id = product_dir.name

        for image_path in sorted(product_dir.rglob("*")):

            if image_path.suffix.lower() not in image_extensions:
                continue

            image_paths.append(image_path)
            product_ids.append(product_id)

    return image_paths, product_ids


# ============================================================
# Extract Batch Embedding
# ============================================================

@torch.inference_mode()
def extract_batch(
    model,
    transform,
    image_paths,
    device,
):

    images = []

    valid_paths = []

    for image_path in image_paths:

        try:
            image = Image.open(image_path).convert("RGB")

            image = transform(image)

            images.append(image)
            valid_paths.append(image_path)

        except Exception as e:

            print(
                f"[Warning] Failed to load "
                f"{image_path}: {e}"
            )

    if not images:
        return None, []

    batch = torch.stack(images).to(device)

    embeddings = model(batch)

    # L2 Normalize
    embeddings = F.normalize(
        embeddings,
        p=2,
        dim=1,
    )

    return embeddings.cpu(), valid_paths


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("Build Feature Library")
    print("=" * 60)

    device = get_device()

    print(f"Device: {device}")
    print(f"Feature library: {FEATURE_LIBRARY_DIR}")

    if not FEATURE_LIBRARY_DIR.exists():
        raise FileNotFoundError(
            f"Feature library not found: "
            f"{FEATURE_LIBRARY_DIR}"
        )

    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    image_paths, product_ids = load_images(
        FEATURE_LIBRARY_DIR
    )

    if not image_paths:
        raise RuntimeError(
            "No images found in feature library."
        )

    print(f"Images found: {len(image_paths)}")

    # --------------------------------------------------------
    # Print product statistics
    # --------------------------------------------------------

    product_counts = {}

    for product_id in product_ids:
        product_counts[product_id] = (
            product_counts.get(product_id, 0) + 1
        )

    print()
    print("Products:")
    print("-" * 60)

    for product_id, count in product_counts.items():

        print(
            f"{product_id:<20} "
            f"{count} images"
        )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model, weights = build_model(device)

    transform = weights.transforms()

    print()
    print("Model loaded.")

    # --------------------------------------------------------
    # Extract embeddings
    # --------------------------------------------------------

    all_embeddings = []
    all_paths = []
    all_product_ids = []

    total = len(image_paths)

    for start in range(0, total, BATCH_SIZE):

        end = min(
            start + BATCH_SIZE,
            total,
        )

        batch_paths = image_paths[start:end]

        embeddings, valid_paths = extract_batch(
            model,
            transform,
            batch_paths,
            device,
        )

        if embeddings is None:
            continue

        all_embeddings.append(embeddings)

        all_paths.extend(
            str(path)
            for path in valid_paths
        )

        # 根据实际成功读取的图片
        for path in valid_paths:

            # feature_library/product_id/image.jpg
            product_id = path.parent.name

            all_product_ids.append(
                product_id
            )

        print(
            f"[{end}/{total}] "
            f"processed"
        )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    embeddings = torch.cat(
        all_embeddings,
        dim=0,
    )

    # 再做一次 Normalize，保证最终库严格归一化
    embeddings = F.normalize(
        embeddings,
        p=2,
        dim=1,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    feature_data = {
        "embeddings": embeddings,
        "image_paths": all_paths,
        "product_ids": all_product_ids,

        "dimension": embeddings.shape[1],

        "model": "resnet50_imagenet",

        "normalized": True,
    }

    torch.save(
        feature_data,
        OUTPUT_PATH,
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    norms = torch.norm(
        embeddings,
        p=2,
        dim=1,
    )

    print()
    print("=" * 60)
    print("Feature Library Built")
    print("=" * 60)

    print(f"Images:       {len(all_paths)}")
    print(f"Embeddings:   {embeddings.shape}")
    print(f"Dimension:    {embeddings.shape[1]}")
    print(f"Products:     {len(set(all_product_ids))}")
    print(f"Normalized:   {feature_data['normalized']}")
    print(
        f"Norm range:   "
        f"{norms.min().item():.6f} ~ "
        f"{norms.max().item():.6f}"
    )
    print(f"Output:       {OUTPUT_PATH}")

    print("=" * 60)


if __name__ == "__main__":
    main()