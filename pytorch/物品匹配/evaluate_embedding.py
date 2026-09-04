#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ML-learning 
@File    ：evaluate_embedding.py.py
@IDE     ：PyCharm 
@Author  ：张鹏
@Date    ：2026/9/5 00:01 
@Description： 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image
from pathlib import Path
from itertools import combinations


# =========================
# 配置
# =========================

PRODUCT_ROOT = Path("data/products")

PRODUCTS = [
    "百事可乐",
    "可口可乐",
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


# =========================
# 构建 ResNet50
# =========================

def build_resnet50(device):
    weights = models.ResNet50_Weights.DEFAULT

    model = models.resnet50(weights=weights)

    # 去掉最后的分类层
    # ResNet50 原本：
    #
    # Feature
    #   ↓
    # FC
    #   ↓
    # 1000 classes
    #
    # 我们只需要 FC 前面的 2048D embedding
    model.fc = nn.Identity()

    model = model.to(device)
    model.eval()

    return model, weights


# =========================
# 提取单张图片 embedding
# =========================

def extract_embedding(
    model,
    transform,
    image_path,
    device,
):
    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0).to(device)

    with torch.inference_mode():
        embedding = model(image)

    # L2 Normalize
    embedding = F.normalize(
        embedding,
        p=2,
        dim=1,
    )

    return embedding.squeeze(0)


# =========================
# 读取一个产品的所有图片
# =========================

def get_image_paths(product_name):
    product_dir = PRODUCT_ROOT / product_name

    if not product_dir.exists():
        raise FileNotFoundError(
            f"Product directory not found: {product_dir}"
        )

    image_paths = [
        path
        for path in product_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    image_paths.sort()

    return image_paths


# =========================
# 提取整个产品的 embeddings
# =========================

def extract_product_embeddings(
    model,
    transform,
    product_name,
    device,
):
    image_paths = get_image_paths(product_name)

    if len(image_paths) < 2:
        raise RuntimeError(
            f"{product_name} needs at least 2 images."
        )

    print(
        f"\nExtracting {product_name}: "
        f"{len(image_paths)} images"
    )

    embeddings = []

    for index, image_path in enumerate(image_paths):

        embedding = extract_embedding(
            model=model,
            transform=transform,
            image_path=image_path,
            device=device,
        )

        embeddings.append(embedding)

        print(
            f"  [{index + 1:03d}/{len(image_paths)}] "
            f"{image_path.name}"
        )

    embeddings = torch.stack(embeddings)

    return image_paths, embeddings


# =========================
# 统计 Similarity
# =========================

def print_statistics(name, similarities):

    similarities = torch.tensor(
        similarities,
        dtype=torch.float32,
    )

    print(f"\n{name}")

    print("-" * 50)

    print(
        f"count : {len(similarities)}"
    )

    print(
        f"mean  : {similarities.mean():.4f}"
    )

    print(
        f"std   : {similarities.std():.4f}"
    )

    print(
        f"min   : {similarities.min():.4f}"
    )

    print(
        f"max   : {similarities.max():.4f}"
    )


# =========================
# 同一个产品内部比较
# =========================

def calculate_same_product_similarity(
    embeddings,
):
    similarities = []

    # combinations 可以保证：
    #
    # 0001 ↔ 0002
    # 0001 ↔ 0003
    # 0002 ↔ 0003
    #
    # 不会计算：
    #
    # 0001 ↔ 0001
    #
    for i, j in combinations(
        range(len(embeddings)),
        2,
    ):
        similarity = torch.sum(
            embeddings[i] * embeddings[j]
        )

        similarities.append(
            similarity.item()
        )

    return similarities


# =========================
# 不同产品之间比较
# =========================

def calculate_different_product_similarity(
    embeddings_a,
    embeddings_b,
):
    similarities = []

    for embedding_a in embeddings_a:

        for embedding_b in embeddings_b:

            similarity = torch.sum(
                embedding_a * embedding_b
            )

            similarities.append(
                similarity.item()
            )

    return similarities


# =========================
# 主程序
# =========================

def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # -------------------------
    # 加载模型
    # -------------------------

    model, weights = build_resnet50(
        device
    )

    transform = weights.transforms()

    # -------------------------
    # 提取两个产品 embedding
    # -------------------------

    product_data = {}

    for product_name in PRODUCTS:

        image_paths, embeddings = (
            extract_product_embeddings(
                model=model,
                transform=transform,
                product_name=product_name,
                device=device,
            )
        )

        product_data[product_name] = {
            "paths": image_paths,
            "embeddings": embeddings,
        }

        print(
            f"{product_name} embedding shape: "
            f"{embeddings.shape}"
        )

    # -------------------------
    # 百事内部
    # -------------------------

    pepsi_embeddings = product_data[
        "百事可乐"
    ]["embeddings"]

    pepsi_same = (
        calculate_same_product_similarity(
            pepsi_embeddings
        )
    )

    print_statistics(
        "百事可乐 ↔ 百事可乐",
        pepsi_same,
    )

    # -------------------------
    # 可口内部
    # -------------------------

    coke_embeddings = product_data[
        "可口可乐"
    ]["embeddings"]

    coke_same = (
        calculate_same_product_similarity(
            coke_embeddings
        )
    )

    print_statistics(
        "可口可乐 ↔ 可口可乐",
        coke_same,
    )

    # -------------------------
    # 百事 vs 可口
    # -------------------------

    different = (
        calculate_different_product_similarity(
            pepsi_embeddings,
            coke_embeddings,
        )
    )

    print_statistics(
        "百事可乐 ↔ 可口可乐",
        different,
    )

    # -------------------------
    # 最终结论辅助信息
    # -------------------------

    print("\n" + "=" * 60)
    print("Embedding Evaluation Summary")
    print("=" * 60)

    print(
        f"Pepsi same-product min : "
        f"{min(pepsi_same):.4f}"
    )

    print(
        f"Coke same-product min  : "
        f"{min(coke_same):.4f}"
    )

    print(
        f"Different-product max  : "
        f"{max(different):.4f}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
