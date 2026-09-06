#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：_annotations.coco.json 
@File    ：recognize_product.py
@IDE     ：PyCharm 
@Author  ：张鹏
@Date    ：2026/9/6 22:30 
@Description： 
'''
# recognize_product.py

from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models


# ============================================================
# Config
# ============================================================

QUERY_IMAGE = "./query.jpg"
FEATURES_PATH = "features.pt"

TOP_K = 5

# 现在先不依赖这个阈值做最终结论
# 后面根据正负样本 similarity 分布确定
MATCH_THRESHOLD = 0.80


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

    # 去掉分类层，只保留特征提取部分
    model.fc = nn.Identity()

    model = model.to(device)
    model.eval()

    return model, weights


# ============================================================
# Extract Embedding
# ============================================================

@torch.inference_mode()
def extract_embedding(model, weights, image_path, device):

    image = Image.open(image_path).convert("RGB")

    transform = weights.transforms()

    image = transform(image).unsqueeze(0).to(device)

    embedding = model(image)

    # L2 Normalize
    embedding = F.normalize(
        embedding,
        p=2,
        dim=1,
    )

    return embedding


# ============================================================
# Product ID
# ============================================================

def get_product_id(image_path):
    """
    当前临时规则：

    crops/
        product_xxx/
            xxx.jpg

    使用父目录作为 product_id。

    注意：
    这只是当前测试阶段的临时方案。
    正式特征库必须有明确的 product_id。
    """

    return Path(image_path).parent.name


# ============================================================
# Product Aggregation
# ============================================================

def aggregate_by_product(
    similarities,
    image_paths,
    top_k,
):
    """
    先取 Top-K 图片，
    然后按照 product_id 聚合。

    当前商品得分：
        该商品 Top-K 图片中的最高 similarity

    例如：

        coca-cola:
            0.91
            0.88
            0.84

        sprite:
            0.87

    coca-cola 最终得分 = 0.91
    """

    values, indices = torch.topk(
        similarities,
        k=min(top_k, len(similarities)),
    )

    products = defaultdict(list)

    top_results = []

    for score, index in zip(values, indices):

        score = score.item()
        index = index.item()

        image_path = image_paths[index]

        product_id = get_product_id(image_path)

        products[product_id].append({
            "score": score,
            "image_path": image_path,
        })

        top_results.append({
            "score": score,
            "image_path": image_path,
            "product_id": product_id,
        })

    # 商品最终得分 = 该商品最高相似度
    product_scores = []

    for product_id, results in products.items():

        best_score = max(
            item["score"]
            for item in results
        )

        product_scores.append({
            "product_id": product_id,
            "score": best_score,
            "matches": results,
        })

    product_scores.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return product_scores, top_results


# ============================================================
# Main
# ============================================================

def main():

    device = get_device()

    print("=" * 60)
    print("Product Recognition")
    print("=" * 60)

    print(f"Device: {device}")
    print(f"Query: {QUERY_IMAGE}")
    print(f"Features: {FEATURES_PATH}")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model, weights = build_model(device)

    print("Model loaded.")

    # --------------------------------------------------------
    # Load feature database
    # --------------------------------------------------------

    feature_data = torch.load(
        FEATURES_PATH,
        map_location="cpu",
        weights_only=False,
    )

    database_embeddings = feature_data["embeddings"]
    image_paths = feature_data["image_paths"]

    print(f"Database images: {len(image_paths)}")
    print(f"Embedding dimension: {database_embeddings.shape[1]}")

    # --------------------------------------------------------
    # Normalize database
    # --------------------------------------------------------

    database_embeddings = F.normalize(
        database_embeddings,
        p=2,
        dim=1,
    ).to(device)

    # --------------------------------------------------------
    # Extract query embedding
    # --------------------------------------------------------

    query_embedding = extract_embedding(
        model,
        weights,
        QUERY_IMAGE,
        device,
    )

    print(f"Query embedding shape: {query_embedding.shape}")

    # --------------------------------------------------------
    # Cosine Similarity
    #
    # 因为 query 和 database 都已经 L2 normalize，
    # 所以 cosine similarity = dot product
    # --------------------------------------------------------

    similarities = torch.matmul(
        database_embeddings,
        query_embedding.T,
    ).squeeze(1)

    # --------------------------------------------------------
    # Product aggregation
    # --------------------------------------------------------

    product_results, top_results = aggregate_by_product(
        similarities,
        image_paths,
        TOP_K,
    )

    if not product_results:
        print("No results.")
        return

    best_product = product_results[0]

    # --------------------------------------------------------
    # Final decision
    # --------------------------------------------------------

    if best_product["score"] >= MATCH_THRESHOLD:
        status = "MATCH"
    else:
        status = "UNKNOWN"

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("Recognition Result")
    print("=" * 60)

    print(f"Product: {best_product['product_id']}")
    print(f"Score:   {best_product['score']:.4f}")
    print(f"Status:  {status}")

    print()
    print("Top-K Image Matches")
    print("-" * 60)

    for i, result in enumerate(top_results, start=1):

        print(
            f"Top {i}: "
            f"product={result['product_id']:<20} "
            f"score={result['score']:.4f} "
            f"path={result['image_path']}"
        )

    print()
    print("Product Ranking")
    print("-" * 60)

    for i, product in enumerate(product_results, start=1):

        print(
            f"{i}. "
            f"{product['product_id']:<20} "
            f"score={product['score']:.4f}"
        )

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()