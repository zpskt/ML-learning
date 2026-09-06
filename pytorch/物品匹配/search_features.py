#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：_annotations.coco.json 
@File    ：search_features.py
@IDE     ：PyCharm 
@Author  ：张鹏
@Date    ：2026/9/5 18:27 
@Description： 
'''
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models


DEFAULT_FEATURE_FILE = "features.pt"
DEFAULT_TOP_K = 5


# ============================================================
# Device
# ============================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# Model
# ============================================================

def build_resnet50(device):
    weights = models.ResNet50_Weights.DEFAULT

    model = models.resnet50(weights=weights)

    # 去掉 ImageNet 分类层
    model.fc = nn.Identity()

    model = model.to(device)
    model.eval()

    transform = weights.transforms()

    return model, transform


# ============================================================
# Query Feature
# ============================================================

@torch.inference_mode()
def extract_feature(model, transform, image_path, device):
    """
    提取单张查询图片的 2048 维 embedding。
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Query image does not exist: {image_path}"
        )

    with Image.open(image_path) as image:
        image = image.convert("RGB")

    image = transform(image).unsqueeze(0)
    image = image.to(device)

    embedding = model(image)

    # L2 Normalize
    embedding = F.normalize(
        embedding,
        p=2,
        dim=1,
    )

    return embedding


# ============================================================
# Load Feature Library
# ============================================================

def load_feature_library(feature_file):
    """
    加载 extract_features.py 生成的 features.pt。
    """

    feature_file = Path(feature_file)

    if not feature_file.exists():
        raise FileNotFoundError(
            f"Feature file does not exist: {feature_file}"
        )

    data = torch.load(
        feature_file,
        map_location="cpu",
    )

    required_keys = {
        "embeddings",
        "image_paths",
    }

    missing_keys = required_keys - data.keys()

    if missing_keys:
        raise ValueError(
            f"Feature file is missing keys: {missing_keys}"
        )

    embeddings = data["embeddings"]

    # 确保是二维 Tensor
    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected embeddings with shape [N, D], "
            f"but got {embeddings.shape}"
        )

    # 再归一化一次。
    # 这样即使以后 feature 文件不是归一化的，
    # search 也不会出问题。
    embeddings = F.normalize(
        embeddings,
        p=2,
        dim=1,
    )

    return data, embeddings


# ============================================================
# Search
# ============================================================

def search(
    query_embedding,
    database_embeddings,
    image_paths,
    top_k,
):
    """
    使用 cosine similarity 做 Top-K 搜索。

    因为 query 和 database 都已经 L2 normalize，
    所以：

        cosine_similarity
        =
        dot product
    """

    similarities = torch.matmul(
        database_embeddings,
        query_embedding.T,
    ).squeeze(1)

    top_k = min(
        top_k,
        len(similarities),
    )

    scores, indices = torch.topk(
        similarities,
        k=top_k,
    )

    results = []

    for score, index in zip(scores, indices):

        index = index.item()
        score = score.item()

        results.append(
            {
                "score": score,
                "image_path": image_paths[index],
            }
        )

    return results


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Search image features using cosine similarity."
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query image path.",
    )

    parser.add_argument(
        "--features",
        type=str,
        default=DEFAULT_FEATURE_FILE,
        help="Feature library file.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of results.",
    )

    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError(
            "top-k must be greater than 0."
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    print("=" * 60)
    print("Feature Search")
    print("=" * 60)

    print(f"Query:    {args.query}")
    print(f"Features: {args.features}")
    print(f"Top-K:    {args.top_k}")
    print(f"Device:   {device}")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading ResNet50...")

    model, transform = build_resnet50(device)

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Load database
    # --------------------------------------------------------

    data, database_embeddings = load_feature_library(
        args.features
    )

    image_paths = data["image_paths"]

    print(
        f"Feature library: "
        f"{database_embeddings.shape[0]} images"
    )

    print(
        f"Embedding dimension: "
        f"{database_embeddings.shape[1]}"
    )

    # --------------------------------------------------------
    # Extract query feature
    # --------------------------------------------------------

    print("\nExtracting query feature...")

    query_embedding = extract_feature(
        model=model,
        transform=transform,
        image_path=args.query,
        device=device,
    )

    print(
        f"Query embedding: "
        f"{query_embedding.shape}"
    )

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    results = search(
        query_embedding=query_embedding.cpu(),
        database_embeddings=database_embeddings,
        image_paths=image_paths,
        top_k=args.top_k,
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("Top-K Results")
    print("=" * 60)

    for rank, result in enumerate(results, start=1):

        print(
            f"Top {rank}: "
            f"score={result['score']:.4f} "
            f"path={result['image_path']}"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()
