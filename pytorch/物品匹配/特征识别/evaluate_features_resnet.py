#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Project ：图像算法
@File    ：evaluate_features_resnet.py
@IDE     ：PyCharm
@Author  ：张鹏
@Date    ：2026/9/6
@Description：
    评估 Embedding 的商品识别能力。

    评价指标：
    1. Top-1 Accuracy
    2. Recall@5
    3. Recall@10
    4. Per Product Top-1 Accuracy
    5. Confusion Matrix
    6. Intra-class / Inter-class Similarity
"""

from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


# ============================================================
# Config
# ============================================================

TEST_DIR = Path("./test_images")
FEATURES_PATH = Path("./features.pt")
CHECKPOINT_PATH = Path("checkpoints/resnet50_best.pth")

TOP_K_LIST = [1, 5, 10]


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

    model = models.resnet50(weights=None)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    num_classes = checkpoint["num_classes"]

    model.fc = nn.Sequential(
        # nn.Dropout(p=0.3),
        nn.Linear(
            2048,
            num_classes,
        ),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # 去掉分类层，只保留 Embedding
    model.fc = nn.Identity()

    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),

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

    return model, transform


# ============================================================
# Load Test Images
# ============================================================

def load_test_images(test_dir):

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    image_paths = []
    ground_truths = []

    for product_dir in sorted(test_dir.iterdir()):

        if not product_dir.is_dir():
            continue

        product_id = product_dir.name

        for image_path in sorted(product_dir.rglob("*")):

            if image_path.suffix.lower() not in image_extensions:
                continue

            image_paths.append(image_path)
            ground_truths.append(product_id)

    return image_paths, ground_truths


# ============================================================
# Extract Embedding
# ============================================================

@torch.inference_mode()
def extract_embedding(
    model,
    transform,
    image_path,
    device,
):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0).to(device)

    embedding = model(image)

    # L2 Normalize
    embedding = F.normalize(
        embedding,
        p=2,
        dim=1,
    )

    return embedding


# ============================================================
# Build Product Database
# ============================================================

def build_product_database(
    database_embeddings,
    database_product_ids,
):

    """
    将：

        多张数据库图片
            ↓
        product_id

    进行分组。

    返回：

        {
            product_id: tensor([indices])
        }
    """

    product_to_indices = defaultdict(list)

    for index, product_id in enumerate(
        database_product_ids
    ):
        product_to_indices[product_id].append(index)

    return product_to_indices


# ============================================================
# Evaluate One Image
# ============================================================

def evaluate_one(
    model,
    transform,
    database_embeddings,
    product_to_indices,
    image_path,
    ground_truth,
    device,
):

    query_embedding = extract_embedding(
        model=model,
        transform=transform,
        image_path=image_path,
        device=device,
    )

    # --------------------------------------------------------
    # Image-level cosine similarity
    # --------------------------------------------------------

    similarities = torch.matmul(
        database_embeddings,
        query_embedding.T,
    ).squeeze(1)

    # --------------------------------------------------------
    # Product-level similarity
    #
    # 一个商品可能有多张数据库图片。
    #
    # 当前采用：
    # 一个商品的最大相似度作为该商品最终得分
    # --------------------------------------------------------

    product_scores = []

    for product_id, indices in product_to_indices.items():

        indices = torch.tensor(
            indices,
            device=device,
            dtype=torch.long,
        )

        product_similarity = similarities[indices].max()

        product_scores.append(
            (
                product_id,
                product_similarity.item(),
            )
        )

    # --------------------------------------------------------
    # Sort products by similarity
    # --------------------------------------------------------

    product_scores.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    predictions = [
        {
            "product_id": product_id,
            "score": score,
        }
        for product_id, score in product_scores
    ]

    # --------------------------------------------------------
    # Top-K
    # --------------------------------------------------------

    topk_results = {}

    for k in TOP_K_LIST:

        actual_k = min(
            k,
            len(predictions),
        )

        topk_products = [
            item["product_id"]
            for item in predictions[:actual_k]
        ]

        topk_results[k] = (
            ground_truth in topk_products
        )

    return (
        topk_results,
        predictions,
        query_embedding,
    )


# ============================================================
# Calculate Pairwise Similarity Statistics
# ============================================================

def calculate_embedding_statistics(
    embeddings,
    product_ids,
):

    """
    计算：

    Intra-class similarity
        同商品之间的平均相似度

    Inter-class similarity
        不同商品之间的平均相似度

    embeddings 已经 L2 normalize，
    因此 dot product = cosine similarity。
    """

    embeddings = F.normalize(
        embeddings,
        p=2,
        dim=1,
    )

    product_ids = list(product_ids)

    intra_similarities = []
    inter_similarities = []

    n = len(embeddings)

    for i in range(n):

        # 与后面的样本比较，避免重复计算
        for j in range(i + 1, n):

            similarity = torch.dot(
                embeddings[i],
                embeddings[j],
            ).item()

            if product_ids[i] == product_ids[j]:

                intra_similarities.append(
                    similarity
                )

            else:

                inter_similarities.append(
                    similarity
                )

    statistics = {
        "intra_mean": (
            sum(intra_similarities)
            / len(intra_similarities)
            if intra_similarities
            else 0.0
        ),

        "inter_mean": (
            sum(inter_similarities)
            / len(inter_similarities)
            if inter_similarities
            else 0.0
        ),
    }

    return statistics


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("Embedding Evaluation")
    print("=" * 70)

    device = get_device()

    print(f"Device: {device}")
    print(f"Test directory: {TEST_DIR}")
    print(f"Feature database: {FEATURES_PATH}")

    # ========================================================
    # Check Test Directory
    # ========================================================

    if not TEST_DIR.exists():

        raise FileNotFoundError(
            f"Test directory not found: {TEST_DIR}"
        )

    # ========================================================
    # Load Test Images
    # ========================================================

    image_paths, ground_truths = load_test_images(
        TEST_DIR
    )

    if not image_paths:

        raise RuntimeError(
            "No test images found."
        )

    print(
        f"Test images: {len(image_paths)}"
    )

    # ========================================================
    # Load Feature Database
    # ========================================================

    feature_data = torch.load(
        FEATURES_PATH,
        map_location="cpu",
        weights_only=False,
    )

    database_embeddings = feature_data[
        "embeddings"
    ]

    database_product_ids = feature_data[
        "product_ids"
    ]

    print(
        f"Database embeddings: "
        f"{database_embeddings.shape}"
    )

    # ========================================================
    # Normalize Database
    # ========================================================

    database_embeddings = F.normalize(
        database_embeddings,
        p=2,
        dim=1,
    ).to(device)

    # ========================================================
    # Build Product Database
    # ========================================================

    product_to_indices = build_product_database(
        database_embeddings,
        database_product_ids,
    )

    product_count = len(
        product_to_indices
    )

    print(
        f"Database products: "
        f"{product_count}"
    )

    # ========================================================
    # Build Model
    # ========================================================

    model, transform = build_model(
        device
    )

    print("Model loaded.")

    # ========================================================
    # Statistics
    # ========================================================

    total = 0

    topk_correct_count = {
        k: 0
        for k in TOP_K_LIST
    }

    # 每个商品
    product_total = defaultdict(int)

    product_correct = defaultdict(int)

    # Confusion Matrix
    confusion_matrix = defaultdict(
        lambda: defaultdict(int)
    )

    # Embedding statistics
    test_embeddings = []
    test_product_ids = []

    # ========================================================
    # Evaluate
    # ========================================================

    print()
    print("=" * 70)
    print("Results")
    print("=" * 70)

    for i, (
        image_path,
        ground_truth,
    ) in enumerate(
        zip(
            image_paths,
            ground_truths,
        ),
        start=1,
    ):

        try:

            (
                topk_results,
                predictions,
                query_embedding,
            ) = evaluate_one(
                model=model,
                transform=transform,
                database_embeddings=database_embeddings,
                product_to_indices=product_to_indices,
                image_path=image_path,
                ground_truth=ground_truth,
                device=device,
            )

        except Exception as e:

            print(
                f"[ERROR] {image_path}: {e}"
            )

            continue

        # ----------------------------------------------------
        # Valid sample count
        # ----------------------------------------------------

        total += 1

        # ----------------------------------------------------
        # Save embedding
        # ----------------------------------------------------

        test_embeddings.append(
            query_embedding.squeeze(0).cpu()
        )

        test_product_ids.append(
            ground_truth
        )

        # ----------------------------------------------------
        # Top-K
        # ----------------------------------------------------

        for k in TOP_K_LIST:

            if topk_results[k]:

                topk_correct_count[k] += 1

        # ----------------------------------------------------
        # Top-1
        # ----------------------------------------------------

        top1_prediction = predictions[0]

        top1_product = (
            top1_prediction["product_id"]
        )

        top1_correct = (
            top1_product == ground_truth
        )

        # ----------------------------------------------------
        # Per Product
        # ----------------------------------------------------

        product_total[
            ground_truth
        ] += 1

        if top1_correct:

            product_correct[
                ground_truth
            ] += 1

        # ----------------------------------------------------
        # Confusion Matrix
        # ----------------------------------------------------

        confusion_matrix[
            ground_truth
        ][
            top1_product
        ] += 1

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        status = (
            "OK"
            if top1_correct
            else "WRONG"
        )

        print(
            f"[{i}/{len(image_paths)}] "
            f"{image_path.name:<35} "
            f"GT={ground_truth:<15} "
            f"Pred={top1_product:<15} "
            f"Score={top1_prediction['score']:.4f} "
            f"{status}"
        )

    # ========================================================
    # Accuracy
    # ========================================================

    print()
    print("=" * 70)
    print("Evaluation Summary")
    print("=" * 70)

    print(
        f"Evaluated samples: "
        f"{total}"
    )

    for k in TOP_K_LIST:

        accuracy = (
            topk_correct_count[k] / total
            if total > 0
            else 0.0
        )

        if k == 1:

            print(
                f"Top-1 Accuracy: "
                f"{accuracy:.2%}"
            )

        else:

            print(
                f"Recall@{k}: "
                f"{accuracy:.2%}"
            )

    # ========================================================
    # Per Product Accuracy
    # ========================================================

    print()
    print(
        "Per Product Top-1 Accuracy"
    )
    print("-" * 70)

    for product_id in sorted(
        product_total
    ):

        total_count = (
            product_total[product_id]
        )

        correct_count = (
            product_correct.get(
                product_id,
                0,
            )
        )

        accuracy = (
            correct_count
            / total_count
        )

        print(
            f"{product_id:<20} "
            f"{correct_count:>4}/"
            f"{total_count:<4} "
            f"({accuracy:.2%})"
        )

    # ========================================================
    # Confusion Matrix
    # ========================================================

    print()
    print(
        "Confusion Matrix"
    )
    print("-" * 70)

    products = sorted(
        product_total.keys()
    )

    # Header
    print(
        f"{'GT Pred':<20}",
        end="",
    )

    for product_id in products:

        print(
            f"{product_id:<15}",
            end="",
        )

    print()

    # Rows
    for gt in products:

        print(
            f"{gt:<20}",
            end="",
        )

        for pred in products:

            count = confusion_matrix[
                gt
            ][
                pred
            ]

            print(
                f"{count:<15}",
                end="",
            )

        print()

    # ========================================================
    # Embedding Space Statistics
    # ========================================================

    if len(test_embeddings) >= 2:

        test_embeddings = torch.stack(
            test_embeddings
        )

        statistics = (
            calculate_embedding_statistics(
                embeddings=test_embeddings,
                product_ids=test_product_ids,
            )
        )

        print()
        print(
            "Embedding Space Statistics"
        )
        print("-" * 70)

        print(
            "Intra-class Mean Similarity: "
            f"{statistics['intra_mean']:.4f}"
        )

        print(
            "Inter-class Mean Similarity: "
            f"{statistics['inter_mean']:.4f}"
        )

        print(
            "Separation Gap: "
            f"{statistics['intra_mean'] - statistics['inter_mean']:.4f}"
        )

    # ========================================================
    # Finish
    # ========================================================

    print()
    print("=" * 70)
    print("Evaluation Finished")
    print("=" * 70)


if __name__ == "__main__":
    main()