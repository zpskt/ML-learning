#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：_annotations.coco.json 
@File    ：evaluate_features_resnet.py
@IDE     ：PyCharm 
@Author  ：张鹏
@Date    ：2026/9/6 23:24 
@Description： 评估embedding，用testimage文件夹评估
'''
# evaluate_features_resnet.py

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from transformers import AutoModel

# ============================================================
# Config
# ============================================================

TEST_DIR = Path("./test_images")
FEATURES_PATH = Path("./features_dinov2.pt")
MODEL_NAME = "facebook/dinov2-base"

TOP_K = 5

# 先不要用阈值影响 Accuracy
# 当前阶段只评价“最像哪个商品”
# 阈值后面单独评估
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
    model = AutoModel.from_pretrained(MODEL_NAME)

    model = model.to(device)
    model.eval()

    # DINOv2 的输入尺寸通常使用 224x224。
    # Resize(224) 保持原始宽高比，再 CenterCrop。
    # todo 这里需要修改，因为这里的crop会裁掉一部分瓶子尺寸
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
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
    image = Image.open(image_path).convert("RGB")

    image = transform(image).unsqueeze(0).to(device)

    outputs = model(image)

    # DINOv2 的 CLS Token 作为整张图片的 embedding
    embedding = outputs.last_hidden_state[:, 0]

    # L2 Normalize
    embedding = F.normalize(
        embedding,
        p=2,
        dim=1,
    )

    return embedding


# ============================================================
# Evaluate One Image
# ============================================================

def evaluate_one(
    model,
    transform,
    database_embeddings,
    database_product_ids,
    image_path,
    ground_truth,
    device,
):
    query_embedding = extract_embedding(
        model,
        transform,
        image_path,
        device,
    )

    # cosine similarity
    similarities = torch.matmul(
        database_embeddings,
        query_embedding.T,
    ).squeeze(1)

    k = min(
        TOP_K,
        len(similarities),
    )

    scores, indices = torch.topk(
        similarities,
        k=k,
    )

    predictions = []

    for score, index in zip(scores, indices):

        index = index.item()
        score = score.item()

        predictions.append({
            "product_id": database_product_ids[index],
            "score": score,
        })

    top1_prediction = predictions[0]["product_id"]

    top1_correct = (
        top1_prediction == ground_truth
    )

    top5_correct = any(
        result["product_id"] == ground_truth
        for result in predictions
    )

    return (
        top1_correct,
        top5_correct,
        predictions,
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("Feature Evaluation")
    print("=" * 60)

    device = get_device()

    print(f"Device: {device}")
    print(f"Test directory: {TEST_DIR}")
    print(f"Feature database: {FEATURES_PATH}")

    # --------------------------------------------------------
    # Check test directory
    # --------------------------------------------------------

    if not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Test directory not found: {TEST_DIR}"
        )

    # --------------------------------------------------------
    # Load test images
    # --------------------------------------------------------

    image_paths, ground_truths = load_test_images(
        TEST_DIR
    )

    if not image_paths:
        raise RuntimeError(
            "No test images found."
        )

    print(f"Test images: {len(image_paths)}")

    # --------------------------------------------------------
    # Load feature database
    # --------------------------------------------------------

    feature_data = torch.load(
        FEATURES_PATH,
        map_location="cpu",
        weights_only=False,
    )

    database_embeddings = feature_data["embeddings"]

    # 新版 features.pt
    database_product_ids = feature_data["product_ids"]

    # --------------------------------------------------------
    # Normalize database
    # --------------------------------------------------------

    database_embeddings = F.normalize(
        database_embeddings,
        p=2,
        dim=1,
    ).to(device)

    print(
        f"Database embeddings: "
        f"{database_embeddings.shape}"
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model, transform = build_model(device)

    print("Model loaded.")

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total = len(image_paths)

    top1_correct_count = 0
    top5_correct_count = 0

    # 每个商品统计
    product_total = {}
    product_correct = {}

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)

    for i, (image_path, ground_truth) in enumerate(
        zip(image_paths, ground_truths),
        start=1,
    ):

        try:

            (
                top1_correct,
                top5_correct,
                predictions,
            ) = evaluate_one(
                model=model,
                transform=transform,
                database_embeddings=database_embeddings,
                database_product_ids=database_product_ids,
                image_path=image_path,
                ground_truth=ground_truth,
                device=device,
            )

        except Exception as e:

            print(
                f"[ERROR] {image_path}: {e}"
            )

            continue

        # 总体统计
        if top1_correct:
            top1_correct_count += 1

        if top5_correct:
            top5_correct_count += 1

        # 商品统计
        product_total[ground_truth] = (
            product_total.get(ground_truth, 0) + 1
        )

        if top1_correct:
            product_correct[ground_truth] = (
                product_correct.get(ground_truth, 0) + 1
            )

        # 输出
        top1 = predictions[0]

        status = (
            "OK"
            if top1_correct
            else "WRONG"
        )

        print(
            f"[{i}/{total}] "
            f"{image_path.name:<35} "
            f"GT={ground_truth:<15} "
            f"Pred={top1['product_id']:<15} "
            f"Score={top1['score']:.4f} "
            f"{status}"
        )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    top1_accuracy = (
        top1_correct_count / total
        if total > 0
        else 0.0
    )

    top5_accuracy = (
        top5_correct_count / total
        if total > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Print Summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("Evaluation Summary")
    print("=" * 60)

    print(
        f"Top-1 Accuracy: "
        f"{top1_accuracy:.2%}"
    )

    print(
        f"Top-5 Accuracy: "
        f"{top5_accuracy:.2%}"
    )

    print()
    print("Per Product Top-1 Accuracy")
    print("-" * 60)

    for product_id in sorted(product_total):

        total_count = product_total[product_id]

        correct_count = product_correct.get(
            product_id,
            0,
        )

        accuracy = (
            correct_count / total_count
        )

        print(
            f"{product_id:<20} "
            f"{correct_count}/{total_count} "
            f"({accuracy:.2%})"
        )

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()