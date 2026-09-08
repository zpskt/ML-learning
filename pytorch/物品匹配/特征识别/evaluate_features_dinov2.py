#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Project ：图像算法
@File    ：evaluate_features_dinov2.py
@IDE     ：PyCharm
@Author  ：张鹏
@Date    ：2026/9/6
@Description：
    使用 test_images 评估 DINOv2 embedding 检索效果

    Evaluation Metrics:
    1. Top-1 Accuracy
    2. Recall@5
    3. Recall@10
    4. Per Product Top-1 Accuracy
    5. Confusion Matrix
    6. Intra-class Similarity
    7. Inter-class Similarity
    8. Separation Gap
"""

from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms
from transformers import AutoModel
CHECKPOINT_PATH = Path("./checkpoints/dinov2_best.pth")


# ============================================================
# Config
# ============================================================

TEST_DIR = Path("./test_images")

FEATURES_PATH = Path("./features_dinov2.pt")

MODEL_NAME = "facebook/dinov2-base"

# Retrieval metrics
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
# Transform
# ============================================================

class ResizePad:

    def __init__(
        self,
        size=224,
        fill=0,
    ):
        self.size = size
        self.fill = fill

    def __call__(self, image):

        # 保持原始宽高比缩放
        image = ImageOps.contain(
            image,
            (self.size, self.size),
        )

        # Padding 到固定尺寸
        image = ImageOps.pad(
            image,
            (self.size, self.size),
            method=Image.Resampling.BICUBIC,
            color=self.fill,
            centering=(0.5, 0.5),
        )

        return image


def build_transform():

    transform = transforms.Compose([
        ResizePad(224),

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

    return transform


# ============================================================
# Model
# ============================================================

def build_model(device):
    #创建原始back bone
    model = AutoModel.from_pretrained(MODEL_NAME)
    # 2. 加载 Fine-tuning 后的 checkpoint，先加载到内存李米娜
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    state_dict = checkpoint["model_state_dict"]

    # 3. checkpoint 里面包含：
    #    backbone.xxx
    #    classifier.xxx
    #
    #    我们这里只取 backbone
    backbone_state_dict = {
        key[len("backbone."):]: value
        for key, value in state_dict.items()
        if key.startswith("backbone.")
    }

    # 4. 把 Fine-tuning 后的权重加载进 DINOv2
    model.load_state_dict(
        backbone_state_dict,
        strict=True,
    )

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

    for product_dir in sorted(
        test_dir.iterdir()
    ):

        if not product_dir.is_dir():
            continue

        product_id = product_dir.name

        for image_path in sorted(
            product_dir.rglob("*")
        ):

            if (
                image_path.suffix.lower()
                not in image_extensions
            ):
                continue

            image_paths.append(
                image_path
            )

            ground_truths.append(
                product_id
            )

    return (
        image_paths,
        ground_truths,
    )


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

    image = transform(
        image
    )

    image = image.unsqueeze(
        0
    ).to(device)

    outputs = model(
        pixel_values=image
    )

    # --------------------------------------------------------
    # DINOv2 CLS Token
    # --------------------------------------------------------

    embedding = (
        outputs.last_hidden_state[:, 0]
    )

    # --------------------------------------------------------
    # L2 Normalize
    # --------------------------------------------------------

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

    product_to_indices = defaultdict(list)

    for index, product_id in enumerate(
        database_product_ids
    ):

        product_to_indices[
            product_id
        ].append(index)

    return product_to_indices


# ============================================================
# Evaluate One Image
# ============================================================

def evaluate_one(
    model,
    transform,
    database_embeddings,
    database_product_ids,
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
    # Image-level similarity
    #
    # database_embeddings:
    #     [N, 768]
    #
    # query_embedding:
    #     [1, 768]
    #
    # result:
    #     [N]
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
    # 这里采用：
    #
    # Product Score =
    #     该商品所有图片中最大的 similarity
    # --------------------------------------------------------

    product_scores = {}

    for product_id, indices in (
        product_to_indices.items()
    ):

        product_similarity = (
            similarities[indices].max()
        )

        product_scores[
            product_id
        ] = product_similarity.item()

    # --------------------------------------------------------
    # Product ranking
    # --------------------------------------------------------

    sorted_products = sorted(
        product_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    predictions = []

    for product_id, score in sorted_products:

        predictions.append({
            "product_id": product_id,
            "score": score,
        })

    # --------------------------------------------------------
    # Top-K
    # --------------------------------------------------------

    top1_correct = (
        predictions[0]["product_id"]
        == ground_truth
    )

    top5_correct = any(
        result["product_id"]
        == ground_truth
        for result in predictions[:5]
    )

    top10_correct = any(
        result["product_id"]
        == ground_truth
        for result in predictions[:10]
    )

    return (
        top1_correct,
        top5_correct,
        top10_correct,
        predictions,
        query_embedding.squeeze(0),
    )


# ============================================================
# Calculate Embedding Statistics
# ============================================================

def calculate_embedding_statistics(
    embeddings,
    labels,
):

    embeddings = torch.stack(
        embeddings
    )

    # --------------------------------------------------------
    # Similarity Matrix
    # --------------------------------------------------------

    similarity_matrix = torch.matmul(
        embeddings,
        embeddings.T,
    )

    intra_similarities = []

    inter_similarities = []

    n = len(labels)

    for i in range(n):

        for j in range(i + 1, n):

            similarity = (
                similarity_matrix[i, j]
                .item()
            )

            if labels[i] == labels[j]:

                intra_similarities.append(
                    similarity
                )

            else:

                inter_similarities.append(
                    similarity
                )

    # --------------------------------------------------------
    # Mean
    # --------------------------------------------------------

    intra_mean = (
        sum(intra_similarities)
        / len(intra_similarities)
        if intra_similarities
        else 0.0
    )

    inter_mean = (
        sum(inter_similarities)
        / len(inter_similarities)
        if inter_similarities
        else 0.0
    )

    separation_gap = (
        intra_mean
        - inter_mean
    )

    return (
        intra_mean,
        inter_mean,
        separation_gap,
    )


# ============================================================
# Print Confusion Matrix
# ============================================================

def print_confusion_matrix(
    confusion_matrix,
    product_ids,
):

    print()

    print("=" * 80)
    print("Confusion Matrix")
    print("=" * 80)

    print(
        "Rows = Ground Truth"
    )

    print(
        "Columns = Prediction"
    )

    print()

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print(
        f"{'GT / Pred':<20}",
        end="",
    )

    for product_id in product_ids:

        print(
            f"{product_id:<12}",
            end="",
        )

    print()

    print("-" * 80)

    # --------------------------------------------------------
    # Rows
    # --------------------------------------------------------

    for gt in product_ids:

        print(
            f"{gt:<20}",
            end="",
        )

        for pred in product_ids:

            count = confusion_matrix[
                gt
            ].get(
                pred,
                0,
            )

            print(
                f"{count:<12}",
                end="",
            )

        print()


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)

    print(
        "DINOv2 Feature Evaluation"
    )

    print("=" * 60)

    device = get_device()

    print(
        f"Device: {device}"
    )

    print(
        f"Test directory: {TEST_DIR}"
    )

    print(
        f"Feature database: {FEATURES_PATH}"
    )

    # --------------------------------------------------------
    # Check test directory
    # --------------------------------------------------------

    if not TEST_DIR.exists():

        raise FileNotFoundError(
            f"Test directory not found: "
            f"{TEST_DIR}"
        )

    # --------------------------------------------------------
    # Load test images
    # --------------------------------------------------------

    (
        image_paths,
        ground_truths,
    ) = load_test_images(
        TEST_DIR
    )

    if not image_paths:

        raise RuntimeError(
            "No test images found."
        )

    print(
        f"Test images: "
        f"{len(image_paths)}"
    )

    # --------------------------------------------------------
    # Load feature database
    # --------------------------------------------------------

    feature_data = torch.load(
        FEATURES_PATH,
        map_location="cpu",
        weights_only=False,
    )

    database_embeddings = (
        feature_data["embeddings"]
    )

    database_product_ids = (
        feature_data["product_ids"]
    )

    # --------------------------------------------------------
    # Normalize database
    # --------------------------------------------------------

    database_embeddings = (
        F.normalize(
            database_embeddings,
            p=2,
            dim=1,
        )
        .to(device)
    )

    print(
        f"Database embeddings: "
        f"{database_embeddings.shape}"
    )

    # --------------------------------------------------------
    # Build product database
    # --------------------------------------------------------

    product_to_indices = (
        build_product_database(
            database_embeddings=
                database_embeddings,
            database_product_ids=
                database_product_ids,
        )
    )

    product_ids = sorted(
        product_to_indices.keys()
    )

    print(
        f"Database products: "
        f"{len(product_ids)}"
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model, transform = (
        build_model(device)
    )

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        "Model loaded."
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    top1_correct_count = 0

    top5_correct_count = 0

    top10_correct_count = 0

    successful_count = 0

    # Product statistics

    product_total = defaultdict(int)

    product_correct = defaultdict(int)

    # Confusion Matrix

    confusion_matrix = defaultdict(
        lambda: defaultdict(int)
    )

    # Embeddings

    test_embeddings = []

    test_labels = []

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    print()

    print("=" * 60)

    print("Results")

    print("=" * 60)

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
                top1_correct,
                top5_correct,
                top10_correct,
                predictions,
                query_embedding,
            ) = evaluate_one(
                model=model,
                transform=transform,
                database_embeddings=
                    database_embeddings,
                database_product_ids=
                    database_product_ids,
                product_to_indices=
                    product_to_indices,
                image_path=image_path,
                ground_truth=ground_truth,
                device=device,
            )

        except Exception as e:

            print(
                f"[ERROR] "
                f"{image_path}: {e}"
            )

            continue

        # ----------------------------------------------------
        # Successful evaluation
        # ----------------------------------------------------

        successful_count += 1

        # ----------------------------------------------------
        # Overall statistics
        # ----------------------------------------------------

        if top1_correct:

            top1_correct_count += 1

        if top5_correct:

            top5_correct_count += 1

        if top10_correct:

            top10_correct_count += 1

        # ----------------------------------------------------
        # Product statistics
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

        predicted_product = (
            predictions[0][
                "product_id"
            ]
        )

        confusion_matrix[
            ground_truth
        ][
            predicted_product
        ] += 1

        # ----------------------------------------------------
        # Save embeddings
        # ----------------------------------------------------

        test_embeddings.append(
            query_embedding.cpu()
        )

        test_labels.append(
            ground_truth
        )

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        top1 = predictions[0]

        status = (
            "OK"
            if top1_correct
            else "WRONG"
        )

        print(
            f"[{i}/{len(image_paths)}] "
            f"{image_path.name:<35} "
            f"GT={ground_truth:<15} "
            f"Pred={top1['product_id']:<15} "
            f"Score={top1['score']:.4f} "
            f"{status}"
        )

    # ========================================================
    # Accuracy
    # ========================================================

    if successful_count == 0:

        raise RuntimeError(
            "No images were successfully evaluated."
        )

    top1_accuracy = (
        top1_correct_count
        / successful_count
    )

    recall_at_5 = (
        top5_correct_count
        / successful_count
    )

    recall_at_10 = (
        top10_correct_count
        / successful_count
    )

    # ========================================================
    # Embedding Statistics
    # ========================================================

    (
        intra_mean,
        inter_mean,
        separation_gap,
    ) = calculate_embedding_statistics(
        embeddings=test_embeddings,
        labels=test_labels,
    )

    # ========================================================
    # Summary
    # ========================================================

    print()

    print("=" * 60)

    print(
        "Evaluation Summary"
    )

    print("=" * 60)

    print(
        f"Total test images: "
        f"{len(image_paths)}"
    )

    print(
        f"Successfully evaluated: "
        f"{successful_count}"
    )

    print()

    print(
        f"Top-1 Accuracy: "
        f"{top1_accuracy:.2%}"
    )

    print(
        f"Recall@5: "
        f"{recall_at_5:.2%}"
    )

    print(
        f"Recall@10: "
        f"{recall_at_10:.2%}"
    )

    # ========================================================
    # Per Product Accuracy
    # ========================================================

    print()

    print(
        "Per Product Top-1 Accuracy"
    )

    print("-" * 60)

    for product_id in sorted(
        product_total
    ):

        total_count = (
            product_total[
                product_id
            ]
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
            f"{correct_count}/"
            f"{total_count} "
            f"({accuracy:.2%})"
        )

    # ========================================================
    # Confusion Matrix
    # ========================================================

    print_confusion_matrix(
        confusion_matrix=
            confusion_matrix,
        product_ids=product_ids,
    )

    # ========================================================
    # Embedding Statistics
    # ========================================================

    print()

    print("=" * 60)

    print(
        "Embedding Statistics"
    )

    print("=" * 60)

    print(
        f"Intra-class Similarity: "
        f"{intra_mean:.4f}"
    )

    print(
        f"Inter-class Similarity: "
        f"{inter_mean:.4f}"
    )

    print(
        f"Separation Gap: "
        f"{separation_gap:.4f}"
    )

    print()

    print("=" * 60)


if __name__ == "__main__":

    main()