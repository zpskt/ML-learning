from pathlib import Path

import torch

from search_features_dinov2_metric import DINOv2MetricSearch


# =========================
# Configuration
# =========================

TEST_IMAGE_DIR = Path("./test_images")

TOP_K = 10


# =========================
# Dataset
# =========================

def load_test_images(test_image_dir):
    """
    Load all test images.

    Returns:
        list[tuple[Path, str]]
        (image_path, ground_truth_product_id)
    """

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    samples = []

    for product_dir in sorted(
            test_image_dir.iterdir()
    ):
        if not product_dir.is_dir():
            continue

        product_id = product_dir.name

        for image_path in sorted(
                product_dir.iterdir()
        ):
            if image_path.suffix.lower() not in image_extensions:
                continue

            samples.append(
                (
                    image_path,
                    product_id,
                )
            )

    return samples


# =========================
# Similarity Metrics
# =========================

def calculate_similarity_metrics(
        embeddings,
        product_ids,
):
    """
    Calculate:

        Intra-class Similarity
        Inter-class Similarity
        Separation Gap

    Args:
        embeddings:
            Tensor [N, D], L2 normalized.

        product_ids:
            list[str], ground-truth labels.

    Returns:
        dict
    """

    # [N, D] @ [D, N]
    # -> [N, N]
    similarity_matrix = torch.matmul(
        embeddings,
        embeddings.T,
    )

    product_ids = list(product_ids)

    product_ids_tensor = torch.tensor([
        hash(product_id)
        for product_id in product_ids
    ])

    same_class = (
        product_ids_tensor.unsqueeze(0)
        == product_ids_tensor.unsqueeze(1)
    )

    different_class = ~same_class

    # Remove self-similarity.
    identity = torch.eye(
        len(product_ids),
        dtype=torch.bool,
        device=embeddings.device,
    )

    same_class = (
        same_class.to(embeddings.device)
        & ~identity
    )

    intra_similarity = (
        similarity_matrix[same_class]
        .mean()
        .item()
    )

    inter_similarity = (
        similarity_matrix[different_class]
        .mean()
        .item()
    )

    separation_gap = (
        intra_similarity
        - inter_similarity
    )

    return {
        "intra": intra_similarity,
        "inter": inter_similarity,
        "gap": separation_gap,
    }


# =========================
# Evaluation
# =========================

def evaluate(
        search_engine,
        samples,
        top_k=10,
):
    """
    Evaluate retrieval accuracy and
    embedding-space similarity metrics.
    """

    image_paths = [
        image_path
        for image_path, _ in samples
    ]

    ground_truths = [
        product_id
        for _, product_id in samples
    ]

    print(
        f"Total test images: {len(samples)}"
    )

    # ---------------------------------
    # Extract query embeddings
    # ---------------------------------

    embeddings = (
        search_engine._extract_embeddings(
            image_paths
        )
    )

    # ---------------------------------
    # Retrieval
    # ---------------------------------

    results = search_engine.search(
        image_paths,
        top_k=top_k,
    )

    top1_correct = 0
    recall5_correct = 0
    recall10_correct = 0

    for ground_truth, query_results in zip(
            ground_truths,
            results,
    ):

        predicted_product_ids = [
            result["product_id"]
            for result in query_results
        ]

        if predicted_product_ids[0] == ground_truth:
            top1_correct += 1

        if ground_truth in predicted_product_ids[:5]:
            recall5_correct += 1

        if ground_truth in predicted_product_ids[:10]:
            recall10_correct += 1

    total = len(samples)

    # ---------------------------------
    # Similarity metrics
    # ---------------------------------

    similarity_metrics = (
        calculate_similarity_metrics(
            embeddings,
            ground_truths,
        )
    )

    metrics = {
        "top1": top1_correct / total,
        "recall@5": recall5_correct / total,
        "recall@10": recall10_correct / total,
        "intra": similarity_metrics["intra"],
        "inter": similarity_metrics["inter"],
        "gap": similarity_metrics["gap"],
    }

    return metrics


# =========================
# Main
# =========================

def main():

    samples = load_test_images(
        TEST_IMAGE_DIR
    )

    if not samples:
        raise RuntimeError(
            f"No test images found: "
            f"{TEST_IMAGE_DIR}"
        )

    search_engine = (
        DINOv2MetricSearch()
    )

    metrics = evaluate(
        search_engine,
        samples,
        top_k=TOP_K,
    )

    print()
    print("=" * 70)
    print("Evaluation Results")
    print("=" * 70)

    print(
        f"Model                 : "
        f"DINOv2-Metric"
    )

    print(
        f"Test Images           : "
        f"{len(samples)}"
    )

    print(
        f"Gallery Images        : "
        f"{search_engine.library_embeddings.shape[0]}"
    )

    print(
        f"Embedding Dimension   : "
        f"{search_engine.library_embeddings.shape[1]}"
    )

    print()

    print(
        f"Top-1 Accuracy        : "
        f"{metrics['top1'] * 100:.2f}%"
    )

    print(
        f"Recall@5              : "
        f"{metrics['recall@5'] * 100:.2f}%"
    )

    print(
        f"Recall@10             : "
        f"{metrics['recall@10'] * 100:.2f}%"
    )

    print()

    print(
        f"Intra-class Similarity: "
        f"{metrics['intra']:.4f}"
    )

    print(
        f"Inter-class Similarity: "
        f"{metrics['inter']:.4f}"
    )

    print(
        f"Separation Gap        : "
        f"{metrics['gap']:.4f}"
    )


if __name__ == "__main__":
    main()