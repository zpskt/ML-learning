from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import AutoModel


# =========================
# Configuration
# =========================

MODEL_NAME = "facebook/dinov2-base"

CHECKPOINT_PATH = Path(
    "./checkpoints_metric/latest.pth"
)

FEATURE_LIBRARY_PATH = Path(
    "./features_dinov2_metric.pt"
)

BATCH_SIZE = 32


# =========================
# Device
# =========================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# =========================
# Search Engine
# =========================

class DINOv2MetricSearch:

    def __init__(
            self,
            checkpoint_path=CHECKPOINT_PATH,
            feature_library_path=FEATURE_LIBRARY_PATH,
            batch_size=BATCH_SIZE,
    ):
        self.device = get_device()
        self.batch_size = batch_size

        print(f"Device: {self.device}")

        self.transform = self._build_transform()

        self.model = self._load_model(
            checkpoint_path
        )

        (
            self.library_embeddings,
            self.library_image_paths,
            self.library_product_ids,
        ) = self._load_feature_library(
            feature_library_path
        )

        print(
            f"Feature library: "
            f"{self.library_embeddings.shape}"
        )

    # =========================
    # Model
    # =========================

    @staticmethod
    def _build_transform():

        return transforms.Compose([
            transforms.Resize((224, 224)),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _load_model(self, checkpoint_path):

        model = AutoModel.from_pretrained(
            MODEL_NAME
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model = model.to(self.device)

        model.eval()

        return model

    # =========================
    # Feature Library
    # =========================

    def _load_feature_library(
            self,
            feature_library_path,
    ):

        feature_data = torch.load(
            feature_library_path,
            map_location="cpu",
            weights_only=False,
        )

        embeddings = feature_data["embeddings"]

        image_paths = feature_data[
            "image_paths"
        ]

        product_ids = feature_data[
            "product_ids"
        ]

        embeddings = embeddings.to(
            self.device
        )

        return (
            embeddings,
            image_paths,
            product_ids,
        )

    # =========================
    # Image Loading
    # =========================

    def _load_image(self, image_path):

        image = Image.open(
            image_path
        ).convert("RGB")

        image = self.transform(image)

        return image

    # =========================
    # Batch Embedding
    # =========================

    @torch.inference_mode()
    def _extract_embeddings(
            self,
            image_paths,
    ):

        embeddings = []

        for start in range(
            0,
            len(image_paths),
            self.batch_size,
        ):

            batch_paths = image_paths[
                start:start + self.batch_size
            ]

            images = []

            for image_path in batch_paths:

                image = self._load_image(
                    image_path
                )

                images.append(image)

            batch = torch.stack(
                images
            ).to(self.device)

            output = self.model(
                pixel_values=batch
            )

            batch_embeddings = (
                output.last_hidden_state[:, 0]
            )

            batch_embeddings = F.normalize(
                batch_embeddings,
                p=2,
                dim=1,
            )

            embeddings.append(
                batch_embeddings
            )

        return torch.cat(
            embeddings,
            dim=0,
        )

    # =========================
    # Search
    # =========================

    @torch.inference_mode()
    def search(
            self,
            image_paths,
            top_k=5,
    ):
        """
        Search one or multiple images.

        Args:
            image_paths:
                str / Path / list[str] / list[Path]

            top_k:
                Number of results per query.

        Returns:
            list[list[dict]]
        """

        if isinstance(
                image_paths,
                (str, Path),
        ):
            image_paths = [
                image_paths
            ]

        image_paths = [
            Path(path)
            for path in image_paths
        ]

        if not image_paths:
            return []

        query_embeddings = (
            self._extract_embeddings(
                image_paths
            )
        )

        # [N, 768] @ [768, 4460]
        # -> [N, 4460]
        similarities = torch.matmul(
            query_embeddings,
            self.library_embeddings.T,
        )

        top_k = min(
            top_k,
            self.library_embeddings.shape[0],
        )

        values, indices = torch.topk(
            similarities,
            k=top_k,
            dim=1,
        )

        results = []

        for query_index in range(
            len(image_paths)
        ):

            query_results = []

            for rank in range(top_k):

                library_index = indices[
                    query_index,
                    rank,
                ].item()

                similarity = values[
                    query_index,
                    rank,
                ].item()

                query_results.append({
                    "similarity": similarity,

                    "product_id":
                        self.library_product_ids[
                            library_index
                        ],

                    "image_path":
                        self.library_image_paths[
                            library_index
                        ],
                })

            results.append(
                query_results
            )

        return results


# =========================
# Example
# =========================

if __name__ == "__main__":

    search_engine = (
        DINOv2MetricSearch()
    )

    image_paths = [
        "./test_images/东方树叶/131_786.jpg",
        "./test_images/东鹏特饮/136_125.jpg",
        "./test_images/雪碧/9_31.jpg",
    ]

    results = search_engine.search(
        image_paths,
        top_k=5,
    )

    for image_path, query_results in zip(
            image_paths,
            results,
    ):

        print()
        print("=" * 70)
        print(f"Query: {image_path}")
        print("=" * 70)

        for rank, result in enumerate(
                query_results,
                start=1,
        ):

            print(
                f"Top-{rank} | "
                f"Similarity: "
                f"{result['similarity']:.4f} | "
                f"Product: "
                f"{result['product_id']} | "
                f"Image: "
                f"{result['image_path']}"
            )