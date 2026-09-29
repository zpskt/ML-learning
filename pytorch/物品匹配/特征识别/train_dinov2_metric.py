import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoModel

# ============================================================
# Config
# ============================================================

DATA_DIR = Path("feature_library")

MODEL_NAME = "facebook/dinov2-base"
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
CHECKPOINT_DIR = Path("checkpoints_metric")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "best.pth"
CHECKPOINT_PATH = CHECKPOINT_DIR / "latest.pth"
BEST_PATH = CHECKPOINT_DIR / "best.pth"

LEARNING_RATE = 1e-5
MARGIN = 0.2
SAMPLES_PER_EPOCH = 1000
NUM_EPOCHS = 200
BATCH_SIZE = 16

# ============================================================
# Load Dataset
# ============================================================

def load_dataset(data_dir):
    class_to_images = {}

    for class_dir in sorted(data_dir.iterdir()):

        if not class_dir.is_dir():
            continue

        image_paths = []

        for image_path in sorted(class_dir.iterdir()):

            if image_path.suffix.lower() not in {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp",
            }:
                continue

            image_paths.append(image_path)

        if image_paths:
            class_to_images[class_dir.name] = image_paths

    return class_to_images


def load_image(image_path, transform=None):
    image = Image.open(image_path).convert("RGB")
    if transform is not None:
        image = transform(image)
    return image


class TripletDataset(Dataset):
    def __init__(
            self,
            class_to_images,
            transform=None,
            samples_per_epoch=10000,
    ):
        self.class_to_images = class_to_images
        self.transform = transform
        self.classes = list(class_to_images.keys())
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, index):
        anchor_class = random.choice(self.classes)
        images = self.class_to_images[anchor_class]
        anchor, positive = random.sample(images, 2)
        negative_class = random.choice([
            class_name for class_name in self.classes if class_name != anchor_class
        ])
        negative = random.choice(self.class_to_images[negative_class])

        return (load_image(image_path=anchor, transform=self.transform),
                load_image(image_path=positive, transform=self.transform),
                load_image(image_path=negative, transform=self.transform))


# 创建transform
def build_transform():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
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


# create model with train mode
def build_model_train(device):
    model = AutoModel.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.train()
    return model


# get embedding
def extract_embedding(model, image, ):
    output = model(pixel_values=image)
    embedding = (output.last_hidden_state[:, 0])
    return embedding

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    global_step,
    best_val_metric=None,
):
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,

        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),

        "best_val_metric": best_val_metric,

        "model_name": MODEL_NAME,
        "embedding_dim": 768,

        "image_size": 224,

        "normalize_mean": [
            0.485,
            0.456,
            0.406,
        ],

        "normalize_std": [
            0.229,
            0.224,
            0.225,
        ],

        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "margin": MARGIN,
        "samples_per_epoch": SAMPLES_PER_EPOCH,
    }

    torch.save(checkpoint, path)

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    class_to_images = load_dataset(
        DATA_DIR
    )

    for class_name, image_paths in class_to_images.items():
        print(
            f"{class_name:<20} "
            f"{len(image_paths)} images"
        )

    transform = build_transform()
    dataset = TripletDataset(class_to_images, samples_per_epoch=1000, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    print("Dataset size:", len(dataset))
    model = build_model_train(device)
    # 定义迭代器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # 定义loss计算方式
    criterion = nn.TripletMarginWithDistanceLoss(
        distance_function=lambda x, y: 1 - F.cosine_similarity(x, y),
        margin=MARGIN
    )
    #---------------------------------
    # ---------尝试断点恢复-------------------
    #---------------------------------
    start_epoch = 0
    global_step = 0
    best_val_metric = None

    if CHECKPOINT_PATH.exists():
        print(f"Loading checkpoint: {CHECKPOINT_PATH}")

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
        best_val_metric = checkpoint["best_val_metric"]

        print(
            f"Resume from epoch {start_epoch}, "
            f"global_step {global_step}"
        )

    for epoch in range(start_epoch,NUM_EPOCHS,):
        model.train()

        total_loss = 0.0
        total_positive_similarity = 0.0
        total_negative_similarity = 0.0
        num_batches = 0
        for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
            images = torch.cat(
                [
                    anchor,
                    positive,
                    negative,
                ],
                dim=0,
            )
            images = images.to(device)

            # --------------------------------------------------------
            # 3. 前向传播
            # --------------------------------------------------------

            optimizer.zero_grad()

            output = model(
                pixel_values=images
            )

            embeddings = output.last_hidden_state[:, 0]

            # --------------------------------------------------------
            # 4. 拆分 embedding
            # --------------------------------------------------------
            batch_size = anchor.size(0)
            anchor_embedding = embeddings[:batch_size]
            positive_embedding = embeddings[
                batch_size:batch_size * 2
            ]
            negative_embedding = embeddings[
                batch_size * 2:
            ]
            # --------------------------------------------------------
            # 5. 计算 Similarity
            # --------------------------------------------------------
            positive_similarity = F.cosine_similarity(
                anchor_embedding,
                positive_embedding,
                dim=1,
            )

            negative_similarity = F.cosine_similarity(
                anchor_embedding,
                negative_embedding,
                dim=1,
            )
            # ------------------------------------------------------------
            # Triplet Loss
            # ------------------------------------------------------------
            loss = criterion(
                anchor_embedding,
                positive_embedding,
                negative_embedding,
            )

            # --------------------------------------------------------
            # 7. 反向传播
            # --------------------------------------------------------
            loss.backward()
            # --------------------------------------------------------
            # 8. 更新模型
            # --------------------------------------------------------

            optimizer.step()
            # ----------------------------------------------------
            # 9. Statistics
            # ----------------------------------------------------

            total_loss += loss.item()

            total_positive_similarity += (
                positive_similarity.mean().item()
            )

            total_negative_similarity += (
                negative_similarity.mean().item()
            )

            num_batches += 1

        # --------------------------------------------------------
        # Epoch Statistics
        # --------------------------------------------------------

        avg_loss = total_loss / num_batches

        avg_positive_similarity = (
                total_positive_similarity / num_batches
        )

        avg_negative_similarity = (
                total_negative_similarity / num_batches
        )

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
            f"Loss: {avg_loss:.4f} "
            f"AP: {avg_positive_similarity:.4f} "
            f"AN: {avg_negative_similarity:.4f}"
        )
        # --------------------------------------------------------
        # 保存模型
        # --------------------------------------------------------
        save_checkpoint(
            path=CHECKPOINT_PATH,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            best_val_metric=best_val_metric,
        )

        print(
            f"Checkpoint saved: {CHECKPOINT_PATH}"
        )
