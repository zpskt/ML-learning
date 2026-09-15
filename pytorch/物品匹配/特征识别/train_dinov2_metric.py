import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoModel


# ============================================================
# 1. 配置
# ============================================================

MODEL_NAME = "facebook/dinov2-base"

DATA_DIR = Path("./feature_library")
CHECKPOINT_PATH = Path("./checkpoints/dinov2_best.pth")
OUTPUT_DIR = Path("./checkpoints")

BATCH_SIZE = 32
EPOCHS = 20

BACKBONE_LR = 1e-5
PROJECTION_LR = 1e-4
WEIGHT_DECAY = 1e-4

TEMPERATURE = 0.07

VAL_RATIO = 0.2
SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 2. 固定随机种子
# ============================================================

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. Dataset
# ============================================================

class ProductDataset(Dataset):

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        image = self.transform(image)

        return image, label


# ============================================================
# 4. 构建数据集
# ============================================================

def build_samples():

    product_dirs = [
        p for p in DATA_DIR.iterdir()
        if p.is_dir()
    ]

    product_dirs.sort()

    class_to_idx = {
        product_dir.name: idx
        for idx, product_dir in enumerate(product_dirs)
    }

    samples = []

    for product_dir in product_dirs:

        label = class_to_idx[product_dir.name]

        for image_path in product_dir.iterdir():

            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp",
            }:
                continue

            samples.append(
                (image_path, label)
            )

    print(f"Total images: {len(samples)}")
    print(f"Num classes: {len(class_to_idx)}")

    for name, idx in class_to_idx.items():

        count = sum(
            label == idx
            for _, label in samples
        )

        print(
            f"  {idx}: {name} -> {count}"
        )

    return samples, class_to_idx


# ============================================================
# 5. 按类别划分 train / val
# ============================================================

def split_dataset(samples, num_classes):

    class_samples = {
        i: []
        for i in range(num_classes)
    }

    for sample in samples:

        _, label = sample

        class_samples[label].append(sample)

    train_samples = []
    val_samples = []

    for label, items in class_samples.items():

        random.shuffle(items)

        val_count = max(
            1,
            int(len(items) * VAL_RATIO)
        )

        val_samples.extend(
            items[:val_count]
        )

        train_samples.extend(
            items[val_count:]
        )

    return train_samples, val_samples


# ============================================================
# 6. 数据增强
# ============================================================

def build_transforms():

    train_transform = transforms.Compose([
        transforms.Resize(256),

        transforms.RandomResizedCrop(
            224,
            scale=(0.8, 1.0),
        ),

        transforms.RandomHorizontalFlip(),

        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        ),

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

    val_transform = transforms.Compose([
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

    return train_transform, val_transform


# ============================================================
# 7. DINOv2 + Projection Head
# ============================================================

class DINOv2MetricModel(nn.Module):

    def __init__(self, model_name):

        super().__init__()

        self.backbone = AutoModel.from_pretrained(
            model_name
        )

        hidden_size = self.backbone.config.hidden_size

        # Metric Learning 不直接在 backbone embedding
        # 上计算 Loss，而是增加一个 projection head。
        self.projection = nn.Sequential(
            nn.Linear(
                hidden_size,
                hidden_size,
            ),

            nn.GELU(),

            nn.Linear(
                hidden_size,
                hidden_size,
            ),
        )

    def forward(self, pixel_values):

        outputs = self.backbone(
            pixel_values=pixel_values
        )

        embedding = outputs.last_hidden_state[:, 0]

        projection = self.projection(
            embedding
        )

        # SupCon 使用归一化后的 projection
        projection = F.normalize(
            projection,
            p=2,
            dim=1,
        )

        return embedding, projection


# ============================================================
# 8. Supervised Contrastive Loss
# ============================================================

class SupConLoss(nn.Module):

    def __init__(self, temperature=0.07):

        super().__init__()

        self.temperature = temperature

    def forward(self, features, labels):

        """
        features:
            [B, D]

        labels:
            [B]

        同 label：
            Positive

        不同 label：
            Negative
        """

        device = features.device

        batch_size = features.shape[0]

        # ----------------------------------------------------
        # similarity matrix
        # ----------------------------------------------------

        similarity = torch.matmul(
            features,
            features.T,
        )

        similarity = similarity / self.temperature

        # ----------------------------------------------------
        # 数值稳定性处理
        # ----------------------------------------------------

        logits_max, _ = torch.max(
            similarity,
            dim=1,
            keepdim=True,
        )

        logits = similarity - logits_max.detach()

        # ----------------------------------------------------
        # label mask
        # ----------------------------------------------------

        labels = labels.contiguous().view(-1, 1)

        positive_mask = torch.eq(
            labels,
            labels.T,
        ).float().to(device)

        # 自己不能作为自己的 positive
        self_mask = torch.eye(
            batch_size,
            device=device,
        )

        positive_mask = (
            positive_mask - self_mask
        )

        # ----------------------------------------------------
        # denominator
        # ----------------------------------------------------

        logits_mask = (
            torch.ones_like(
                positive_mask
            ) - self_mask
        )

        exp_logits = torch.exp(logits) * logits_mask

        log_prob = (
            logits
            - torch.log(
                exp_logits.sum(
                    dim=1,
                    keepdim=True,
                ) + 1e-12
            )
        )

        # ----------------------------------------------------
        # positive log probability
        # ----------------------------------------------------

        positive_count = positive_mask.sum(
            dim=1
        )

        mean_log_prob_pos = (
            positive_mask * log_prob
        ).sum(dim=1) / (
            positive_count + 1e-12
        )

        # ----------------------------------------------------
        # 最终 Loss
        # ----------------------------------------------------

        loss = -mean_log_prob_pos.mean()

        return loss


# ============================================================
# 9. Train
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):

    model.train()

    total_loss = 0.0

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad()

        _, projection = model(
            images
        )

        loss = criterion(
            projection,
            labels,
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * images.size(0)
        )

    return total_loss / len(loader.dataset)


# ============================================================
# 10. Validation
# ============================================================

@torch.inference_mode()
def validate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        _, projection = model(
            images
        )

        loss = criterion(
            projection,
            labels,
        )

        total_loss += (
            loss.item()
            * images.size(0)
        )

    return total_loss / len(loader.dataset)


# ============================================================
# 11. Main
# ============================================================

def main():

    set_seed(SEED)

    print("=" * 60)
    print("DINOv2 Metric Learning")
    print("=" * 60)

    print(f"Device: {DEVICE}")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    samples, class_to_idx = build_samples()

    num_classes = len(class_to_idx)

    train_samples, val_samples = split_dataset(
        samples,
        num_classes,
    )

    print()
    print(
        f"Train: {len(train_samples)}"
    )

    print(
        f"Val:   {len(val_samples)}"
    )

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    train_transform, val_transform = (
        build_transforms()
    )

    train_dataset = ProductDataset(
        train_samples,
        train_transform,
    )

    val_dataset = ProductDataset(
        val_samples,
        val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = DINOv2MetricModel(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # 加载之前的 DINOv2 classification fine-tuning
    # --------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    state_dict = checkpoint[
        "model_state_dict"
    ]

    backbone_state_dict = {
        key[len("backbone."):]: value
        for key, value in state_dict.items()
        if key.startswith("backbone.")
    }

    missing, unexpected = (
        model.backbone.load_state_dict(
            backbone_state_dict,
            strict=True,
        )
    )

    print()
    print(
        "Loaded DINOv2 fine-tuning checkpoint."
    )

    model = model.to(DEVICE)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = SupConLoss(
        temperature=TEMPERATURE
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.backbone.parameters(),
                "lr": BACKBONE_LR,
            },

            {
                "params": model.projection.parameters(),
                "lr": PROJECTION_LR,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            DEVICE,
        )

        val_loss = validate(
            model,
            val_loader,
            criterion,
            DEVICE,
        )

        scheduler.step()

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f}"
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            checkpoint = {
                "epoch": epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "scheduler_state_dict":
                    scheduler.state_dict(),

                "best_val_loss":
                    best_val_loss,

                "model_name":
                    MODEL_NAME,

                "embedding_dimension":
                    model.backbone.config.hidden_size,

                "projection_dimension":
                    model.projection[-1].out_features,

                "temperature":
                    TEMPERATURE,

                "class_to_idx":
                    class_to_idx,

                "num_classes":
                    num_classes,
            }

            torch.save(
                checkpoint,
                OUTPUT_DIR
                / "dinov2_metric_best.pth",
            )

            print(
                "  -> Best checkpoint saved."
            )

    # --------------------------------------------------------
    # Last checkpoint
    # --------------------------------------------------------

    torch.save(
        {
            "epoch": EPOCHS,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "best_val_loss":
                best_val_loss,

            "model_name":
                MODEL_NAME,

            "embedding_dimension":
                model.backbone.config.hidden_size,

            "projection_dimension":
                model.projection[-1].out_features,

            "temperature":
                TEMPERATURE,

            "class_to_idx":
                class_to_idx,

            "num_classes":
                num_classes,
        },
        OUTPUT_DIR
        / "dinov2_metric_last.pth",
    )

    print()
    print("=" * 60)
    print("Training finished.")
    print(
        f"Best Val Loss: {best_val_loss:.4f}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()