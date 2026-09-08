#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Project ：_annotations.coco.json
@File    ：build_feature_library_dinov2.py
@IDE     ：PyCharm
@Author  ：张鹏
@Date    ：2026/9/7
@Description：使用预训练 DINOv2 创建图像特征库
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import AutoModel
CHECKPOINT_PATH = Path("./checkpoints/dinov2_best.pth")


# =========================
# 配置
# =========================

FEATURE_LIBRARY_DIR = Path("./feature_library")
OUTPUT_PATH = Path("./features_dinov2.pt")

MODEL_NAME = "facebook/dinov2-base"

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
# 构建 DINOv2
# =========================

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


# =========================
# 加载特征库图片
# =========================

def load_images(root_dir):
    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    image_paths = []
    product_ids = []

    for product_dir in sorted(root_dir.iterdir()):

        if not product_dir.is_dir():
            continue

        product_id = product_dir.name

        for image_path in sorted(product_dir.iterdir()):

            if image_path.suffix.lower() not in image_extensions:
                continue

            image_paths.append(image_path)
            product_ids.append(product_id)

    return image_paths, product_ids


# =========================
# 提取 Batch Embedding
# =========================

@torch.inference_mode()
def extract_batch(
    model,
    transform,
    image_paths,
    device,
):
    images = []
    valid_paths = []

    for image_path in image_paths:

        try:
            image = Image.open(image_path).convert("RGB")

            image = transform(image)

            images.append(image)
            valid_paths.append(image_path)

        except Exception as e:
            print(
                f"[Warning] Failed to load "
                f"{image_path}: {e}"
            )

    if not images:
        return None, []

    batch = torch.stack(images).to(device)

    outputs = model(pixel_values=batch)

    # DINOv2 的 CLS Token 作为整张图片的 embedding
    # embeddings = outputs.last_hidden_state[:, 0]
    patch_embeddings = outputs.last_hidden_state[:, 1:]
    embeddings = patch_embeddings.mean(dim=1)

    # L2 Normalize
    embeddings = F.normalize(
        embeddings,
        p=2,
        dim=1,
    )

    # 保存到 CPU，避免 features.pt 占用 GPU 显存
    embeddings = embeddings.cpu()

    return embeddings, valid_paths


# =========================
# Main
# =========================

def main():

    device = get_device()

    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}")

    # -------------------------
    # 加载模型
    # -------------------------

    model, transform = build_model(device)

    # -------------------------
    # 加载图片
    # -------------------------

    image_paths, product_ids = load_images(
        FEATURE_LIBRARY_DIR
    )

    print(f"Found {len(image_paths)} images.")

    if not image_paths:
        raise RuntimeError(
            f"No images found in "
            f"{FEATURE_LIBRARY_DIR}"
        )

    # -------------------------
    # 提取特征
    # -------------------------

    all_embeddings = []
    all_paths = []
    all_product_ids = []

    total = len(image_paths)

    for start in range(0, total, BATCH_SIZE):

        end = min(
            start + BATCH_SIZE,
            total,
        )

        batch_paths = image_paths[start:end]
        batch_product_ids = product_ids[start:end]

        embeddings, valid_paths = extract_batch(
            model=model,
            transform=transform,
            image_paths=batch_paths,
            device=device,
        )

        if embeddings is None:
            continue

        all_embeddings.append(embeddings)
        all_paths.extend(valid_paths)

        # valid_paths 可能比 batch_paths 少，
        # 因为部分图片可能读取失败。
        valid_path_set = set(valid_paths)

        for image_path, product_id in zip(
            batch_paths,
            batch_product_ids,
        ):
            if image_path in valid_path_set:
                all_product_ids.append(product_id)

        print(
            f"[{end}/{total}] "
            f"processed"
        )

    # -------------------------
    # 合并特征
    # -------------------------

    embeddings = torch.cat(
        all_embeddings,
        dim=0,
    )

    print()
    print("Feature library built.")
    print(f"Images     : {len(all_paths)}")
    print(f"Embedding  : {embeddings.shape}")
    print(f"Dimension  : {embeddings.shape[1]}")

    # -------------------------
    # 保存
    # -------------------------

    feature_data = {
        "embeddings": embeddings,
        "image_paths": all_paths,
        "product_ids": all_product_ids,

        "dimension": embeddings.shape[1],

        "model": MODEL_NAME,
        "model_type": "dinov2",

        "normalized": True,
    }

    torch.save(
        feature_data,
        OUTPUT_PATH,
    )

    print()
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()