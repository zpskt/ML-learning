#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ML-learning 
@File    ：infer_rtdetr.py.py
@IDE     ：PyCharm 
@Author  ：张鹏
@Date    ：2026/9/5 09:13 
@Description： 
'''

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
)


# ============================================================
# 配置
# ============================================================

# 和训练时使用的预训练模型保持一致
PRETRAINED_MODEL = "PekingU/rtdetr_r50vd_coco_o365"

# 训练得到的最佳模型
CHECKPOINT_PATH = Path("./checkpoints/best.pth")

# 输入图片目录
INPUT_DIR = Path("./test_images")

# 检测目标保存目录
OUTPUT_DIR = Path("./crops")

# Batch 推理，一次处理多少张图片
BATCH_SIZE = 4

# 置信度阈值
CONFIDENCE_THRESHOLD = 0.05

# 你的 6 个类别
#
# 注意：
# 你训练时 category_to_label 如果是 enumerate(categories)，
# 那么对应关系就是：
#
# 0 -> objects
# 1 -> beer
# 2 -> coca-cola
# 3 -> fanta
# 4 -> sprite
# 5 -> waterbottle
#
# 这里必须和训练时保持完全一致。
ID_TO_LABEL = {
    0: "objects",
    1: "beer",
    2: "coca-cola",
    3: "fanta",
    4: "sprite",
    5: "waterbottle",
}

NUM_CLASSES = len(ID_TO_LABEL)


# ============================================================
# Device
# ============================================================

def get_device():
    """
    当前模型已经使用 CPU 训练，因此这里默认使用 CPU。

    如果以后把模型放到 CUDA 机器，
    可以自动使用 CUDA。
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# 加载模型
# ============================================================

def load_model(checkpoint_path):
    device = get_device()

    print("=" * 60)
    print("RT-DETR Inference")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Num classes: {NUM_CLASSES}")

    # Image Processor
    processor = AutoImageProcessor.from_pretrained(
        PRETRAINED_MODEL,
        use_fast=False,
    )

    # 创建模型结构
    model = AutoModelForObjectDetection.from_pretrained(
        PRETRAINED_MODEL,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )

    # 加载训练 checkpoint
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    # 兼容训练脚本保存的 checkpoint
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    print("Model loaded successfully.")

    return model, processor, device


# ============================================================
# 获取图片
# ============================================================

def get_image_paths(input_dir):
    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    image_paths = []

    for path in input_dir.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() in extensions:
            image_paths.append(path)

    image_paths.sort()

    return image_paths


# ============================================================
# Batch 推理
# ============================================================

@torch.inference_mode()
def infer_batch(
    model,
    processor,
    images,
    device,
    confidence_threshold,
):
    """
    一次推理多张图片。

    images:
        List[PIL.Image]

    返回：
        List[dict]
    """

    # 图片预处理
    inputs = processor(
        images=images,
        return_tensors="pt",
    )

    pixel_values = inputs["pixel_values"].to(device)

    # RT-DETR forward
    outputs = model(
        pixel_values=pixel_values,
    )

    # 原始图片尺寸
    #
    # PIL:
    # image.size = (width, height)
    #
    # post_process 需要：
    # (height, width)
    target_sizes = torch.tensor(
        [
            [image.height, image.width]
            for image in images
        ],
        device=device,
    )

    # 后处理：
    # 将模型输出转换成真实图片坐标下的 bbox
    results = processor.post_process_object_detection(
        outputs,
        threshold=confidence_threshold,
        target_sizes=target_sizes,
    )

    return results


# ============================================================
# 单张图片切图
# ============================================================

def crop_objects(
    image,
    image_path,
    result,
    output_dir,
):
    """
    根据 RT-DETR 检测结果切目标。

    每张原图建立一个独立目录：

    crops/
        image001/
            image001_obj000_beer_0.952.jpg
            image001_obj001_sprite_0.871.jpg
    """

    boxes = result["boxes"]
    scores = result["scores"]
    labels = result["labels"]

    # 当前图片对应的输出目录
    image_output_dir = output_dir / image_path.stem

    image_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    crop_count = 0

    for index, (box, score, label) in enumerate(
        zip(boxes, scores, labels)
    ):
        label_id = int(label.item())
        score_value = float(score.item())

        # bbox:
        # x1, y1, x2, y2
        x1, y1, x2, y2 = box.tolist()

        # 防止 bbox 超出图片范围
        x1 = max(0, min(int(x1), image.width))
        y1 = max(0, min(int(y1), image.height))
        x2 = max(0, min(int(x2), image.width))
        y2 = max(0, min(int(y2), image.height))

        # 无效 bbox
        if x2 <= x1 or y2 <= y1:
            continue

        # 切图
        crop = image.crop(
            (x1, y1, x2, y2)
        )

        # 类别名称
        class_name = ID_TO_LABEL.get(
            label_id,
            f"class_{label_id}",
        )

        # 文件名
        crop_name = (
            f"{image_path.stem}"
            f"_obj{index:03d}"
            f"_{class_name}"
            f"_{score_value:.3f}"
            ".jpg"
        )

        crop_path = (
            image_output_dir / crop_name
        )

        crop.save(
            crop_path,
            quality=95,
        )

        crop_count += 1

    return crop_count


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="RT-DETR batch inference and object cropping"
    )

    parser.add_argument(
        "--input",
        type=str,
        default=str(INPUT_DIR),
        help="输入图片目录",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_DIR),
        help="目标切图输出目录",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINT_PATH),
        help="best.pth 路径",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch 推理大小",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help="检测置信度阈值",
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    checkpoint_path = Path(args.checkpoint)

    # --------------------------------------------------------
    # 参数检查
    # --------------------------------------------------------

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}"
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    if args.batch_size <= 0:
        raise ValueError(
            "batch-size must be greater than 0"
        )

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError(
            "threshold must be between 0 and 1"
        )

    # --------------------------------------------------------
    # 加载模型
    # --------------------------------------------------------

    model, processor, device = load_model(
        checkpoint_path
    )

    # --------------------------------------------------------
    # 获取图片
    # --------------------------------------------------------

    image_paths = get_image_paths(
        input_dir
    )

    if not image_paths:
        print(
            f"No images found in: {input_dir}"
        )
        return

    print(f"Found {len(image_paths)} images.")
    print(f"Batch size: {args.batch_size}")
    print(
        f"Confidence threshold: {args.threshold}"
    )
    print()

    # --------------------------------------------------------
    # Batch 推理
    # --------------------------------------------------------

    total_objects = 0

    for start in range(
        0,
        len(image_paths),
        args.batch_size,
    ):
        batch_paths = image_paths[
            start:start + args.batch_size
        ]

        images = []
        valid_paths = []

        # 读取图片
        for image_path in batch_paths:
            try:
                image = Image.open(
                    image_path
                ).convert("RGB")

                images.append(image)
                valid_paths.append(
                    image_path
                )

            except Exception as e:
                print(
                    f"[Skip] {image_path}: {e}"
                )

        if not images:
            continue

        # Batch 推理
        results = infer_batch(
            model=model,
            processor=processor,
            images=images,
            device=device,
            confidence_threshold=args.threshold,
        )

        # 每张图片分别切目标
        for image, image_path, result in zip(
            images,
            valid_paths,
            results,
        ):
            count = crop_objects(
                image=image,
                image_path=image_path,
                result=result,
                output_dir=output_dir,
            )

            total_objects += count

            print(
                f"[{start + len(valid_paths)}/"
                f"{len(image_paths)}] "
                f"{image_path.name}: "
                f"{count} objects"
            )

    print()
    print("=" * 60)
    print("Inference finished")
    print(f"Images: {len(image_paths)}")
    print(f"Objects: {total_objects}")
    print(f"Crops: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
