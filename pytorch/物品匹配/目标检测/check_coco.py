#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ML-learning 
@File    ：check_coco.py.py
@IDE     ：PyCharm 
@Author  ：张鹏
@Date    ：2026/9/5 01:05 
@Description： 
'''

import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


DATASET_DIR = Path(
    "/Users/zhangpeng/Desktop/workspace/github/ML-learning/pytorch/物品匹配/目标检测/data/Bottle"
)

SPLIT = "train"


def load_coco_dataset(dataset_dir: Path, split: str):
    split_dir = dataset_dir / split

    # 找 JSON
    json_files = list(split_dir.glob("*.json"))

    if len(json_files) != 1:
        raise RuntimeError(
            f"{split_dir} 中应该有且只有一个 JSON 文件，"
            f"实际找到 {len(json_files)} 个"
        )

    annotation_file = json_files[0]

    with open(annotation_file, "r", encoding="utf-8") as f:
        coco = json.load(f)

    return split_dir, coco


def show_image_with_annotations(split_dir: Path, coco: dict, image_index: int = 0):
    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    image_info = images[image_index]

    image_id = image_info["id"]
    image_file = image_info["file_name"]

    image_path = split_dir / image_file

    if not image_path.exists():
        raise FileNotFoundError(f"图片不存在: {image_path}")

    image = Image.open(image_path).convert("RGB")

    # category_id -> category_name
    category_map = {
        category["id"]: category["name"]
        for category in categories
    }

    # 找到当前图片对应的所有 annotation
    image_annotations = [
        ann
        for ann in annotations
        if ann["image_id"] == image_id
    ]

    print("Image:")
    print(f"  file: {image_file}")
    print(f"  image_id: {image_id}")
    print(f"  size: {image.size}")

    print(f"\nAnnotations: {len(image_annotations)}")

    plt.figure(figsize=(10, 8))
    plt.imshow(image)

    ax = plt.gca()

    for ann in image_annotations:
        x, y, w, h = ann["bbox"]

        category_id = ann["category_id"]
        category_name = category_map[category_id]

        print(
            f"  category={category_name}, "
            f"bbox=[{x}, {y}, {w}, {h}]"
        )

        rectangle = plt.Rectangle(
            (x, y),
            w,
            h,
            fill=False,
            linewidth=2,
        )

        ax.add_patch(rectangle)

        ax.text(
            x,
            y,
            category_name,
            fontsize=12,
            bbox=dict(
                facecolor="white",
                alpha=0.7,
            ),
        )

    plt.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    split_dir, coco = load_coco_dataset(
        DATASET_DIR,
        SPLIT,
    )

    print("Categories:")
    for category in coco["categories"]:
        print(
            f"  id={category['id']}, "
            f"name={category['name']}"
        )

    print(f"\nImages: {len(coco['images'])}")
    print(f"Annotations: {len(coco['annotations'])}")

    show_image_with_annotations(
        split_dir,
        coco,
        image_index=0,
    )


if __name__ == "__main__":
    main()