#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ML-learning 
@File    ：download_images.py.py
@IDE     ：PyCharm 
@Author  ：张鹏
@Date    ：2026/9/4 22:32 
@Description： 
'''

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import requests
from PIL import Image
import serpapi


IMAGE_EXTENSIONS = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Download product images from Google Images via SerpApi."
    )

    parser.add_argument(
        "category",
        type=str,
        default="百事可乐",
        help="Product category, e.g. 可口可乐"
    )

    parser.add_argument(
        "--num",
        type=int,
        default=20,
        help="Number of images to download. Default: 20"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/products",
        help="Output root directory. Default: data/products"
    )

    parser.add_argument(
        "--location",
        type=str,
        default="Tokyo, Japan",
        help="Google search location. Default: Tokyo, Japan"
    )

    return parser


def get_api_key():
    api_key = os.getenv("SERPAPI_KEY")

    if not api_key:
        raise RuntimeError(
            "SERPAPI_KEY environment variable is not set."
        )

    return api_key


def search_images(category: str, api_key: str, num: int, location: str):
    """
    使用 SerpApi Google Images API 搜索图片。
    """

    results = []

    # Google Images 每页最多取一批结果
    pages = max(1, (num + 99) // 100)

    for page in range(pages):

        params = {
            "engine": "google_images",
            "q": category,
            "ijn": page,
            "location": location,
            "safe": "active",
            "api_key": api_key,
        }

        response = requests.get(
            "https://serpapi.com/search.json",
            params=params,
            timeout=30,
        )

        response.raise_for_status()

        data = response.json()

        # SerpApi 返回错误时直接抛出来
        if "error" in data:
            raise RuntimeError(
                f"SerpApi error: {data['error']}"
            )

        image_results = data.get(
            "images_results",
            []
        )

        if not image_results:
            break

        results.extend(image_results)

        if len(results) >= num:
            break

    return results[:num]

def calculate_md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def download_image(url: str, timeout: int = 15):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=timeout,
    )

    response.raise_for_status()

    return response.content


def validate_and_convert_image(data: bytes):
    """
    验证图片是否真的能被 Pillow 打开。

    最终统一转换成 RGB JPEG，
    避免后面的模型读取 PNG/WebP 时出现格式差异。
    """
    image = Image.open(io.BytesIO(data))

    image.load()

    if image.mode != "RGB":
        image = image.convert("RGB")

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=95,
    )

    return buffer.getvalue(), image.size


def save_metadata(output_dir: Path, metadata: list):
    metadata_path = output_dir / "metadata.json"

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )


def main():
    parser = build_parser()
    args = parser.parse_args()

    args.category = "百事可乐"
    args.num = 50
    output = "data/products"
    location = "Tokyo, Japan"


    if args.num <= 0:
        raise ValueError("--num must be greater than 0")

    api_key = get_api_key()

    output_dir = Path(args.output) / args.category
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Category : {args.category}")
    print(f"Target   : {args.num}")
    print(f"Output   : {output_dir}")

    print("\nSearching images...")

    search_results = search_images(
        category=args.category,
        api_key=api_key,
        num=args.num * 3,
        location=args.location,
    )

    print(f"Search results: {len(search_results)}")

    metadata = []

    downloaded = 0
    hashes = set()

    for result in search_results:

        if downloaded >= args.num:
            break

        image_url = result.get("original")

        if not image_url:
            continue

        try:
            raw_data = download_image(image_url)

            image_hash = calculate_md5(raw_data)

            # 去除完全重复的图片
            if image_hash in hashes:
                continue

            image_data, image_size = validate_and_convert_image(
                raw_data
            )

            image_hash = calculate_md5(image_data)

            if image_hash in hashes:
                continue

            hashes.add(image_hash)

            downloaded += 1

            filename = f"{downloaded:04d}.jpg"
            image_path = output_dir / filename

            image_path.write_bytes(image_data)

            metadata.append({
                "filename": filename,
                "product": args.category,
                "width": image_size[0],
                "height": image_size[1],
                "md5": image_hash,
                "image_url": image_url,
                "source_url": result.get("link"),
                "title": result.get("title"),
                "source": result.get("source"),
                "download_time": datetime.now(
                    timezone.utc
                ).isoformat(),
            })

            print(
                f"[{downloaded:03d}/{args.num}] "
                f"{filename} "
                f"{image_size[0]}x{image_size[1]}"
            )

        except Exception as e:
            print(
                f"[SKIP] {image_url}\n"
                f"       reason: {e}"
            )

    save_metadata(
        output_dir,
        metadata,
    )

    print("\nFinished.")
    print(f"Downloaded: {downloaded}")
    print(f"Directory : {output_dir}")
    print(f"Metadata  : {output_dir / 'metadata.json'}")


if __name__ == "__main__":

    main()
